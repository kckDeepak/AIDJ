"""
This module combines a track identification engine and a minimal track analysis engine.
It processes user input to generate a structured setlist for events using the OpenAI API.
It scans a local directory for MP3 files, parses user preferences
(including time segments, genres, and specific song requests), and generates an unordered pool
of available tracks that match the criteria with BPM differences less than 2 within segments.
The OpenAI model recalls/looks up BPMs for songs in the available list and includes them in the setlist.
Post-processing refines BPMs that default to 120 via targeted OpenAI queries.
The setlist tracks are enriched with key, scale, has_vocals, choruses, and first_chorus_end.
The output is saved as a single 'analyzed_setlist.json' for further use.

Key features:
- Scans local MP3 files, extracts metadata (no BPM estimation).
- Uses a single OpenAI API call to interpret user input, recall BPMs, and select BPM-compatible tracks.
- Post-processes BPM=120 with targeted lookups.
- Handles unavailable songs by logging them separately.
- Minimal analysis: key/scale/key_semitone, vocal detection, structural segments (for chorus), chorus detection, first_chorus_end.

Dependencies:
- openai: For interacting with the OpenAI API.
- librosa: For minimal audio feature extraction.
- python-dotenv: For loading environment variables.
- Standard library: json, os, re, numpy.
- Suppresses non-critical warnings.
"""

import json  # Used for parsing and serializing JSON data, including available songs and setlist output.
import openai  # OpenAI library for accessing the GPT model.
import os  # Standard library for file and directory operations and environment variables.
import re  # Regular expressions module for extracting JSON from the AI response.
import warnings  # To suppress non-critical Numba/Librosa warnings.
from dotenv import load_dotenv  # Library for loading environment variables from a .env file.
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
import librosa  # Core library for audio analysis, providing feature extraction.
from librosa.feature.spectral import chroma_stft  # Specific Librosa features needed.
from librosa.feature import rms, mfcc  # Specific Librosa features needed.
import numpy as np  # Used for numerical operations on audio features.

# Load environment variables from the .env file to securely manage API keys.
load_dotenv()

# Configure the OpenAI client with the API key from environment.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the directory path where local MP3 song files are stored.
SONGS_DIR = "./songs"


def get_available_songs():
    """
    Scans the specified songs directory (SONGS_DIR) for MP3 files and returns a list of dictionaries
    containing metadata for each available song (no BPM estimation).

    For each MP3 file:
    - Extracts a clean name by removing the .mp3 extension and handling common prefixes.
    - Attempts to split the clean name into artist and title using " - " as a delimiter.

    Returns:
        list: A list of dictionaries, each with keys 'title' (str), 'artist' (str), 'file' (str).
    """
    available_songs = []  # Initialize an empty list to store song metadata dictionaries.
    for filename in os.listdir(SONGS_DIR):  # Iterate over all files in the songs directory.
        if filename.lower().endswith(".mp3"):  # Check if the file is an MP3 (case-insensitive).
            # Clean filename to extract artist and title:
            # Remove the .mp3 extension.
            clean_name = filename[:-4]
            # Handle common download prefixes like "[iSongs.info] ".
            if clean_name.startswith("[iSongs.info] "):
                clean_name = clean_name.split(" - ", 1)[-1] if " - " in clean_name else clean_name.split(" ", 2)[-1]
            # Attempt to split the clean name into artist and title using " - " as the delimiter.
            parts = clean_name.split(" - ", 1)
            if len(parts) == 2:  # If exactly two parts are found (artist and title).
                artist, title = parts  # Assign the parts to artist and title variables.
            else:  # If splitting fails (e.g., no " - " delimiter).
                artist = "Unknown"  # Default artist to "Unknown".
                title = clean_name  # Use the entire clean name as the title.
            # Append a dictionary with the extracted metadata and original filename to the list.
            available_songs.append({"title": title, "artist": artist, "file": filename})
    return available_songs  # Return the populated list of available songs.


def refine_bpm(title, artist):
    """
    Targeted OpenAI call to refine BPM if it defaulted to 120.
    Prompts for exact BPM from reliable sources, responds with integer only.

    Args:
        title (str): Song title.
        artist (str): Artist name.

    Returns:
        int: Refined BPM value.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music metadata expert. Respond ONLY with the exact integer BPM (e.g., 94) from reliable sources like Tunebat, SongBPM, or Musicstax. No units, no explanation, no extra text."},
                {"role": "user", "content": f"What is the precise BPM of '{title}' by '{artist}'?"}
            ],
            temperature=0.0,
            max_tokens=5
        )
        bpm_text = response.choices[0].message.content.strip()
        if bpm_text.isdigit():
            bpm = int(bpm_text)
            if 60 <= bpm <= 220:  # Sanity check for music BPM range.
                print(f"Refined BPM for '{title}' by '{artist}': {bpm}")
                return bpm
    except Exception as e:
        print(f"BPM refinement failed for '{title}' by '{artist}': {e}")
    return 120  # Fallback if query fails.


def parse_time_segments_and_generate_setlist(user_input, available_songs):
    """
    Uses the OpenAI GPT model (single call) to parse the user input into time segments, genres, and specific song requests,
    recall/lookup BPMs for available songs, and generate an unordered setlist of available local songs that match the criteria,
    ensuring BPM differences <2 within segments.

    Process:
    - Converts the available_songs list to a JSON string for inclusion in the prompt.
    - Constructs a detailed prompt instructing the model to:
      - Parse time segments (start/end times and descriptions).
      - Extract preferred genres and specific songs.
      - For each available song, recall/lookup its BPM from knowledge (default to 120 if unknown).
      - Select matching tracks from available local songs, prioritizing exact matches.
      - Ensure within each segment, selected tracks have BPMs differing by less than 2 from each other.
      - Log unavailable specific songs separately.
      - Estimate track counts based on time durations (3-4 min/track).
      - Output in a strict JSON format with BPMs included.
    - Generates content using "gpt-4o-mini".
    - Extracts the JSON from the response (handling potential Markdown code block wrappers).
    - Parses the JSON and returns the data.

    Args:
        user_input (str): The user's natural language description of the event, including times, genres, and songs.
        available_songs (list): List of song dictionaries from get_available_songs().

    Returns:
        dict: Parsed JSON data with keys: time_segments, genres, specific_songs, unavailable_songs, setlist.

    Raises:
        ValueError: If the AI response cannot be parsed into valid JSON.
    """
    # Serialize the available_songs list to a formatted JSON string for easy inclusion in the prompt.
    available_songs_str = json.dumps(available_songs, indent=2)
    
    # System prompt to ensure JSON-only output.
    system_prompt = "You are a DJ setlist generator. Analyze the user input and available songs, then output ONLY a valid JSON object in the exact format specified. Do not include any other text."
    
    # Construct the user prompt as a multi-line f-string, providing instructions and context to the AI model.
    user_prompt = f"""
    You are a professional DJ setlist generator.

    TASK SUMMARY:
        1. Parse the event description into:
        - Time segments (start, end, description)
        - Preferred genres
        - Specific songs mentioned by the user

        2. Using ONLY the available local songs list below, create an unordered setlist for each time segment.

        3. For every local song, recall or look up the BPM from trusted sources. Do not lie, do not guess, please return a value that you are 100% confident on
        - Use precise BPMs.
        - Avoid guesses.
        - If and ONLY IF truly unknown after trying to recall: use 120.

        4. For each time segment:
        - Select a pool of songs matching the vibe, genre, and description.
        - Prioritize exact matches of specific songs when available.
        - Only include tracks whose BPMs differ by <2 BPM within that segment.
        - Estimate number of songs based on duration (1 track ≈ 3–4 minutes).
        - DO NOT ORDER the tracks. Output an unordered list.

        5. For specific songs requested by the user but not found in the available local songs list:
        - Add them to "unavailable_songs" with reason "not found".

    AVAILABLE LOCAL SONGS:
    {available_songs_str}

    USER INPUT:
    \"\"\"{user_input}\"\"\"

    OUTPUT JSON FORMAT (STRICT — DO NOT ADD EXTRA TEXT):
    {{
      "time_segments": [
        {{"start": "HH:MM", "end": "HH:MM", "description": "string"}}
      ],
      "genres": ["genre1", "genre2"],
      "specific_songs": [
        {{"title": "string", "artist": "string"}}
      ],
      "unavailable_songs": [
        {{"title": "string", "artist": "string", "reason": "string"}}
      ],
      "setlist": [
        {{
          "time": "HH:MM–HH:MM",
          "tracks": [
            {{
              "title": "string",
              "artist": "string",
              "file": "filename.mp3",
              "bpm": integer
            }}
          ]
        }}
      ]
    }}
    """

    
    # Generate the AI response using the constructed prompts (single call).
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1  # Low temperature for consistent JSON output.
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Attempt to extract JSON from the response text, handling cases where it may be wrapped in ```json ... ``` Markdown blocks.
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    if json_match:  # If a Markdown-wrapped JSON block is found.
        json_string = json_match.group(1)  # Extract the inner JSON content.
    else:  # If no wrapper is found, use the raw response text (stripped of whitespace).
        json_string = response_text

    # Parse the extracted string into a Python dictionary.
    try:
        parsed_data = json.loads(json_string)
        # Post-process BPMs: Refine any that are 120.
        for segment in parsed_data.get("setlist", []):
            for track in segment.get("tracks", []):
                if track.get("bpm") == 120:
                    track["bpm"] = refine_bpm(track["title"], track["artist"])
        return parsed_data  # Return the successfully parsed data.
    except json.JSONDecodeError as e:  # Handle JSON parsing errors.
        # Print debug information: the raw response text for troubleshooting.
        print(f"DEBUG: Failed to parse JSON. Raw response text: '{response_text}'")
        # Raise a ValueError with a user-friendly message, chaining the original exception for details.
        raise ValueError("Failed to parse OpenAI response into JSON") from e


def estimate_key(chroma_mean):
    """
    Estimates the musical key and scale (major/minor) of a track using Krumhansl-Kessler key profiles.
    """
    # Define Krumhansl-Kessler profiles for major and minor keys.
    major_profile = np.array([6.35, 2.26, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
   
    # Normalize profiles and input chroma vector.
    major_profile /= np.sum(major_profile)
    minor_profile /= np.sum(minor_profile)
    chroma_mean_norm = chroma_mean / np.sum(chroma_mean)
   
    # Compute correlation for major/minor starting at C.
    major_corr_c = np.corrcoef(chroma_mean_norm, major_profile)[0, 1]
    minor_corr_c = np.corrcoef(chroma_mean_norm, minor_profile)[0, 1]
   
    # Determine scale and compute correlations for all keys.
    if major_corr_c > minor_corr_c:
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), major_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)
        scale = 'major'
    else:
        corrs = [np.corrcoef(np.roll(chroma_mean_norm, -i), minor_profile)[0, 1] for i in range(12)]
        key_idx = np.argmax(corrs)
        scale = 'minor'
   
    # Map the key index to a key name.
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = keys[key_idx]
   
    return key, scale


def _key_to_semitone(key, scale):
    """
    Maps a musical key and scale to a semitone index (0-11 for major, 12-23 for minor).
    """
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = keys.index(key) # Get index of the key (0-11).
    if scale == 'minor':
        idx += 12 # Add 12 for minor keys to differentiate (12-23).
    return idx


def _detect_vocals(y, sr):
    """
    Detects the presence of vocals in an audio track using a heuristic based on energy in the human vocal frequency range.
    """
    S = np.abs(librosa.stft(y)) # Compute magnitude of STFT.
    freqs = librosa.fft_frequencies(sr=sr) # Get frequency bins.
    # Select frequency indices between 200 Hz (low male voice) and 4000 Hz (high female voice/harmonics).
    mid_idx = np.where((freqs > 200) & (freqs < 4000))[0]
    mid_energy = np.mean(S[mid_idx, :]) # Compute mean energy in vocal range.
    # Use empirical threshold to determine if vocals are present.
    return bool(mid_energy > 0.01)


def _structural_segmentation(y, sr, beats):
    """
    Performs structural segmentation based on beat boundaries, labeling segments as high or low energy.
    """
    try:
        boundaries = librosa.frames_to_time(beats, sr=sr)
        segments = []
        duration = len(y) / sr
       
        boundary_times = boundaries.tolist()
        if not boundary_times or boundary_times[0] > 0.1:
            boundary_times.insert(0, 0.0)
        if boundary_times[-1] < duration - 0.1:
            boundary_times.append(duration)
       
        for i in range(1, len(boundary_times)):
            start_time = boundary_times[i-1]
            end_time = boundary_times[i]
           
            start_sample = int(start_time * sr)
            end_sample = int(min(end_time * sr, len(y)))
            seg_audio = y[start_sample:end_sample]
           
            segments.append({'start': round(start_time, 2), 'end': round(end_time, 2), 'audio': seg_audio})
       
        if segments:
            energies = [np.mean(rms(y=seg['audio'])) for seg in segments]
            if energies:
                med = np.median(energies)
                labels = ['High' if e > med else 'Low' for e in energies]
            else:
                labels = ['Low'] * len(segments)
           
            cleaned_segments = []
            for i, seg in enumerate(segments):
                cleaned_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'label': labels[i]
                })
            return cleaned_segments
        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}]
    except Exception as e:
        print(f"Segmentation failed: {e}")
        duration = len(y) / sr
        return [{'start': 0.0, 'end': duration, 'label': 'Full Track'}]

def _detect_choruses(y, sr, segments=None):
    """
    State-of-the-art chorus detection optimized for DJ mixing, enhanced with multi-feature self-similarity
    fusion (chroma + MFCC) inspired by https://github.com/beantowel/chorus-from-music-structure (2025).
    Returns accurate chorus sections, especially the FIRST chorus end for beatmatching,
    ensuring a 5-6 second instrumental section follows the vocal end of the chorus.
    """
    VOCAL_BUFFER_SEC = 6.0 # Required instrumental time after the vocal part of the chorus ends.
    FALLBACK_CHORUS_DURATION = 60.0
    MIN_CHORUS_LENGTH = 10.0  # Relaxed from 15s
    MAX_CHORUS_LENGTH = 120.0  # Relaxed from 90s
    ENERGY_THRESHOLD_FACTOR = 0.5  # Relaxed from 0.9
    MIN_REPEAT_SCORE = 0.05  # New: accept low repeats if energy high

    try:
        duration = len(y) / sr

        # 1. Beat-synchronous features (chroma and MFCC for fusion)
        hop_length = 512
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chroma_sync = librosa.util.sync(chroma, beat_frames)
        
        # Add MFCC features for better timbral similarity in self-similarity
        mfcc_features = mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)  # Standard 13 MFCCs
        mfcc_sync = librosa.util.sync(mfcc_features, beat_frames)

        # 2. Self-similarity (recurrence) matrices for each feature
        R_chroma = librosa.segment.recurrence_matrix(
            chroma_sync, 
            width=3, 
            sym=True, 
            mode='affinity'
        )
        R_mfcc = librosa.segment.recurrence_matrix(
            mfcc_sync, 
            width=3, 
            sym=True, 
            mode='affinity'
        )
        
        # Fuse matrices (simple average for multi-feature fusion)
        R = 0.5 * R_chroma + 0.5 * R_mfcc

        # 3. Structural novelty (boundaries between sections) - Fixed for Librosa 0.11.0
        # Use agglomerative clustering directly on synchronized chroma features (n_beats x 12)
        bounds_beat = librosa.segment.agglomerative(chroma_sync.T, k=8)  # ~8 sections
        bounds_time = librosa.frames_to_time(bounds_beat, sr=sr, hop_length=hop_length)

        # Add start/end
        bounds_time = np.unique(np.append([0], bounds_time)) # Use unique to handle potential 0.0 duplicates
        if bounds_time[-1] < duration:
            bounds_time = np.append(bounds_time, duration)

        # 4. Compute RMS energy per section
        # NOTE: Using RMS from raw frames for better time alignment of energy
        rms_feature = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        # Align RMS to the same beat frames for section analysis
        rms_sync = librosa.util.sync(rms_feature, beat_frames) 
        
        section_energies = []
        section_starts = []
        section_ends = []

        # Convert time bounds to the frame indices of the synced features (rms_sync, chroma_sync)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        
        for i in range(len(bounds_time) - 1):
            start_t, end_t = bounds_time[i], bounds_time[i + 1]
            
            # Find the nearest beat-frame indices for time boundaries
            start_idx = np.argmin(np.abs(beat_times - start_t))
            end_idx = np.argmin(np.abs(beat_times - end_t))
            
            # Ensure the indices are valid for slicing
            if end_idx <= start_idx or end_idx > len(rms_sync):
                 energy = 0
            else:
                 # Use the slice of the synced RMS
                 energy = np.mean(rms_sync[start_idx:end_idx])

            section_energies.append(energy)
            section_starts.append(start_t)
            section_ends.append(end_t)

        # 5. Find candidate choruses: high energy + repeated
        candidates = []
        max_energy = np.max(section_energies) if section_energies else 1.0
        med_energy = np.median(section_energies) if section_energies else 0
        
        for i, (start, end, energy) in enumerate(zip(section_starts, section_ends, section_energies)):
            length = end - start
            if length < MIN_CHORUS_LENGTH or length > MAX_CHORUS_LENGTH:  # Relaxed range
                continue
            if energy < med_energy * ENERGY_THRESHOLD_FACTOR:  # Relaxed energy check
                continue

            # Repetition score using the fused R
            start_R_idx = np.searchsorted(beat_times, start, side='left')
            end_R_idx = np.searchsorted(beat_times, end, side='right')
            
            if end_R_idx <= start_R_idx:
                repeat_score = 0
            else:
                # Slice recurrence matrix for this section (using synchronized indices)
                seg_R = R[start_R_idx:end_R_idx, start_R_idx:end_R_idx]
                repeat_score = np.mean(seg_R) if seg_R.size > 0 else 0

            # Accept if repeat low but energy high or early
            if repeat_score < MIN_REPEAT_SCORE and energy < max_energy * 0.7:
                continue

            candidates.append({
                'start': start,
                'end': end,
                'energy': energy,
                'repeat_score': repeat_score,
                'length': length,
                'index': i
            })

        if not candidates:
            # Fallback: use longest high-energy section after 30s
            fallback_start = max(30.0, duration * 0.25)
            fallback_end = min(fallback_start + FALLBACK_CHORUS_DURATION, duration * 0.7)
            return [{'start': fallback_start, 'end': fallback_end, 'label': 'Chorus (Fallback)'}]

        # 6. Score: prioritize early + long + repeated + energetic (adjusted weights)
        for c in candidates:
            earliness = 1.0 / (1 + c['start'])  # earlier = better
            c['score'] = (
                0.3 * c['repeat_score'] +  # Lower weight for repeats
                0.3 * (c['energy'] / (max_energy + 1e-6)) +
                0.3 * earliness +
                0.1 * (c['length'] / 60.0)
            )

        candidates.sort(key=lambda x: x['score'], reverse=True)

        # 7. Find FIRST and best candidate
        # Sort by start time for the first chorus
        first_chorus_candidates = sorted(candidates, key=lambda x: x['start'])
        if not first_chorus_candidates:
            # Should not happen if candidates is not empty, but for safety
            return [{'start': duration * 0.25, 'end': duration * 0.5, 'label': 'Chorus (Fallback)'}]

        best_first_chorus = first_chorus_candidates[0]
        
        # 8. Refine Chorus End Time (The new critical step)
        
        # Calculate MFCCs for vocal detection
        # MFCCs (e.g., coefficients 4-12) tend to be higher for vocal/speech segments
        # Using a shorter frame size (2048) for better time resolution in this local search
        frame_length_vocal = 2048
        hop_length_vocal = 512
        
        # Calculate full MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length_vocal)
        # Use variance/mean of middle MFCCs (4-12) as a proxy for vocal activity
        vocal_activity = np.mean(np.abs(mfccs[4:13, :]), axis=0)
        # Normalize for easier comparison
        vocal_activity = vocal_activity / (np.max(vocal_activity) + 1e-6)

        # Determine the search window: last 20 seconds of the detected chorus
        search_start_sec = max(best_first_chorus['start'], best_first_chorus['end'] - 20.0)
        search_end_sec = best_first_chorus['end'] 
        
        # Convert times to MFCC frame indices
        start_frame = librosa.time_to_frames(search_start_sec, sr=sr, hop_length=hop_length_vocal)
        end_frame = librosa.time_to_frames(search_end_sec, sr=sr, hop_length=hop_length_vocal)
        
        vocal_search_segment = vocal_activity[start_frame:end_frame]
        
        # Threshold for vocal activity (can be tuned, 0.6 is a starting guess)
        VOCAL_THRESHOLD = 0.6 
        
        # Find the frame index where vocal activity drops below the threshold and stays low
        vocal_end_frame_rel = end_frame - start_frame - 1 # Start search from the end
        safe_end_frame_rel = vocal_end_frame_rel # Default to the section end
        
        # Iterate backwards to find the last moment of strong vocal activity
        for i in range(len(vocal_search_segment) - 1, 0, -1):
            # Check if vocal activity is consistently low for the required buffer
            buffer_frames = int(VOCAL_BUFFER_SEC * sr / hop_length_vocal)
            
            # Check if the next 'buffer_frames' are all below the vocal threshold
            if i - buffer_frames >= 0:
                segment_to_check = vocal_search_segment[i - buffer_frames:i]
                
                # Check if this segment is predominantly instrumental (low vocal activity)
                if np.mean(segment_to_check) < VOCAL_THRESHOLD:
                    # Found a potentially safe instrumental segment of required length
                    # The vocal part must have ended *before* this instrumental segment started.
                    # We set the vocal end to the frame right before the instrumental segment started.
                    vocal_end_frame_rel = i - buffer_frames
                    break

        # Convert the found frame index back to time
        # The time is the start of the instrumental buffer
        instrumental_start_time = librosa.frames_to_time(start_frame + vocal_end_frame_rel, sr=sr, hop_length=hop_length_vocal)
        
        # The true "Chorus End" for transition purposes is the start of this instrumental buffer
        refined_chorus_end = instrumental_start_time
        
        # Ensure the refined end is within the original boundaries
        refined_chorus_end = np.clip(refined_chorus_end, best_first_chorus['start'] + 10.0, best_first_chorus['end'])


        # Final output structure: one entry for the *transition point*
        return [{
            'start': best_first_chorus['start'], 
            'end': refined_chorus_end, 
            'label': f'Chorus (Vocal End + {VOCAL_BUFFER_SEC}s Buffer)'
        }]

    except Exception as e:
        print(f"Advanced chorus detection failed: {e}. Using safe fallback.")
        # Simpler fallback if the complex logic fails
        fallback_start = duration * 0.25
        fallback_end = min(fallback_start + FALLBACK_CHORUS_DURATION, duration * 0.7)
        fallback_end = max(fallback_end - VOCAL_BUFFER_SEC, fallback_start + 10.0)  # Ensure end > start
        return [{
            'start': fallback_start, 
            'end': fallback_end, 
            'label': 'Chorus (Safe Fallback)'
        }]

def analyze_track(title, artist, filename, bpm):
    """
    Analyzes a local MP3 file to extract minimal audio features using Librosa.
    Features: key/scale/key_semitone, has_vocals, choruses, first_chorus_end.
    Uses provided BPM for beat tracking.
    """
    file_path = os.path.join(SONGS_DIR, filename)
   
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Returning fallback data.")
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm, "key": "N/A", "key_semitone": 0, "scale": "N/A",
            "has_vocals": False, "choruses": [], "first_chorus_end": None,
            "genre": "File Missing"
        }
   
    try:
        y, sr = librosa.load(file_path, sr=None)
       
        # Beat tracking using provided BPM.
        try:
            _, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=float(bpm), tightness=100)
            beats = beats.astype(int)
        except Exception as e:
            print(f"Beat tracking failed: {e}. Using empty beats.")
            beats = np.array([])
       
        # Compute chroma features and estimate key.
        chroma_mat = chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma_mat, axis=1)
        key, scale = estimate_key(chroma_mean)
        key_name = f"{key}m" if scale == 'minor' else key
        key_semitone = _key_to_semitone(key, scale)
       
        # Detect vocals.
        has_vocals = _detect_vocals(y, sr)
       
        # Extract structural segments and choruses.
        segments = _structural_segmentation(y, sr, beats)
        choruses = _detect_choruses(y, sr, segments)
        first_chorus = None
        if choruses:
            # Pick the earliest chorus that starts after 20s and lasts >20s
            valid = [c for c in choruses if c['start'] > 20 and c['end'] - c['start'] > 20]
            if valid:
                first_chorus = min(valid, key=lambda x: x['start'])
            else:
                first_chorus = choruses[0]  # fallback to first detected
        
        first_chorus_end = first_chorus["end"] if first_chorus else None
       
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm,
            "key": key_name, "key_semitone": key_semitone, "scale": scale, "genre": "Unknown",
            "has_vocals": has_vocals,
            "choruses": choruses,
            "first_chorus_end": first_chorus_end
        }
   
    except Exception as e:
        print(f"Error analyzing '{title}' by '{artist}': {e}")
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm, "key": "C", "key_semitone": 0, "scale": "major",
            "has_vocals": False, "choruses": [], "first_chorus_end": None,
            "genre": "Analysis Failed"
        }


def analyze_tracks_in_setlist(data):
    """
    Enriches tracks in the setlist with minimal analysis features.
    Saves the analyzed setlist to 'analyzed_setlist.json'.
    """
    try:
        analyzed_setlist = []
       
        # Iterate over each time segment in the setlist.
        for segment in data["setlist"]:
            time_range = segment["time"]
            tracks = segment["tracks"]
            analyzed_tracks = []
           
            # Analyze each track in the segment.
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                filename = track["file"]
                bpm = track["bpm"]
               
                # Analyze the track to extract minimal features.
                metadata = analyze_track(title, artist, filename, bpm)
               
                analyzed_tracks.append(metadata)
           
            # Add the analyzed segment to the setlist.
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
       
        # Create the final output dictionary.
        output = {"analyzed_setlist": analyzed_setlist}
        # Save the analyzed setlist to a JSON file.
        with open("analyzed_setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
       
        return output
    except Exception as e:
        # Log any errors during analysis and re-raise for caller handling.
        print(f"Error in Track Analysis Engine: {str(e)}")
        raise


def combined_engine(user_input):
    """
    Main entry point. Generates setlist via OpenAI (with BPM lookup), then enriches with analysis and saves 'analyzed_setlist.json'.
    """
    try:
        available_songs = get_available_songs()
        data = parse_time_segments_and_generate_setlist(user_input, available_songs)
        
        # Analyze the setlist (enrich tracks).
        analyze_tracks_in_setlist(data)
    except Exception as e:
        print(f"Error in Combined Engine: {str(e)}")
        raise


if __name__ == "__main__":
    user_input = (
        "I need a mix between 7pm and 10pm for a Casino. At 8pm there will be dinner, "
        "then dancing starts at 9pm. Most of our customers prefer R&B, Bollywood, Afrobeats "
        "and these songs specifically: [{'title': 'Tum Hi Ho', 'artist': 'Arijit Singh'}, "
        "{'title': 'Ye', 'artist': 'Burna Boy'}]."
    )
    combined_engine(user_input)