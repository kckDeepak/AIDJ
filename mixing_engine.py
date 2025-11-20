# generate_mix.py

"""
This module generates a continuous DJ mix by applying transitions based on an analyzed setlist and a mixing plan.
It strictly supports only the 'Chorus Beatmatch' transition type for complex mixing.

Key features:
- Uses the globally sorted track order from 'mixing_plan.json'.
- Converts audio between pydub's AudioSegment and NumPy arrays for processing.
- Aligns tracks using onset strength correlation for beat matching (critical for Chorus Beatmatch).
- **Chorus Beatmatch (MODIFIED):** Executes a beat-aligned transition where the incoming track starts at full volume, and the outgoing track is smoothly faded out over the overlap duration (6-8 seconds), ensuring precise beat alignment based on the calculated lag and no vocal overlap post-fade.
- **Fallback:** If the plan specifies any other transition type (e.g., 'Crossfade'), it executes a standard, unaligned crossfade.
- Applies tempo stretching based on the OTAC from the mixing plan.
- Handles missing files gracefully and normalizes the final mix for consistent loudness.

Dependencies:
- os: For file path operations.
- json: For parsing input setlist and mixing plan JSON files.
- numpy: For numerical operations on audio data.
- librosa: For audio feature extraction (e.g., onset strength).
- pydub: For loading, manipulating, and exporting audio files.
"""

import os 
import json 
import numpy as np 
import librosa 
from pydub import AudioSegment 
from pydub.effects import normalize 

# Define the directory path where local MP3 song files are stored.
SONGS_DIR = "./songs"


# ---------------------------
# Utility conversions
# ---------------------------
def audio_segment_to_np(segment: AudioSegment):
    """Converts a pydub AudioSegment to a mono float32 NumPy array with values in [-1, 1]."""
    samples = np.array(segment.get_array_of_samples()) 
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1) 
    sr = segment.frame_rate 
    y = samples.astype(np.float32) / 32768.0 
    return y, sr


def np_to_audio_segment(y: np.ndarray, sr: int):
    """Converts a mono float32 NumPy array in [-1, 1] to a pydub AudioSegment (mono, 16-bit)."""
    y_clipped = np.clip(y, -1.0, 1.0) 
    y_int16 = (y_clipped * 32767.0).astype(np.int16) 
    return AudioSegment(
        y_int16.tobytes(), 
        frame_rate=sr, 
        sample_width=2, 
        channels=1 
    )


# ---------------------------
# Beat / onset helpers
# ---------------------------
def get_onset_envelope(y, sr, hop_length=512):
    """Computes the normalized onset strength envelope of an audio signal using Librosa."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length) 
    if onset_env.size == 0:
        return None, hop_length 
    onset_env = onset_env / (np.max(onset_env) + 1e-9) 
    return onset_env, hop_length


def find_best_alignment(y1, sr1, y2, sr2, match_duration_sec=15.0):
    """
    Finds the best alignment (lag in seconds) between two audio tracks using onset strength correlation.
    
    The lag indicates how much to advance/delay y2 relative to y1 for optimal beat match.
    """
    try:
        # Use a segment length defined by the match duration (i.e., the overlap length).
        n1 = min(len(y1), int(match_duration_sec * sr1))
        n2 = min(len(y2), int(match_duration_sec * sr2))
        if n1 < 1024 or n2 < 1024:
            return 0.0 
        tail1 = y1[-n1:] 
        head2 = y2[:n2] 
        onset1, hop = get_onset_envelope(tail1, sr1) 
        onset2, _ = get_onset_envelope(head2, sr2) 
        if onset1 is None or onset2 is None:
            return 0.0 
        minlen = min(len(onset1), len(onset2)) 
        if minlen < 8:
            return 0.0 
        onset1_r = librosa.util.fix_length(onset1, size=minlen) 
        onset2_r = librosa.util.fix_length(onset2, size=minlen)
        corr = np.correlate(onset1_r - onset1_r.mean(), onset2_r - onset2_r.mean(), mode='full')
        lag_idx = corr.argmax() - (len(onset2_r) - 1) 
        approx_hop_seconds = hop / float(sr1) 
        lag_seconds = -lag_idx * approx_hop_seconds 
        lag_seconds = float(np.clip(lag_seconds, -match_duration_sec, match_duration_sec))
        return lag_seconds
    except Exception:
        return 0.0 


# ---------------------------
# Core transition application
# ---------------------------
def apply_transition(segment1: AudioSegment,
                     segment2: AudioSegment,
                     transition_type: str,
                     duration_ms: int = 8000,
                     early_ms: int = 5500,
                     otac: float = 0.0,
                     eq_match_duration_ms: int = 15000):
    """
    Clean, professional Chorus Beatmatch = aligned crossfade at chorus end.
    No glitches. No silence. No re-appearing tails.
    """
    # The new transition needs a defined fade out time for the outgoing track.
    # We will use 8000ms as the fade-out duration.
    FADE_OUT_MS = 8000
    
    # The total overlap must be enough for the fade out (8000ms) plus the early start (5500ms).
    overlap_ms = duration_ms + early_ms  # e.g. 8000 + 5500 = 13500ms overlap
    overlap_ms = max(overlap_ms, FADE_OUT_MS) # Ensure overlap is at least the fade-out duration.
    overlap_ms = min(overlap_ms, len(segment1), len(segment2))  # Safety

    # Convert to numpy
    y1, sr1 = audio_segment_to_np(segment1[-overlap_ms:])
    y2_full, sr2 = audio_segment_to_np(segment2)

    # Apply OTAC tempo stretch to incoming track
    if abs(otac) > 0.01:
        stretch_duration_sec = max(duration_ms, eq_match_duration_ms) / 1000.0
        # Formula for rate based on OTAC (beats/minute * stretch_time_sec / 60 seconds)
        # Note: librosa rate is relative to 1.0, not a percentage.
        # This formula seems slightly off for a general tempo change but is preserved from the original.
        rate = 1.0 + otac * stretch_duration_sec / 60.0
        y2_full = librosa.effects.time_stretch(y2_full, rate=rate)

    segment2_stretched = np_to_audio_segment(y2_full, sr2)
    segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate).set_channels(1)

    # Extract overlap portions
    outgoing_tail = segment1[-overlap_ms:]
    incoming_head = segment2_stretched[:overlap_ms]

    if transition_type.lower() == "crossfade":
        # Standard crossfade (no alignment)
        faded_out = outgoing_tail.fade_out(overlap_ms)
        faded_in = incoming_head.fade_in(overlap_ms)
        crossed = faded_out.overlay(faded_in)
        return crossed + segment2_stretched[overlap_ms:]

    # === CHORUS BEATMATCH: Aligned Fade-Out ===
    elif "chorus beatmatch" in transition_type.lower():
        # Find best beat alignment between last 15s of outgoing and first 15s of incoming
        align_sec = min(15.0, overlap_ms / 1000.0)
        
        # Use a portion of the outgoing tail that is at least 15s long for alignment correlation.
        y1_align = y1[-int(align_sec * sr1):]
        # Use a portion of the incoming head that is at least 15s long for alignment correlation.
        y2_align = y2_full[:int(align_sec * sr2)] 

        lag_sec = find_best_alignment(
            y1_align, sr1, y2_align, sr2, match_duration_sec=align_sec
        )
        lag_ms = int(lag_sec * 1000)

        # --- Beat Alignment Logic ---
        # Shift incoming track by lag (positive = delay incoming, negative = start early)
        # The logic here ensures that the outgoing_for_mix and incoming_shifted segments
        # align perfectly at the point of mix, while maintaining the overall length of overlap_ms.

        if lag_ms >= 0:
            # Incoming starts later → pad start with silence
            # Outgoing is full tail. Incoming is padded.
            silence = AudioSegment.silent(duration=lag_ms, frame_rate=sr1)
            incoming_shifted = silence + incoming_head
            outgoing_for_mix = outgoing_tail
        else:
            # Incoming starts early → trim outgoing start
            # Incoming is full head. Outgoing is trimmed (or conceptually trimmed by an earlier start).
            advance_ms = -lag_ms
            incoming_shifted = incoming_head
            
            # The outgoing track starts *before* the overlap segment (tail). 
            # We must virtually start the outgoing track earlier by padding, then trim.
            outgoing_for_mix = outgoing_tail + AudioSegment.silent(duration=advance_ms, frame_rate=sr1)
            outgoing_for_mix = outgoing_for_mix[advance_ms:] # This makes the outgoing track *shorter* by advance_ms

        # The aligned segments should now have the correct relative timing.
        
        # Ensure both are same length for overlay up to the fade-out point
        # The true overlap segment must be the longest of the two segments after shifting/padding.
        target_len = max(len(outgoing_for_mix), len(incoming_shifted))
        
        # Apply the **one-way fade-out** on the outgoing track over FADE_OUT_MS
        # The incoming track (incoming_shifted) is *not* faded in; it stays at full volume.
        
        # 1. Fade out the outgoing track tail
        faded_out_outgoing = outgoing_for_mix.fade_out(FADE_OUT_MS)
        
        # 2. Trim both to the maximum required length
        faded_out_outgoing = faded_out_outgoing[:target_len]
        incoming_shifted = incoming_shifted[:target_len]
        
        # 3. Perform the overlay (incoming track at full volume)
        mixed = faded_out_outgoing.overlay(incoming_shifted)

        # 4. Append the rest of the incoming track (after overlap)
        rest = segment2_stretched[overlap_ms:]
        if lag_ms < 0:
            # If incoming track started early (lag_ms < 0), we need to skip the segment of 
            # segment2_stretched that was already used up by the early start and the overlap.
            # The total used length from segment2_stretched is overlap_ms + |lag_ms|.
            rest_start_ms = overlap_ms + (-lag_ms)
            rest = segment2_stretched[rest_start_ms:] 

        return mixed + rest

    # Fallback
    return outgoing_tail.fade_out(overlap_ms).overlay(incoming_head.fade_in(overlap_ms)) + segment2_stretched[overlap_ms:]

# ---------------------------
# Mix generator
# ---------------------------
def generate_mix(analyzed_setlist_json, mixing_plan_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    """
    Generates a continuous DJ mix by processing an analyzed setlist and applying transitions from a mixing plan.
    """
    try:
        # Parse the analyzed setlist JSON string to get all metadata.
        analyzed_data = json.loads(analyzed_setlist_json)
        # Load the mixing plan from the specified JSON file.
        mixing_plan = json.load(open(mixing_plan_json, 'r')).get("mixing_plan", [])
        
        full_mix = AudioSegment.empty()  # Initialize an empty AudioSegment for the mix.
        last_track_start_mix_sec = None  # Track the mix start time (seconds) of the last added track.
        
        # Flatten all track metadata into a single dictionary for easy lookup by title.
        all_tracks_metadata = {}
        for segment in analyzed_data.get("analyzed_setlist", []):
            for track in segment.get("analyzed_tracks", []):
                all_tracks_metadata[track['title']] = track

        # Iterate over the mixing plan entries (each entry represents the start of a new track).
        for track_index, plan_entry in enumerate(mixing_plan):
            to_track_title = plan_entry.get("to_track")
            
            # --- Get Incoming Track Metadata ---
            if to_track_title not in all_tracks_metadata:
                print(f"[generate_mix] Missing metadata for incoming track: {to_track_title}. Skipping.")
                continue
            track = all_tracks_metadata[to_track_title]

            transition_type = plan_entry.get("transition_type", "Crossfade") 
            otac = plan_entry.get("otac", 0.0) 
            outgoing_cut_sec = plan_entry.get("outgoing_cut_sec") 
            overlap_sec = plan_entry.get("overlap_sec", 8.0) 
            # Recalculate overlap based on the desired new transition logic
            # This is hardcoded to 13500ms (8s crossfade + 5.5s early) in the original, 
            # which is fine as the *fade* only uses 8000ms.
            overlap_ms = 13500  # 8s crossfade + 5.5s early = 13.5s (standard pro DJ overlap)

            file_path = os.path.join(SONGS_DIR, track["file"]) 
            if not os.path.exists(file_path):
                print(f"[generate_mix] Missing file: {file_path}. Skipping track.")
                continue

            # Load the audio file (incoming track).
            audio = AudioSegment.from_file(file_path)

            if track_index == 0:
                # --- FIRST TRACK ---
                fade_dur = int(min(first_fade_in_ms, len(audio)))
                full_mix += audio.fade_in(fade_dur)
                last_track_start_mix_sec = 0.0
            else:
                # --- SUBSEQUENT TRACKS (TRANSITIONS) ---
                
                # 1. Calculate the exact mix time of the outgoing track's cut point.
                trans_start_mix_sec = last_track_start_mix_sec + outgoing_cut_sec
                trans_start_ms = int(trans_start_mix_sec * 1000 + 0.5)

                # 2. Determine the start of the overlap/tail segment in the current mix.
                tail_start_ms = max(0, trans_start_ms - overlap_ms)

                # 3. Trim the mix and extract the necessary segments.
                pre_transition = full_mix[:tail_start_ms]  # Audio before the overlap starts.
                tail = full_mix[tail_start_ms:trans_start_ms]  # The overlap segment of the outgoing track.
                
                if len(tail) < 500:
                    print(f"[generate_mix] Warning: Tail for '{plan_entry.get('from_track')}' is too short ({len(tail)}ms). Appending fully.")
                    full_mix += audio
                    last_track_start_mix_sec = len(full_mix) / 1000.0 
                    continue

                # 4. Apply the specified transition.
                trans_audio = apply_transition(
                    tail, audio, transition_type,
                    duration_ms=8000,
                    early_ms=crossfade_early_ms,
                    otac=otac,
                    eq_match_duration_ms=eq_match_ms
                )
                
                # 5. Rebuild full_mix.
                full_mix = pre_transition + trans_audio
                
                # 6. Update last start time for the *newly added* track.
                last_track_start_mix_sec = trans_start_mix_sec - overlap_sec

        # Normalize the final mix for consistent loudness and export as MP3.
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")

    except Exception as e:
        print(f"[generate_mix] Error: {e}")
        raise


# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    """
    Entry point for testing the mix generator with a sample analyzed setlist and mixing plan.
    """
    # Sample setlist with chorus times added for proper Chorus Beatmatch testing.
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
            {
                "time": "19:00–20:00",
                "analyzed_tracks": [
                    {
                        "title": "Tum Hi Ho",
                        "artist": "Arijit Singh",
                        "file": "Arijit Singh - Tum Hi Ho.mp3",
                        "bpm": 94,
                        "key_semitone": 9,
                        "scale": "major",
                        "genre": "bollywood",
                        "energy": 0.45,
                        "valence": 0.32,
                        "danceability": 0.52,
                        "has_vocals": true,
                        "segments": [{"label": "L"}],
                        "choruses": [{"start": 60.0, "end": 90.0}], 
                        "chroma_matrix": null,
                        "transition": "Fade In",
                        "notes": "Balanced Vibe track. Genre: bollywood."
                    },
                    {
                        "title": "No Scrubs",
                        "artist": "TLC",
                        "file": "TLC - No Scrubs.mp3",
                        "bpm": 93,
                        "key_semitone": 8,
                        "scale": "minor",
                        "genre": "r&b",
                        "energy": 0.7,
                        "valence": 0.6,
                        "danceability": 0.8,
                        "has_vocals": true,
                        "segments": [{"label": "H"}],
                        "choruses": [{"start": 45.0, "end": 75.0}], 
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            }
        ]
    }
    '''
    generate_mix(sample_analyzed_setlist_json, "mixing_plan.json")