"""
DJ Mixing Pipeline: Setlist Generation + OpenAI Analysis.
Handles MP3 scanning, OpenAI setlist parsing, BPM refinement, and full song analysis via Whisper transcription + GPT-4o.
Analysis covers key, vocals, genre, structure segments, and chorus detection from lyrics/transcript.
Outputs 'analyzed_setlist.json' with segments and first chorus times.
"""

import os
import json
import re
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    # OpenAI client might not be installed in local dev/test environments.
    OpenAI = None
OpenAI = OpenAI if OpenAI else None

# Load environment variables from the .env file to securely manage API keys.
load_dotenv()

# Configure the OpenAI client with the API key from environment.
client = None
if OpenAI is not None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None

# Define the directory path where local MP3 song files are stored.
SONGS_DIR = "./songs"


def clean_json_output(text: str) -> str:
    """Strip code fences from GPT output."""
    return text.replace("```json", "").replace("```", "").strip()


def get_available_songs():
    """
    Scans the specified songs directory (SONGS_DIR) for MP3 files and returns a list of dictionaries
    containing metadata for each available song (no BPM estimation).
    """
    available_songs = []
    for filename in os.listdir(SONGS_DIR):
        if filename.lower().endswith(".mp3"):
            clean_name = filename[:-4]
            if clean_name.startswith("[iSongs.info] "):
                clean_name = clean_name.split(" - ", 1)[-1] if " - " in clean_name else clean_name.split(" ", 2)[-1]
            parts = clean_name.split(" - ", 1)
            if len(parts) == 2:
                artist, title = parts
            else:
                artist = "Unknown"
                title = clean_name
            available_songs.append({"title": title, "artist": artist, "file": filename})
    return available_songs


def refine_bpm(title, artist):
    """
    Targeted OpenAI call to refine BPM if it defaulted to 120.
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
            if 60 <= bpm <= 220:
                print(f"Refined BPM for '{title}' by '{artist}': {bpm}")
                return bpm
    except Exception as e:
        print(f"BPM refinement failed for '{title}' by '{artist}': {e}")
    return 120


def parse_time_segments_and_generate_setlist(user_input, available_songs):
    """
    Uses OpenAI to parse user input into setlist with BPMs, then refines defaults.
    """
    available_songs_str = json.dumps(available_songs, indent=2)
    system_prompt = "You are a DJ setlist generator. Analyze the user input and available songs, then output ONLY a valid JSON object in the exact format specified. Do not include any other text."
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
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    response_text = response.choices[0].message.content.strip()
    json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    json_string = json_match.group(1) if json_match else response_text
    try:
        parsed_data = json.loads(json_string)
        for segment in parsed_data.get("setlist", []):
            for track in segment.get("tracks", []):
                if track.get("bpm") == 120:
                    track["bpm"] = refine_bpm(track["title"], track["artist"])
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON. Raw response: '{response_text}'")
        raise ValueError("Failed to parse OpenAI response into JSON") from e


def _key_to_semitone(key, scale):
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = keys.index(key)
    if scale == 'minor':
        idx += 12
    return idx


def transcribe_song(client, audio_path):
    """Transcribe song using Whisper."""
    print(f"🔊 Transcribing: {os.path.basename(audio_path)}")
    audio_file = open(audio_path, "rb")
    try:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=(os.path.basename(audio_path), audio_file, "audio/mpeg"),
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
        return result.model_dump()
    finally:
        audio_file.close()


def detect_structure(client, title, artist, bpm, transcript_dict):
    """Analyze transcript for structure, metadata using GPT-4o."""
    duration = transcript_dict.get('duration', 180.0)
    segments_data = transcript_dict.get('segments', [])
    text = transcript_dict.get('text', '')[:2000]  # Truncate long text

    prompt = f"""
You are an expert music analyst and DJ assistant.

Song: '{title}' by '{artist}', BPM: {bpm}, Duration: {duration}s

Transcript segments:
{json.dumps(segments_data, indent=2)}

Lyrics excerpt:
{text}

TASK: Analyze for DJ mixing. Use song knowledge for metadata, transcript timestamps for structure.

- Genre: e.g., "R&B" or "Bollywood"
- Key: e.g., "C" (from reliable recall, no guess)
- Scale: "major" or "minor"
- Segments: 5-10 chronological sections (intro, verse, chorus, bridge, outro, break). Use transcript timestamps for accurate start/end (seconds).
  - Label: "intro", "verse", "chorus", "bridge", "outro", "break"
  - Energy: -1 (low) to 1 (high), e.g., high in choruses
  - Repetition: 0-1 (high for repeated lyrics)
  - Combined: 0-1 (overall score, high for choruses)
- Choruses: Up to 3, derived from segments (start/end from segments, label "Chorus 1", etc.)
- First chorus: Earliest chorus starting >15s with duration >15s. End: start + 30s or to next section start.

Output ONLY valid JSON (no extra text):
{{
  "genre": "string",
  "key": "C",
  "scale": "major",
  "segments": [
    {{"label": "intro", "start": 0.0, "end": 15.0, "energy": -0.5, "repetition": 0.2, "combined": 0.3}},
    {{"label": "chorus", "start": 45.0, "end": 75.0, "energy": 0.8, "repetition": 0.9, "combined": 0.85}}
  ],
  "choruses": [
    {{"start": 45.0, "end": 75.0, "label": "Chorus 1"}}
  ],
  "first_chorus_start": 45.0,
  "first_chorus_end": 75.0
}}
Ensure segments chain (end_n = start_{n+1}), cover 0 to duration.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise music structure analyzer. Output only valid JSON matching the format exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    text_resp = response.choices[0].message.content
    cleaned = clean_json_output(text_resp)

    try:
        data = json.loads(cleaned)
        # Ensure numeric types
        for seg in data.get('segments', []):
            for k in ['start', 'end', 'energy', 'repetition', 'combined']:
                if k in seg:
                    seg[k] = float(seg[k])
        for ch in data.get('choruses', []):
            for k in ['start', 'end']:
                ch[k] = float(ch[k])
        data['first_chorus_start'] = float(data['first_chorus_start'])
        data['first_chorus_end'] = float(data['first_chorus_end'])
        return data
    except json.JSONDecodeError:
        print("⚠️ GPT returned invalid JSON:")
        print(text_resp)
        return None


def analyze_track(title, artist, filename, bpm):
    """
    Analyzes MP3 via Whisper + GPT for key, vocals, structure (segments), choruses, first_chorus_start/end.
    """
    file_path = os.path.join(SONGS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Returning fallback data.")
        duration = 180.0  # assume 3 min
        fallback_segments = [
            {"label": "intro", "start": 0.0, "end": 30.0, "energy": -0.5, "repetition": 0.0, "combined": 0.2},
            {"label": "chorus", "start": 60.0, "end": 90.0, "energy": 0.8, "repetition": 0.8, "combined": 0.7},
            {"label": "outro", "start": 150.0, "end": duration, "energy": 0.0, "repetition": 0.0, "combined": 0.1}
        ]
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm, "key": "N/A", "key_semitone": 0, "scale": "N/A",
            "has_vocals": False, "segments": fallback_segments,
            "choruses": [{"start": 60.0, "end": 90.0, "label": "Chorus 1"}],
            "first_chorus_start": 60.0, "first_chorus_end": 90.0,
            "genre": "File Missing"
        }
    try:
        transcript = transcribe_song(client, file_path)
        has_vocals = bool(transcript.get('text', '').strip())
        analysis = detect_structure(client, title, artist, bpm, transcript)
        if analysis is None:
            raise ValueError("Structure detection failed")
        key = analysis['key']
        scale = analysis['scale']
        key_semitone = _key_to_semitone(key, scale)
        genre = analysis.get('genre', 'Unknown')
        segments = analysis['segments']
        choruses = analysis['choruses']
        first_chorus_start = analysis['first_chorus_start']
        first_chorus_end = analysis['first_chorus_end']
        key_name = f"{key}m" if scale == 'minor' else key
        print(f"Detected segments for {title} by {artist}: {[s['label'] for s in segments]}")
        chorus_times = [(c['start'], c['end']) for c in choruses]
        print(f"  Choruses at: {chorus_times}")
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm,
            "key": key_name, "key_semitone": key_semitone, "scale": scale, "genre": genre,
            "has_vocals": has_vocals,
            "segments": segments,
            "choruses": choruses,
            "first_chorus_start": first_chorus_start,
            "first_chorus_end": first_chorus_end
        }
    except Exception as e:
        print(f"Error analyzing '{title}' by '{artist}': {e}")
        duration = 180.0  # fallback duration
        fallback_segments = [
            {"label": "intro", "start": 0.0, "end": 30.0, "energy": -0.5, "repetition": 0.0, "combined": 0.2},
            {"label": "chorus", "start": 60.0, "end": 90.0, "energy": 0.8, "repetition": 0.8, "combined": 0.7},
            {"label": "outro", "start": 150.0, "end": duration, "energy": 0.0, "repetition": 0.0, "combined": 0.1}
        ]
        return {
            "title": title, "artist": artist, "file": filename,
            "bpm": bpm, "key": "C", "key_semitone": 0, "scale": "major",
            "has_vocals": False, "segments": fallback_segments,
            "choruses": [{"start": 60.0, "end": 90.0, "label": "Chorus 1"}],
            "first_chorus_start": 60.0,
            "first_chorus_end": 90.0,
            "genre": "Analysis Failed"
        }


def analyze_tracks_in_setlist(data):
    """
    Enriches setlist with analysis and saves to JSON.
    """
    try:
        analyzed_setlist = []
        for segment in data["setlist"]:
            time_range = segment["time"]
            tracks = segment["tracks"]
            analyzed_tracks = []
            for track in tracks:
                title = track["title"]
                artist = track["artist"]
                filename = track["file"]
                bpm = track["bpm"]
                metadata = analyze_track(title, artist, filename, bpm)
                analyzed_tracks.append(metadata)
            analyzed_setlist.append({
                "time": time_range,
                "analyzed_tracks": analyzed_tracks
            })
        output = {"analyzed_setlist": analyzed_setlist}
        with open("analyzed_setlist.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Analyzed setlist saved to 'analyzed_setlist.json'")
        return output
    except Exception as e:
        print(f"Error in Track Analysis Engine: {str(e)}")
        raise


def combined_engine(user_input):
    """
    Main entry point.
    """
    try:
        available_songs = get_available_songs()
        data = parse_time_segments_and_generate_setlist(user_input, available_songs)
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