"""
Structure Timestamp Detector Module.
Uses OpenAI Whisper for transcription and GPT-4o for detecting transition point (end of main part/chorus) for DJ transitions.
Standalone for transition point detection requiring audio. Outputs has_vocals and transition_point.
"""

import os
import json
from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
OpenAI = OpenAI if OpenAI else None

load_dotenv()

client = None
if OpenAI is not None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None


def clean_json_output(text: str) -> str:
    """Strip code fences from GPT output."""
    return text.replace("```json", "").replace("```", "").strip()


def transcribe_song(client, audio_path):
    """Transcribe song using Whisper with word and segment timestamps."""
    print(f"🔊 Transcribing: {os.path.basename(audio_path)}")
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"]
        )
    return result.model_dump()


def detect_transition_points(client, title, artist, bpm, transcript_dict):
    """Analyze transcript for transition point using GPT-4o."""
    duration = transcript_dict.get('duration', 180.0)
    text = transcript_dict.get('text', '')[:2000]  # Truncate long text

    prompt = f"""
You are an expert music DJ assistant.

Song: '{title}' by '{artist}', BPM: {bpm}, Duration: {duration}s

Transcript with timestamps:
{json.dumps(transcript_dict, indent=2)}

Lyrics excerpt:
{text}

TASK: Identify the best transition point out of the song (end of the most energetic/catchy main part, usually first chorus), typically 40-90s in.
This is where DJs cut to the next track for seamless mixing.

Output ONLY valid JSON:
{{
  "transition_point": 0.0,
  "description": "Brief note on why this is the best transition point"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You identify precise DJ transition points from transcripts. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    text_resp = response.choices[0].message.content
    cleaned = clean_json_output(text_resp)

    try:
        data = json.loads(cleaned)
        # Ensure numeric type
        data['transition_point'] = float(data['transition_point'])
        return data
    except json.JSONDecodeError:
        print("⚠️ GPT returned invalid JSON:")
        print(text_resp)
        return None


def analyze_structure(title, artist, filename, bpm, SONGS_DIR="./songs"):
    """
    Analyzes MP3 via Whisper + GPT for transition point and has_vocals.
    Returns minimal data for mixing transitions.
    """
    file_path = os.path.join(SONGS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Returning fallback data.")
        return {
            "has_vocals": False,
            "transition_point": 60.0
        }
    try:
        transcript = transcribe_song(client, file_path)
        has_vocals = bool(transcript.get('text', '').strip())
        analysis = detect_transition_points(client, title, artist, bpm, transcript)
        if analysis is None:
            raise ValueError("Transition point detection failed")
        transition_point = analysis['transition_point']
        print(f"Detected transition point for {title} by {artist}: {transition_point}s")
        return {
            "has_vocals": has_vocals,
            "transition_point": transition_point
        }
    except Exception as e:
        print(f"Error analyzing structure for '{title}' by '{artist}': {e}")
        return {
            "has_vocals": False,
            "transition_point": 60.0
        }


if __name__ == "__main__":
    # Example usage
    title = "Tum Hi Ho"
    artist = "Arijit Singh"
    filename = "example.mp3"
    bpm = 120
    structure_data = analyze_structure(title, artist, filename, bpm)
    print(json.dumps(structure_data, indent=2))