"""
BPM Lookup Module.
Provides OpenAI-based BPM refinement for songs using text prompts.
Standalone for metadata lookup without audio processing.
"""

import os
import json
import re
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


def get_genre(title, artist):
    """
    OpenAI lookup for song genre.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music metadata expert. Respond ONLY with the primary genre (e.g., R&B) from reliable sources. No explanation, no extra text."},
                {"role": "user", "content": f"What is the primary genre of '{title}' by '{artist}'?"}
            ],
            temperature=0.0,
            max_tokens=10
        )
        genre = response.choices[0].message.content.strip()
        if genre:
            print(f"Genre for '{title}' by '{artist}': {genre}")
            return genre
    except Exception as e:
        print(f"Genre lookup failed for '{title}' by '{artist}': {e}")
    return "Unknown"


def estimate_key(title, artist):
    """
    OpenAI lookup for song key and scale.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a music metadata expert. Respond ONLY with 'Key: C Scale: major' format from reliable sources like Tunebat. No extra text."},
                {"role": "user", "content": f"What is the key and scale of '{title}' by '{artist}'?"}
            ],
            temperature=0.0,
            max_tokens=20
        )
        text = response.choices[0].message.content.strip()
        match = re.search(r'Key: (\w+) Scale: (\w+)', text)
        if match:
            key, scale = match.groups()
            print(f"Key for '{title}' by '{artist}': {key} {scale}")
            return key, scale
    except Exception as e:
        print(f"Key lookup failed for '{title}' by '{artist}': {e}")
    return "C", "major"


def _key_to_semitone(key, scale):
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    idx = keys.index(key)
    if scale == 'minor':
        idx += 12
    return idx


if __name__ == "__main__":
    # Example usage
    title = "Tum Hi Ho"
    artist = "Arijit Singh"
    bpm = refine_bpm(title, artist)
    genre = get_genre(title, artist)
    key, scale = estimate_key(title, artist)
    key_semitone = _key_to_semitone(key, scale)
    print(f"BPM: {bpm}, Genre: {genre}, Key: {key} {scale}, Semitone: {key_semitone}")