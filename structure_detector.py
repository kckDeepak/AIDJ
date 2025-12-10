"""
Structure Timestamp Detector Module - OPTIMIZED VERSION
Analyzes songs for DJ transitions using OpenAI Whisper API only (no local model).
Detects transition_point and intro_duration for mixing.

SPEED OPTIMIZATIONS:
1. Uses OpenAI Whisper API (no local model loading)
2. Skips word-level timestamps (not needed for transition detection)
3. Reduces audio analysis to first 90 seconds only
4. Parallel processing ready (can be added if needed)
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

try:
    from openai import OpenAI
    import librosa
except Exception:
    OpenAI = None
    librosa = None

load_dotenv()

client = None
if OpenAI is not None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None

SONGS_DIR = "./songs"
NOTES_DIR = "./notes"

# Create notes directory if it doesn't exist
if not os.path.exists(NOTES_DIR):
    os.makedirs(NOTES_DIR)


def get_cache_path(filename):
    """Generate cache file path for a song's structure data."""
    # Use filename without extension as cache key
    base_name = os.path.splitext(filename)[0]
    # Sanitize filename for use as cache file
    import re
    safe_name = re.sub(r'[^\w\s-]', '_', base_name)
    return os.path.join(NOTES_DIR, f"{safe_name}_structure.json")


def load_cached_structure(filename):
    """Load cached structure analysis if available."""
    cache_path = get_cache_path(filename)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  ✓ Loaded cached structure from notes/")
                return data
        except Exception as e:
            print(f"  ⚠ Cache read failed: {e}")
    return None


def save_cached_structure(filename, structure_data):
    """Save structure analysis to cache for future use."""
    cache_path = get_cache_path(filename)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(structure_data, f, indent=2)
        print(f"  ✓ Cached structure to notes/")
    except Exception as e:
        print(f"  ⚠ Cache write failed: {e}")


def clean_json_output(text: str) -> str:
    """Strip code fences from GPT output."""
    return text.replace("```json", "").replace("```", "").strip()


def transcribe_song_fast(client, audio_path):
    """Transcribe song using OpenAI Whisper API (FAST - no local model)."""
    print(f"🔊 Transcribing: {os.path.basename(audio_path)}")
    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]  # Only segment-level, not word-level
        )
    return result.model_dump()


def extract_beat_times_fast(audio_path, max_duration=90):
    """Extract beat timestamps - OPTIMIZED: only first 90 seconds."""
    if librosa is None:
        return np.array([]), 120.0
    
    try:
        # Only load first 90 seconds for speed
        y, sr = librosa.load(audio_path, sr=22050, duration=max_duration)  # Lower sample rate = faster
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return beat_times, tempo
    except Exception as e:
        print(f"Beat detection failed: {e}")
        return np.array([]), 120.0


def ask_gpt4o_for_transition_point_fast(segments, beats_str, title, artist, duration):
    """
    Ask GPT to detect transition point and check for vocals in first 8 seconds.
    Returns transition point, intro duration, and whether vocals exist in first 8s.
    """
    # Format segments with timestamps
    lyrics_formatted = []
    has_early_vocals = False
    
    for seg in segments[:20]:  # Only first 20 segments (~first 2 minutes)
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        lyrics_formatted.append(f"[{start:.1f}s] {text}")
        
        # Check if there are vocals in first 8 seconds
        if start < 8.0 and text and len(text) > 10:  # Meaningful lyrics
            has_early_vocals = True
    
    lyrics_text = "\n".join(lyrics_formatted)
    
    prompt = f"""You are a professional DJ. Analyze this song for DJ mixing.

Song: "{title}" by {artist}
Duration: {duration:.0f}s

Lyrics with segment timestamps:
{lyrics_text}

Beat timestamps (first 100 beats): [{beats_str}]

CRITICAL RULES:
1. Check if there are vocals/lyrics in the FIRST 8 SECONDS
2. If vocals exist in first 8s:
   - Transition point = END OF A LYRIC LINE (where singer pauses/breathes)
   - This prevents vocal overlap when next song starts
3. If NO vocals in first 8s:
   - Transition point = 8 seconds BEFORE a line end
   - This allows incoming instrumental intro to blend
4. Transition point should be between 50-120 seconds
5. Intro duration = time before main vocals start (0-20s range)
6. Align to beat timestamps for smooth mixing

Return JSON:
{{
  "has_vocals_in_first_8s": <boolean>,
  "transition_point_sec": <float 50-120>,
  "intro_duration_sec": <float 0-20>,
  "transition_is_line_end": <boolean>,
  "reasoning": "<brief explanation>"
}}"""
    
    client_local = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client_local.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    result = json.loads(clean_json_output(response.choices[0].message.content))
    
    # Add Python-detected early vocals as backup
    if "has_vocals_in_first_8s" not in result:
        result["has_vocals_in_first_8s"] = has_early_vocals
    
    return result


def analyze_structure_fast(title, artist, filename, bpm, SONGS_DIR="./songs"):
    """
    OPTIMIZED structure analysis:
    - Uses OpenAI Whisper API (no local model loading)
    - Only processes first 90s of audio
    - Uses segment-level timestamps (not word-level)
    - Uses gpt-4o-mini for speed
    - Caches results in notes/ folder for reuse
    """
    # Check cache first
    cached = load_cached_structure(filename)
    if cached and "transition_point" in cached:
        print(f"  ✓ Using cached structure data")
        return cached
    
    file_path = os.path.join(SONGS_DIR, filename)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Using fallback.")
        return {
            "has_vocals": False, 
            "transition_point": 70.0,
            "intro_duration": 8.0
        }

    try:
        # Step 1: Fast transcription (OpenAI API - no local model)
        transcript = transcribe_song_fast(client, file_path)
        has_vocals = bool(transcript.get("text", "").strip())
        duration = transcript.get("duration", 180.0)
        segments = transcript.get("segments", [])
        
        # Step 2: Fast beat extraction (only first 90s, lower sample rate)
        beats, tempo = extract_beat_times_fast(file_path, max_duration=90)
        beats_str = ", ".join(f"{t:.1f}" for t in beats[:100])  # Only first 100 beats
        
        # Step 3: Fast GPT analysis (gpt-4o-mini, segment-level only)
        result = ask_gpt4o_for_transition_point_fast(segments, beats_str, title, artist, duration)
        transition_point = float(result.get("transition_point_sec", 70.0))
        intro_duration = float(result.get("intro_duration_sec", 8.0))
        has_vocals_in_first_8s = result.get("has_vocals_in_first_8s", False)
        transition_is_line_end = result.get("transition_is_line_end", True)
        
        # Step 4: Beat alignment (if beats available)
        if len(beats) > 0:
            # Find nearest beat to transition point
            transition_candidates = beats[(beats >= transition_point - 5) & (beats <= transition_point + 5)]
            if len(transition_candidates) > 0:
                transition_point = float(transition_candidates[np.argmin(np.abs(transition_candidates - transition_point))])
            
            # Find nearest beat to intro end
            intro_candidates = beats[(beats >= intro_duration - 2) & (beats <= intro_duration + 2)]
            if len(intro_candidates) > 0:
                intro_duration = float(intro_candidates[np.argmin(np.abs(intro_candidates - intro_duration))])
        
        # Step 5: Enforce constraints
        transition_point = float(np.clip(transition_point, 50.0, min(120.0, duration - 10.0)))
        intro_duration = float(np.clip(intro_duration, 0.0, 20.0))
        
        print(f"  ✓ Transition: {transition_point:.1f}s, Intro: {intro_duration:.1f}s")
        print(f"    Vocals in first 8s: {has_vocals_in_first_8s}, Line end: {transition_is_line_end}")
        
        structure_data = {
            "has_vocals": has_vocals, 
            "transition_point": transition_point,
            "intro_duration": intro_duration,
            "has_vocals_in_first_8s": has_vocals_in_first_8s,
            "transition_is_line_end": transition_is_line_end
        }
        transition_point = float(np.clip(transition_point, 50.0, min(120.0, duration - 10.0)))
        intro_duration = float(np.clip(intro_duration, 0.0, 20.0))
        
        print(f"  ✓ Transition: {transition_point:.1f}s, Intro: {intro_duration:.1f}s")
        
        structure_data = {
            "has_vocals": has_vocals, 
            "transition_point": transition_point,
            "intro_duration": intro_duration
        }
        
        # Save to cache
        save_cached_structure(filename, structure_data)
        
        return structure_data
        
    except Exception as e:
        print(f"Error analyzing '{title}': {e}")
        return {
            "has_vocals": False, 
            "transition_point": 70.0,
            "intro_duration": 8.0
        }


def process_structure_detection(input_json: str = "basic_setlist.json", output_json: str = "structure_data.json"):
    """
    Process all songs with OPTIMIZED structure detection.
    """
    if client is None:
        print("[WARN] OpenAI client not configured. Using fallback times.")
    
    try:
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        structure_data = {"analyzed_setlist": []}
        
        for segment in data.get("setlist", []):
            time_slot = segment.get("time", "00:00")
            analyzed_tracks = []
            
            for track in segment.get("tracks", []):
                title = track.get("title", "Unknown")
                artist = track.get("artist", "Unknown")
                filename = track.get("file", "")
                bpm = track.get("bpm", 120)
                
                print(f"🎵 Analyzing: {title} by {artist}")
                
                # FAST structure detection
                structure = analyze_structure_fast(title, artist, filename, bpm, SONGS_DIR)
                
                # Merge data
                analyzed_track = track.copy()
                analyzed_track.update(structure)
                analyzed_tracks.append(analyzed_track)
            
            structure_data["analyzed_setlist"].append({
                "time": time_slot,
                "analyzed_tracks": analyzed_tracks
            })
        
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(structure_data, f, indent=2)
        
        print(f"\n✅ Structure detection complete. Saved to '{output_json}'.")
        return structure_data
    
    except Exception as e:
        print(f"[ERROR] Structure detection failed: {e}")
        raise


if __name__ == "__main__":
    process_structure_detection("basic_setlist.json", "structure_data.json")