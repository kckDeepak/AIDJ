# run_pipeline.py
"""
This module orchestrates the full AI DJ pipeline, integrating multiple stages to generate a DJ mix from user input.
It coordinates the setlist generation, BPM/metadata lookup, structure/timestamp detection, mixing plan generation,
and final mix creation, producing a seamless MP3 mix based on user preferences and local audio files.

Key features:
- Executes multiple stages: setlist generation, BPM/metadata lookup, structure detection, combination, mixing plan, and mix generation.
- Sorts tracks globally by BPM ascending in mixing plan for progressive build-up.
- Prioritizes Chorus Beatmatch at transition point for all transitions (if available); uses Crossfade/EQ only if BPM diff >3.
- Uses logging to track progress and errors for debugging and monitoring.
- Validates the existence of intermediate output files (JSON and MP3) to ensure pipeline integrity.
- Handles errors gracefully, logging issues and raising exceptions for caller handling.

Dependencies:
- json: For reading/writing/combining intermediate JSON files.
- os: For file path operations.
- logging: For detailed logging of pipeline execution.
- bpm_lookup: Module for OpenAI-based BPM, genre, key lookup.
- structure_detector: Module for Whisper+GPT-4o transition point analysis.
- generate_mixing_plan: Creates a mixing plan with transition types and timings (BPM-sorted; modified to accept metadata.json and structure.json).
- mixing_engine: Generates the final MP3 mix.
"""

import json  # Used for reading/writing/combining intermediate JSON files.
import os  # Used for file path operations.
import logging  # Used for logging pipeline progress and errors.
from dotenv import load_dotenv  # For loading OpenAI API key.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
# Import from new modules
from bpm_lookup import refine_bpm, get_genre, estimate_key, _key_to_semitone  # BPM/metadata lookup functions.
from structure_detector import analyze_structure  # Structure/timestamp detection.
from generate_mixing_plan import generate_mixing_plan  # Mixing plan generator (modified for two JSON inputs).
from mixing_engine import generate_mix  # Mix generator.

load_dotenv()  # Load environment variables for OpenAI.

print("This is happening: Starting the AI DJ Pipeline execution...")  # Startup message in terminal.

# Setup logging with DEBUG level and a custom format including timestamp, level, and message.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('colorlogger')  # Create a logger instance named 'colorlogger'.

# Define the directory path where local MP3 song files are stored (relative to the script's execution directory).
SONGS_DIR = "./songs"

# Configure OpenAI client (shared for lookups and transcription).
client = None
if OpenAI is not None:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        client = None


def parse_time_segments_and_generate_setlist(user_input, available_songs):
    """
    Uses OpenAI to parse user input into setlist with BPMs, then refines defaults.
    (Extracted from original combined engine.)
    """
    import re  # For JSON extraction.
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
        # Refine BPMs if defaulted to 120.
        for segment in parsed_data.get("setlist", []):
            for track in segment.get("tracks", []):
                if track.get("bpm") == 120:
                    track["bpm"] = refine_bpm(track["title"], track["artist"])
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"DEBUG: Failed to parse JSON. Raw response: '{response_text}'")
        raise ValueError("Failed to parse OpenAI response into JSON") from e


def get_available_songs():
    """
    Scans the specified songs directory (SONGS_DIR) for MP3 files and returns a list of dictionaries
    containing metadata for each available song.
    (Extracted from original.)
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


def add_metadata_to_setlist(setlist_data):
    """
    Adds genre, key, scale, key_semitone to each track in the setlist using OpenAI lookups.
    Saves to 'metadata.json'.
    """
    metadata_setlist = []
    for segment in setlist_data["setlist"]:
        time_range = segment["time"]
        meta_tracks = []
        for track in segment["tracks"]:
            title = track["title"]
            artist = track["artist"]
            filename = track["file"]
            bpm = track["bpm"]
            genre = get_genre(title, artist)
            key, scale = estimate_key(title, artist)
            key_semitone = _key_to_semitone(key, scale)
            key_name = f"{key}m" if scale == 'minor' else key
            meta_track = {
                "title": title, "artist": artist, "file": filename,
                "bpm": bpm, "key": key_name, "key_semitone": key_semitone, "scale": scale, "genre": genre
            }
            meta_tracks.append(meta_track)
        metadata_setlist.append({"time": time_range, "tracks": meta_tracks})
    output = {"metadata_setlist": metadata_setlist}
    with open("metadata.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Metadata added and saved to 'metadata.json'")
    return output


def add_structure_to_setlist(setlist_data):
    """
    Adds transition_point, has_vocals to each track using Whisper+GPT analysis.
    Saves to 'structure.json'.
    """
    structure_setlist = []
    for segment in setlist_data["setlist"]:
        time_range = segment["time"]
        struct_tracks = []
        for track in segment["tracks"]:
            title = track["title"]
            artist = track["artist"]
            filename = track["file"]
            bpm = track["bpm"]
            struct_data = analyze_structure(title, artist, filename, bpm, SONGS_DIR)
            struct_track = {
                "title": title, "artist": artist, "file": filename,
                "has_vocals": struct_data["has_vocals"],
                "transition_point": struct_data["transition_point"]
            }
            struct_tracks.append(struct_track)
        structure_setlist.append({"time": time_range, "tracks": struct_tracks})
    output = {"structure_setlist": structure_setlist}
    with open("structure.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Structure added and saved to 'structure.json'")
    return output


def combine_metadata_and_structure(metadata_data, structure_data):
    """
    Combines metadata.json and structure.json into full analyzed_setlist.json.
    Matches tracks by title/artist/file.
    """
    analyzed_setlist = []
    for meta_seg, struct_seg in zip(metadata_data["metadata_setlist"], structure_data["structure_setlist"]):
        if meta_seg["time"] != struct_seg["time"]:
            raise ValueError("Time segments mismatch between metadata and structure.")
        time_range = meta_seg["time"]
        analyzed_tracks = []
        for meta_track, struct_track in zip(meta_seg["tracks"], struct_seg["tracks"]):
            if meta_track["title"] != struct_track["title"] or meta_track["artist"] != struct_track["artist"]:
                raise ValueError("Track mismatch between metadata and structure.")
            full_track = {**meta_track, **struct_track}  # Merge dicts
            analyzed_tracks.append(full_track)
        analyzed_setlist.append({"time": time_range, "analyzed_tracks": analyzed_tracks})
    output = {"analyzed_setlist": analyzed_setlist}
    with open("analyzed_setlist.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Combined analyzed setlist saved to 'analyzed_setlist.json'")
    return output


def run_pipeline(user_input):
    """
    Runs the full AI DJ pipeline to generate a DJ mix from user input.

    Process:
    1. Setlist Generation: Parses user input to create basic setlist with refined BPMs.
    2. BPM/Metadata Lookup: Adds genre, key, etc., to tracks; saves metadata.json.
    3. Structure/Timestamp Detection: Adds transition_point, has_vocals via Whisper+GPT; saves structure.json.
    4. Combination: Merges metadata.json and structure.json into analyzed_setlist.json.
    5. Mixing Plan: Creates BPM-sorted mixing plan from the two JSONs (or combined).
    6. Mix Generation: Produces MP3 mix with applied transitions.
    - Validates intermediate files.
    - Logs progress and errors.

    Args:
        user_input (str): User description of the mix.

    Raises:
        FileNotFoundError: If expected files not created.
        Exception: Other errors, logged and re-raised.
    """
    try:
        # Step 1: Generate basic setlist with refined BPMs.
        logger.info("Generating basic setlist...")
        available_songs = get_available_songs()
        setlist_data = parse_time_segments_and_generate_setlist(user_input, available_songs)
        with open("basic_setlist.json", "w") as f:
            json.dump(setlist_data, f, indent=2)
        if not os.path.exists("basic_setlist.json"):
            raise FileNotFoundError("basic_setlist.json not created.")
        logger.info("Basic setlist generated successfully.")

        # Step 2: Add metadata (BPM already refined; add genre/key).
        logger.info("Adding metadata (genre, key)...")
        metadata_data = add_metadata_to_setlist(setlist_data)
        if not os.path.exists("metadata.json"):
            raise FileNotFoundError("metadata.json not created.")
        logger.info("Metadata lookup completed.")

        # Step 3: Add structure/timestamps.
        logger.info("Analyzing structure and timestamps...")
        structure_data = add_structure_to_setlist(setlist_data)
        if not os.path.exists("structure.json"):
            raise FileNotFoundError("structure.json not created.")
        logger.info("Structure analysis completed.")

        # Step 4: Combine into analyzed_setlist.
        logger.info("Combining metadata and structure...")
        analyzed_data = combine_metadata_and_structure(metadata_data, structure_data)
        if not os.path.exists("analyzed_setlist.json"):
            raise FileNotFoundError("analyzed_setlist.json not created.")
        logger.info("Full analysis combined successfully.")

        # Step 5: Generate BPM-sorted mixing plan from metadata.json and structure.json.
        logger.info("Running Mixing Plan Generator...")
        generate_mixing_plan("metadata.json", "structure.json", eq_match_ms=15000)  # Modified to accept two JSONs.
        if not os.path.exists("mixing_plan.json"):
            raise FileNotFoundError("mixing_plan.json not created.")
        logger.info("BPM-sorted mixing plan generated successfully.")

        # Step 6: Generate final MP3 mix.
        logger.info("Running Mix Generator...")
        with open("analyzed_setlist.json", "r") as f:
            analyzed_setlist_json = f.read()
        generate_mix(analyzed_setlist_json, "mixing_plan.json", eq_match_ms=15000)
        if not os.path.exists("mix.mp3"):
            raise FileNotFoundError("mix.mp3 not created.")

        logger.info("Pipeline complete. Check 'analyzed_setlist.json', 'mixing_plan.json', and 'mix.mp3'.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    user_input = (
        "Create a mix between all songs"
    )
    run_pipeline(user_input)