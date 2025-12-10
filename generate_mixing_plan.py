# generate_mixing_plan.py

"""
DJ Mixing Plan Generator (Ready for Mixing Engine)

- Reads basic_setlist.json + structure_data.json
- Sorts songs globally by BPM
- Sliding 5-song window BPM averaging for smooth ramp
- Chorus Beatmatch for normal tracks
- Full-song + crossfade every 5th song
- Outputs mixing_plan.json with exact timings for fade-in/out
"""

import os
import json
from datetime import datetime, timedelta
import librosa

SONGS_DIR = "./songs"


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    return (datetime.min + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def get_chorus_duration(track: dict) -> float:
    return track.get("first_chorus_end", 90.0) - track.get("first_chorus_start", 60.0)


def select_tracks_in_order(basic_setlist: dict, structure_data: dict) -> list[dict]:
    """
    Merge basic setlist and analyzed structure data IN ORIGINAL ORDER (no BPM sorting).
    """
    all_tracks = []
    analyzed_dict = {}
    for segment in structure_data.get("analyzed_setlist", []):
        for track in segment.get("analyzed_tracks", []):
            analyzed_dict[track["title"]] = track

    for segment in basic_setlist.get("setlist", []):
        for track in segment.get("tracks", []):
            title = track["title"]
            analyzed_track = analyzed_dict.get(title)
            if analyzed_track:
                track_copy = analyzed_track.copy()
                track_copy["original_segment_time"] = segment.get("time", "Unknown")
                all_tracks.append(track_copy)
            else:
                print(f"Warning: '{title}' not found in structure data; skipping.")

    # NO SORTING - keep in OpenAI's optimal DJ order
    return all_tracks


def generate_mixing_plan(
    basic_setlist_path: str = "basic_setlist.json",
    structure_json_path: str = "structure_data.json",
    output_path: str = "mixing_plan.json",
    overlap_duration: float = 8.0,  # 8 seconds overlap at transition
    fade_duration: float = 1.0  # 1 second fade out
):
    try:
        basic_setlist = load_json(basic_setlist_path)
        structure_data = load_json(structure_json_path)

        all_tracks = select_tracks_in_order(basic_setlist, structure_data)

        mixing_plan = []
        last_start_sec = 0.0
        last_track = None

        for idx, track in enumerate(all_tracks):
            file_path = os.path.join(SONGS_DIR, track["file"])
            if not os.path.exists(file_path):
                print(f"Missing: {file_path}. Skipping.")
                continue

            try:
                duration_sec = librosa.get_duration(filename=file_path)
            except Exception as e:
                print(f"Error loading {track['file']}: {e}. Skipping.")
                continue

            if last_track is None:
                # First track - play from beginning
                start_sec = 0.0
                incoming_start_sec = 0.0
                transition_type = "Fade In"
                transition_point = None
                incoming_intro = None
                bpm_change_point = None
                comment = f"Start with {track['title']} (BPM {track.get('bpm', 'N/A')})"
            else:
                # Calculate when to start incoming track based on vocal overlap rules
                from_transition_point = last_track.get("transition_point", 70.0)
                to_intro_duration = track.get("intro_duration", 8.0)
                to_has_early_vocals = track.get("has_vocals_in_first_8s", False)
                
                # RULE: If incoming song has vocals in first 8s, previous transition must be at line end
                # Otherwise, transition can be 8s before line end
                # This prevents vocal overlap
                
                if to_has_early_vocals:
                    # Incoming has vocals in first 8s - transition at line end
                    # Start incoming based on intro duration
                    if to_intro_duration > 8.0:
                        incoming_start_offset = 8.0
                    else:
                        incoming_start_offset = to_intro_duration
                    
                    overlap_comment = "Line end transition (incoming has early vocals)"
                else:
                    # No vocals in first 8s - can start earlier
                    incoming_start_offset = 8.0
                    overlap_comment = "Standard 8s overlap (no early vocals)"
                
                incoming_start_sec = last_start_sec + from_transition_point - incoming_start_offset
                
                # BPM change happens 8s before transition
                bpm_change_point = last_start_sec + from_transition_point - 8.0
                
                transition_type = "Transition Overlap"
                transition_point = from_transition_point
                incoming_intro = to_intro_duration
                
                comment = (
                    f"{last_track['title']} (BPM {last_track.get('bpm', 120)}) -> {track['title']} "
                    f"(BPM {track.get('bpm', 120)}). {overlap_comment}. "
                    f"Transition at {from_transition_point:.1f}s, BPM change at {bpm_change_point - last_start_sec:.1f}s"
                )

            start_str = format_time(incoming_start_sec)

            mixing_plan.append(
                {
                    "from_track": last_track["title"] if last_track else None,
                    "to_track": track["title"],
                    "incoming_start_sec": incoming_start_sec,
                    "start_time": start_str,
                    "transition_point": transition_point,
                    "incoming_intro_duration": incoming_intro,
                    "bpm_change_point_sec": bpm_change_point,
                    "overlap_duration": overlap_duration,
                    "fade_duration": fade_duration,
                    "transition_type": transition_type,
                    "to_bpm": track.get("bpm", 120),
                    "from_bpm": last_track.get("bpm", 120) if last_track else None,
                    "comment": comment,
                }
            )

            last_start_sec = incoming_start_sec
            last_track = track

        with open(output_path, "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)

        print(f"Mixing plan saved to '{output_path}' with {len(mixing_plan)} tracks.")

    except Exception as e:
        print(f"Error generating mixing plan: {e}")
        raise


if __name__ == "__main__":
    generate_mixing_plan()
