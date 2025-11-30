# generate_mixing_plan.py

"""
This module generates a DJ mixing plan based on metadata and structure JSON inputs.
It strictly enforces 'Chorus Beatmatch' transitions where possible (BPM diff <= 2, both tracks have transition points, harmonic compatibility).
Fallbacks to 'Crossfade' otherwise. Prioritizes a chain of compatible tracks at the start, sorted globally by BPM ascending.

Key features:
- Merges metadata (BPM, key, genre) and structure (transition_point, has_vocals) data per track.
- Computes OTAC for tempo adjustments, harmonic compatibility, and transition timings.
- Generates plan with start times, cut points, overlaps, and comments.
- Prevents self-transitions and skips missing files.

Dependencies:
- json: For loading inputs and saving plan.
- os: For file paths.
- numpy: For OTAC calculations.
- datetime, timedelta: For time formatting.
- pydub: For audio durations.
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from pydub import AudioSegment

# Directory for song files.
SONGS_DIR = "./songs"


def load_and_merge_data(metadata_path: str, structure_path: str):
    """
    Loads metadata.json and structure.json, merges them into analyzed setlist format.
    Matches segments by time, tracks by title/artist.
    """
    with open(metadata_path, "r") as f:
        metadata_data = json.load(f)
    with open(structure_path, "r") as f:
        structure_data = json.load(f)

    analyzed_setlist = []
    for meta_seg, struct_seg in zip(
        metadata_data["metadata_setlist"], structure_data["structure_setlist"]
    ):
        if meta_seg["time"] != struct_seg["time"]:
            raise ValueError("Time segments mismatch between metadata and structure.")
        time_range = meta_seg["time"]
        analyzed_tracks = []
        for meta_track, struct_track in zip(meta_seg["tracks"], struct_seg["tracks"]):
            if (
                meta_track["title"] != struct_track["title"]
                or meta_track["artist"] != struct_track["artist"]
            ):
                raise ValueError("Track mismatch between metadata and structure.")
            full_track = {**meta_track, **struct_track}
            analyzed_tracks.append(full_track)
        analyzed_setlist.append({"time": time_range, "analyzed_tracks": analyzed_tracks})

    return {"analyzed_setlist": analyzed_setlist}


def get_estimated_overlap(transition_type: str, crossfade_early_ms: int = 5500, duration_ms: int = 8000, eq_match_ms: int = 15000) -> float:
    """
    Returns overlap in seconds based on transition type.
    """
    t = transition_type.lower()
    if "fade in" in t:
        return 0.0
    if "chorus beatmatch" in t:
        return 16.0  # Vocal/instrumental separation fade.
    return (duration_ms + crossfade_early_ms) / 1000.0


def is_harmonic_key(from_key_semitone: int | None, to_key_semitone: int | None) -> bool:
    """
    Checks harmonic compatibility using semitone differences.
    """
    compatible_shifts = [0, 1, 11, 7, 5]
    if from_key_semitone is None or to_key_semitone is None:
        return True
    key_diff = abs(int(from_key_semitone) - int(to_key_semitone)) % 12
    return key_diff in compatible_shifts or (12 - key_diff) in compatible_shifts


def compute_otac(song1_data: dict, song2_data: dict) -> float:
    """
    Computes Optimal Tempo Adjustment Coefficient (OTAC) as log(tempo2 / tempo1) / 60.
    """
    tempo1 = song1_data.get("bpm", 0)
    tempo2 = song2_data.get("bpm", 0)
    try:
        if tempo1 <= 0 or tempo2 <= 0:
            return 0.0
        return np.log(float(tempo2) / float(tempo1)) / 60.0
    except Exception:
        return 0.0


def suggest_transition_type(from_track: dict, to_track: dict) -> str:
    """
    Suggests 'Chorus Beatmatch' if BPM diff <= 2, both have transition points, and harmonic compatible.
    Otherwise, 'Crossfade'.
    """
    if from_track.get("title") == to_track.get("title"):
        return "Crossfade"

    bpm_diff = abs(float(from_track.get("bpm", 0)) - float(to_track.get("bpm", 0)))
    has_transition_from = "transition_point" in from_track and from_track["transition_point"] is not None
    has_transition_to = "transition_point" in to_track and to_track["transition_point"] is not None
    harmonic_compatible = is_harmonic_key(
        from_track.get("key_semitone"), to_track.get("key_semitone")
    )

    if bpm_diff <= 2 and has_transition_from and has_transition_to and harmonic_compatible:
        return "Chorus Beatmatch"
    return "Crossfade"


def format_time(seconds: float) -> str:
    """
    Formats seconds to HH:MM:SS.
    """
    return (datetime.min + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def select_and_sort_tracks_for_mixing(analyzed_data: dict) -> list[dict]:
    """
    Sorts all tracks by BPM ascending. Builds initial chain of Chorus Beatmatch-compatible tracks.
    Appends remaining tracks. Avoids immediate repeats.
    """
    all_tracks = []
    for segment in analyzed_data.get("analyzed_setlist", []):
        for track in segment.get("analyzed_tracks", []):
            track_copy = track.copy()
            track_copy["original_segment_time"] = segment.get("time", "Unknown")
            all_tracks.append(track_copy)

    if not all_tracks:
        return []

    # Global BPM sort (ascending).
    all_tracks.sort(key=lambda x: x.get("bpm", 0))

    selected_chain = []
    remaining_tracks_pool = list(all_tracks)

    # Find starting track with transition point.
    start_track_index = next(
        (i for i, track in enumerate(remaining_tracks_pool) if "transition_point" in track and track["transition_point"] is not None),
        -1,
    )
    if start_track_index == -1:
        print("No tracks with transition points. Using BPM-sorted order.")
        return all_tracks

    current_track = remaining_tracks_pool.pop(start_track_index)
    selected_chain.append(current_track)
    available_tracks = remaining_tracks_pool

    print(
        f"Starting with {current_track['title']} (BPM: {current_track.get('bpm', 'N/A')}). Building Chorus Beatmatch chain..."
    )

    # Greedily build chain.
    while available_tracks:
        next_candidates = [
            (i, t)
            for i, t in enumerate(available_tracks)
            if t["title"] != current_track["title"]
        ]
        best_next_track = None
        best_index = -1
        for orig_i, next_track in next_candidates:
            if suggest_transition_type(current_track, next_track) == "Chorus Beatmatch":
                best_next_track = next_track
                best_index = available_tracks.index(next_track)
                break

        if best_next_track:
            print(
                f" -> {best_next_track['title']} (BPM: {best_next_track.get('bpm', 'N/A')}) via Chorus Beatmatch."
            )
            selected_chain.append(best_next_track)
            available_tracks.pop(best_index)
            current_track = best_next_track
        else:
            break

    final_order = selected_chain + available_tracks
    print(
        f"Total tracks: {len(final_order)}. Chain length: {len(selected_chain)}."
    )
    print(
        f"Order preview: {[(t['title'], t['bpm']) for t in final_order[:5]]}..."
    )
    return final_order


def generate_mixing_plan(
    metadata_json_path: str,
    structure_json_path: str,
    eq_match_ms: int = 15000,
    crossfade_early_ms: int = 5500,
    first_fade_in_ms: int = 5000,
):
    """
    Main function: Merges inputs, selects/sorts tracks, computes transitions, saves mixing_plan.json.
    """
    try:
        # Merge data.
        analyzed_data = load_and_merge_data(metadata_json_path, structure_json_path)

        # Select and sort tracks.
        all_tracks = select_and_sort_tracks_for_mixing(analyzed_data)

        mixing_plan = []
        last_track_meta = None
        last_start_sec = 0.0
        last_full_duration_sec = 0.0

        for track in all_tracks:
            file_path = os.path.join(SONGS_DIR, track["file"])

            # Skip self-transition or missing file.
            if last_track_meta and track["title"] == last_track_meta["title"]:
                print(f"Skipping '{track['title']}' to avoid repeat.")
                continue
            if not os.path.exists(file_path):
                print(f"Missing: {file_path}. Skipping.")
                continue

            # Get duration.
            try:
                audio = AudioSegment.from_file(file_path)
                duration_sec = len(audio) / 1000.0
            except Exception as e:
                print(f"Error loading {track['file']}: {e}. Skipping.")
                continue

            # Compute transition.
            if last_track_meta is None:
                # First track.
                start_sec = 0.0
                transition_type = "Fade In"
                outgoing_cut_sec = None
                overlap_sec = 0.0
                from_track_title = None
                otac_val = 0.0
                transition_point_desc = "downbeat align"
                comment = (
                    f"Start with {track['title']} (BPM: {track.get('bpm', 'N/A')}, "
                    f"segment: {track.get('original_segment_time', 'N/A')})."
                )
            else:
                # Subsequent tracks.
                transition_type = suggest_transition_type(last_track_meta, track)
                otac_val = compute_otac(last_track_meta, track)
                overlap_sec = get_estimated_overlap(
                    transition_type, crossfade_early_ms, 8000, eq_match_ms
                )
                outgoing_cut_sec = last_full_duration_sec
                transition_point_desc = "beat grid match"

                if transition_type == "Chorus Beatmatch":
                    transition_point_val = last_track_meta.get("transition_point", last_full_duration_sec)
                    min_cut_sec = 30.0
                    if transition_point_val > max(overlap_sec, min_cut_sec):
                        outgoing_cut_sec = transition_point_val
                        transition_point_desc = (
                            "transition point beat align "
                            "(hard cut to intro vocals + beats crossfade over 16s)"
                        )
                    else:
                        print(
                            f"Warning: Short/early transition point in {last_track_meta['title']}. Using full duration."
                        )
                        transition_point_desc = "end of track (transition invalid)"
                else:
                    transition_point_desc = "end of track"

                # Start time.
                if transition_type == "Chorus Beatmatch":
                    start_sec = last_start_sec + outgoing_cut_sec
                else:
                    start_sec = last_start_sec + outgoing_cut_sec - overlap_sec
                from_track_title = last_track_meta["title"]
                comment = (
                    f"{from_track_title} (BPM {last_track_meta.get('bpm', 'N/A')}) "
                    f"-> {track['title']} (BPM {track.get('bpm', 'N/A')}, "
                    f"segment: {track.get('original_segment_time', 'N/A')}). "
                    f"'{transition_type}' at {transition_point_desc}. OTAC: {otac_val:.4f}"
                )

            # Add to plan.
            start_str = format_time(start_sec)
            mixing_plan.append(
                {
                    "from_track": from_track_title,
                    "to_track": track["title"],
                    "start_time": start_str,
                    "transition_point": transition_point_desc,
                    "transition_type": transition_type,
                    "outgoing_cut_sec": outgoing_cut_sec,
                    "overlap_sec": overlap_sec,
                    "otac": float(otac_val),
                    "comment": comment,
                }
            )

            # Update for next.
            last_start_sec = start_sec
            last_full_duration_sec = duration_sec
            last_track_meta = track

        # Save.
        with open("mixing_plan.json", "w") as f:
            json.dump({"mixing_plan": mixing_plan}, f, indent=2)
        print("Mixing plan saved to 'mixing_plan.json'.")

    except Exception as e:
        print(f"Error in generate_mixing_plan: {e}")
        raise


if __name__ == "__main__":
    generate_mixing_plan("metadata.json", "structure.json")