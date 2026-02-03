# generate_mixing_plan.py

"""
DJ Mixing Plan Generator (Ready for Mixing Engine)

- Reads basic_setlist.json + structure_data.json
- Uses harmonic mixing (Camelot Wheel) for key compatibility
- Dynamic overlap duration based on energy difference
- Chorus Beatmatch for normal tracks
- Outputs mixing_plan.json with exact timings for fade-in/out
"""

import os
import json
from datetime import datetime, timedelta
import librosa

SONGS_DIR = "./songs"

# ================= GENRE-SPECIFIC TRANSITION RULES =================
GENRE_TRANSITION_RULES = {
    "afrobeats": {
        "preferred_type": "verse_end",
        "avoid": ["mid_chorus"],
        "energy_preference": "smooth"
    },
    "r&b": {
        "preferred_type": "breakdown_start",
        "avoid": ["mid_vocal_phrase"],
        "energy_preference": "smooth"
    },
    "pop": {
        "preferred_type": "chorus_end",
        "avoid": [],
        "energy_preference": "energetic"
    },
    "edm": {
        "preferred_type": "pre_drop",
        "avoid": ["mid_buildup"],
        "energy_preference": "energetic"
    },
    "hip hop": {
        "preferred_type": "verse_end",
        "avoid": ["mid_hook"],
        "energy_preference": "smooth"
    },
    "dancehall": {
        "preferred_type": "chorus_end",
        "avoid": [],
        "energy_preference": "energetic"
    }
}

# ================= CAMELOT WHEEL (HARMONIC MIXING) =================
# Musical key compatibility for smooth DJ transitions
CAMELOT_WHEEL = {
    # Major keys
    "C": {"compatible": ["G", "F", "Am"], "energy_boost": ["G"], "energy_drop": ["Am"]},
    "C#": {"compatible": ["G#", "F#", "A#m"], "energy_boost": ["G#"], "energy_drop": ["A#m"]},
    "D": {"compatible": ["A", "G", "Bm"], "energy_boost": ["A"], "energy_drop": ["Bm"]},
    "D#": {"compatible": ["A#", "G#", "Cm"], "energy_boost": ["A#"], "energy_drop": ["Cm"]},
    "E": {"compatible": ["B", "A", "C#m"], "energy_boost": ["B"], "energy_drop": ["C#m"]},
    "F": {"compatible": ["C", "A#", "Dm"], "energy_boost": ["C"], "energy_drop": ["Dm"]},
    "F#": {"compatible": ["C#", "B", "D#m"], "energy_boost": ["C#"], "energy_drop": ["D#m"]},
    "G": {"compatible": ["D", "C", "Em"], "energy_boost": ["D"], "energy_drop": ["Em"]},
    "G#": {"compatible": ["D#", "C#", "Fm"], "energy_boost": ["D#"], "energy_drop": ["Fm"]},
    "A": {"compatible": ["E", "D", "F#m"], "energy_boost": ["E"], "energy_drop": ["F#m"]},
    "A#": {"compatible": ["F", "D#", "Gm"], "energy_boost": ["F"], "energy_drop": ["Gm"]},
    "B": {"compatible": ["F#", "E", "G#m"], "energy_boost": ["F#"], "energy_drop": ["G#m"]},
    
    # Minor keys
    "Am": {"compatible": ["Em", "Dm", "C"], "energy_boost": ["C"], "energy_drop": ["Dm"]},
    "A#m": {"compatible": ["Fm", "D#m", "C#"], "energy_boost": ["C#"], "energy_drop": ["D#m"]},
    "Bm": {"compatible": ["F#m", "Em", "D"], "energy_boost": ["D"], "energy_drop": ["Em"]},
    "Cm": {"compatible": ["Gm", "Fm", "D#"], "energy_boost": ["D#"], "energy_drop": ["Fm"]},
    "C#m": {"compatible": ["G#m", "F#m", "E"], "energy_boost": ["E"], "energy_drop": ["F#m"]},
    "Dm": {"compatible": ["Am", "Gm", "F"], "energy_boost": ["F"], "energy_drop": ["Gm"]},
    "D#m": {"compatible": ["A#m", "G#m", "F#"], "energy_boost": ["F#"], "energy_drop": ["G#m"]},
    "Em": {"compatible": ["Bm", "Am", "G"], "energy_boost": ["G"], "energy_drop": ["Am"]},
    "Fm": {"compatible": ["Cm", "A#m", "G#"], "energy_boost": ["G#"], "energy_drop": ["A#m"]},
    "F#m": {"compatible": ["C#m", "Bm", "A"], "energy_boost": ["A"], "energy_drop": ["Bm"]},
    "Gm": {"compatible": ["Dm", "Cm", "A#"], "energy_boost": ["A#"], "energy_drop": ["Cm"]},
    "G#m": {"compatible": ["D#m", "C#m", "B"], "energy_boost": ["B"], "energy_drop": ["C#m"]},
}

def calculate_key_compatibility(key1: str, key2: str) -> float:
    """
    Calculate harmonic compatibility score between two musical keys.
    Returns: 1.0 (perfect), 0.5 (acceptable), -1.0 (clash)
    """
    if not key1 or not key2 or key1 not in CAMELOT_WHEEL:
        return 0.5  # Unknown, assume neutral
    
    if key1 == key2:
        return 1.0  # Same key = perfect
    
    compatible_keys = CAMELOT_WHEEL[key1].get("compatible", [])
    if key2 in compatible_keys:
        return 1.0  # Harmonically compatible
    
    # Check if relative minor/major
    if key1.endswith("m") and key2 == key1[:-1]:  # Am -> A
        return 0.8
    if not key1.endswith("m") and key2 == key1 + "m":  # A -> Am
        return 0.8
    
    return -1.0  # Key clash - will sound bad


def score_transition_candidate(candidate, current_track, next_track):
    """
    Score a transition point candidate based on multiple factors.
    Returns score 0-100.
    """
    score = 50.0  # Base score
    
    # Get track metadata
    current_genre = current_track.get("genre", "").lower()
    current_key = current_track.get("key", "")
    current_bpm = current_track.get("bpm", 120)
    
    next_key = next_track.get("key", "")
    next_bpm = next_track.get("bpm", 120)
    
    # Check both locations for structure data (backward compatibility)
    next_structure = next_track.get("structure", next_track)
    next_intro_vocals = next_structure.get("has_vocals_in_first_8s", True)
    
    # FACTOR 1: Genre-specific preference (±15 points)
    genre_rules = GENRE_TRANSITION_RULES.get(current_genre, {})
    preferred_type = genre_rules.get("preferred_type")
    
    if preferred_type and candidate["type"] == preferred_type:
        score += 15
    elif candidate["type"] in genre_rules.get("avoid", []):
        score -= 15
    
    # FACTOR 2: Vocal overlap risk (±20 points)
    has_vocals_after = candidate.get("has_vocals_after", True)
    
    if has_vocals_after and next_intro_vocals:
        score -= 20  # Double vocals = muddy
    elif not has_vocals_after and not next_intro_vocals:
        score += 15  # Clean instrumental blend
    elif not has_vocals_after:
        score += 10  # Outgoing instrumental, any incoming works
    
    # FACTOR 3: Energy compatibility (±15 points)
    candidate_energy = candidate.get("energy", "medium")
    
    # Check both locations for energy analysis
    next_structure = next_track.get("structure", next_track)
    next_energy = next_structure.get("energy_analysis", {}).get("buildups", [])
    
    # Prefer transitions that maintain or build energy
    if candidate_energy == "high" and len(next_energy) > 0:
        score += 10  # High to buildup = great flow
    elif candidate_energy == "building":
        score += 15  # Building energy is ideal for transitions
    elif candidate_energy == "dropping":
        score -= 5   # Dropping energy can work but less ideal
    
    # FACTOR 4: Key compatibility (±10 points)
    key_score = calculate_key_compatibility(current_key, next_key)
    if key_score >= 1.0:
        score += 10
    elif key_score < 0:
        score -= 10
    
    # FACTOR 5: BPM difference (±5 points)
    bpm_diff = abs(current_bpm - next_bpm)
    if bpm_diff < 5:
        score += 5   # Very close BPMs = easy mix
    elif bpm_diff > 20:
        score -= 5   # Large difference needs more work
    
    # FACTOR 6: Transition type bonus
    type_scores = {
        "breakdown_start": 10,  # Best for beat-sync
        "pre_drop": 8,          # Great for energy
        "verse_end": 5,         # Safe choice
        "chorus_end": 3         # Can work but risky
    }
    score += type_scores.get(candidate["type"], 0)
    
    return max(0, min(100, score))  # Clamp 0-100


def select_best_transition_point(current_track, next_track):
    """
    Select the best transition point from all candidates.
    Uses intelligent scoring based on musical context.
    """
    # Check both locations: track["structure"] and track directly (for backward compatibility)
    structure = current_track.get("structure", current_track)
    candidates = structure.get("transition_candidates", [])
    
    if not candidates:
        # Fallback to recommended or transition_point (check both locations)
        recommended = structure.get("recommended_transition")
        if recommended:
            return recommended
        
        transition_pt = structure.get("transition_point")
        if transition_pt:
            return transition_pt
        
        # Final fallback
        return 70.0
    
    # Score each candidate
    scored_candidates = []
    for candidate in candidates:
        score = score_transition_candidate(candidate, current_track, next_track)
        scored_candidates.append({
            "time": candidate["time"],
            "type": candidate["type"],
            "score": score,
            "reasoning": candidate.get("reasoning", "")
        })
    
    # Sort by score (highest first)
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Return best candidate
    best = scored_candidates[0]
    print(f"    → Selected: {best['time']:.1f}s ({best['type']}, score: {best['score']:.0f}/100)")
    print(f"    → Reason: {best['reasoning'][:60]}..." if len(best['reasoning']) > 60 else f"    → Reason: {best['reasoning']}")
    
    return best["time"]


def calculate_dynamic_overlap(from_track: dict, to_track: dict) -> float:
    """
    Calculate optimal overlap duration based on musical characteristics.
    Real DJs vary overlap based on energy, genre, and key compatibility.
    """
    # Base overlap: 8 seconds
    base_overlap = 8.0
    
    # Get track properties
    from_bpm = from_track.get("bpm", 120)
    to_bpm = to_track.get("bpm", 120)
    from_key = from_track.get("key", "")
    to_key = to_track.get("key", "")
    from_genre = from_track.get("genre", "").lower()
    to_genre = to_track.get("genre", "").lower()
    
    # Factor 1: BPM difference (larger = longer transition)
    bpm_diff = abs(from_bpm - to_bpm)
    if bpm_diff > 20:
        base_overlap += 4.0  # Need more time to adjust
    elif bpm_diff > 10:
        base_overlap += 2.0
    
    # Factor 2: Key compatibility (bad match = shorter transition)
    key_score = calculate_key_compatibility(from_key, to_key)
    if key_score < 0:  # Key clash
        base_overlap -= 2.0  # Quick transition to minimize clash
        print(f"    ⚠️ Key clash: {from_key} → {to_key}, shorter transition")
    elif key_score >= 1.0:  # Perfect match
        base_overlap += 2.0  # Can blend longer
        print(f"    ✅ Perfect key match: {from_key} → {to_key}")
    
    # Factor 3: Genre-specific rules
    if "edm" in from_genre or "edm" in to_genre or "house" in from_genre or "house" in to_genre:
        base_overlap += 4.0  # Electronic music = longer blends
    elif "hip" in from_genre or "hip" in to_genre or "rap" in from_genre or "rap" in to_genre:
        base_overlap -= 2.0  # Hip-hop = quicker cuts
    
    # Clamp to reasonable range (4-16 seconds)
    return max(4.0, min(16.0, base_overlap))


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    return (datetime.min + timedelta(seconds=seconds)).strftime("%H:%M:%S")


def get_chorus_duration(track: dict) -> float:
    return track.get("first_chorus_end", 90.0) - track.get("first_chorus_start", 60.0)


def select_tracks_in_order(basic_setlist: dict, structure_data: dict) -> list[dict]:
    """
    Merge basic setlist and analyzed structure data, then SORT BY BPM (lowest to highest).
    This creates a smooth energy progression for DJ mixing.
    """
    all_tracks = []
    analyzed_dict = {}
    
    print(f"\n=== DEBUG: BUILDING ANALYZED DICT ===")
    print(f"structure_data keys: {list(structure_data.keys())}")
    
    for segment in structure_data.get("analyzed_setlist", []):
        print(f"  Segment keys: {list(segment.keys())}")
        for track in segment.get("analyzed_tracks", []):
            title = track.get("title", "").strip().lower()
            file = track.get("file", "").strip().lower()
            analyzed_dict[title] = track
            analyzed_dict[file] = track  # Also index by filename
            print(f"    Added to dict: title='{title}', file='{file}'")
    
    print(f"\n  Total tracks in analyzed_dict: {len(analyzed_dict)}")
    print(f"  Dict keys: {list(analyzed_dict.keys())}")
    
    print(f"\n=== DEBUG: MERGING WITH BASIC SETLIST ===")
    print(f"basic_setlist keys: {list(basic_setlist.keys())}")
    basic_setlist_segments = basic_setlist.get("setlist", [])
    print(f"  Found {len(basic_setlist_segments)} segments in basic_setlist")
    
    for segment in basic_setlist_segments:
        print(f"\n  Segment keys: {list(segment.keys())}")
        tracks_in_segment = segment.get("tracks", [])
        print(f"  Segment '{segment.get('time', 'Unknown')}' has {len(tracks_in_segment)} tracks")
        
        for track in tracks_in_segment:
            title = track.get("title", "").strip().lower()
            file = track.get("file", "").strip().lower()
            print(f"    Looking for: title='{title}', file='{file}'")
            
            # Try matching by title first, then by filename
            analyzed_track = analyzed_dict.get(title) or analyzed_dict.get(file)
            
            if analyzed_track:
                track_copy = analyzed_track.copy()
                track_copy["original_segment_time"] = segment.get("time", "Unknown")
                all_tracks.append(track_copy)
                print(f"      ✓ Matched and added: {analyzed_track.get('title')}")
            else:
                print(f"      ✗ Warning: '{title}' (file: '{file}') not found in structure data")
                print(f"      Available keys: {list(analyzed_dict.keys())}")

    # SORT BY BPM - lowest to highest for smooth energy progression
    all_tracks.sort(key=lambda t: t.get("bpm", 120))
    
    print(f"\n🎵 Song order (sorted by BPM):")
    for i, track in enumerate(all_tracks, 1):
        print(f"  {i}. {track['title']} - {track.get('bpm', 'N/A')} BPM")
    print()
    
    return all_tracks


def generate_mixing_plan(
    basic_setlist_path: str = "basic_setlist.json",
    structure_json_path: str = "structure_data.json",
    output_path: str = "mixing_plan.json",
    overlap_duration: float = 8.0,  # 8 seconds overlap at transition
    fade_duration: float = 1.0  # 1 second fade out
):
    try:
        print(f"Loading basic setlist from: {basic_setlist_path}")
        basic_setlist = load_json(basic_setlist_path)
        print(f"✓ Basic setlist loaded")
        
        print(f"Loading structure data from: {structure_json_path}")
        structure_data = load_json(structure_json_path)
        print(f"✓ Structure data loaded")

        print("Selecting tracks in order...")
        all_tracks = select_tracks_in_order(basic_setlist, structure_data)
        print(f"✓ Found {len(all_tracks)} tracks")
        
        if not all_tracks:
            raise ValueError("No tracks found after merging setlist and structure data! Check that songs have matching titles/files.")

        mixing_plan = []
        last_start_sec = 0.0
        last_track = None

        for idx, track in enumerate(all_tracks):
            print(f"Processing track {idx+1}/{len(all_tracks)}: {track.get('title', 'Unknown')}")
            file_path = os.path.join(SONGS_DIR, track["file"])
            if not os.path.exists(file_path):
                print(f"  ⚠️ Missing: {file_path}. Skipping.")
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
                # === INTELLIGENT TRANSITION POINT SELECTION ===
                print(f"  Analyzing transition: {last_track['title']} → {track['title']}")
                
                # Use intelligent scoring to select best transition point
                from_transition_point = select_best_transition_point(last_track, track)
                to_intro_duration = track.get("intro_duration", 8.0)
                to_has_early_vocals = track.get("has_vocals_in_first_8s", False)
                
                # Calculate dynamic overlap duration based on musical characteristics
                dynamic_overlap = calculate_dynamic_overlap(last_track, track)
                
                # Check key compatibility
                from_key = last_track.get("key", "")
                to_key = track.get("key", "")
                key_score = calculate_key_compatibility(from_key, to_key)
                
                # RULE: If incoming song has vocals in first 8s, previous transition must be at line end
                # Otherwise, transition can be earlier
                # This prevents vocal overlap
                
                if to_has_early_vocals:
                    # Incoming has vocals in first 8s - transition at line end
                    if to_intro_duration > dynamic_overlap:
                        incoming_start_offset = dynamic_overlap
                    else:
                        incoming_start_offset = to_intro_duration
                    
                    overlap_comment = f"Line end transition ({dynamic_overlap:.1f}s, incoming has early vocals)"
                else:
                    # No vocals in first 8s - can use full dynamic overlap
                    incoming_start_offset = dynamic_overlap
                    overlap_comment = f"Dynamic {dynamic_overlap:.1f}s overlap (no early vocals)"
                
                incoming_start_sec = last_start_sec + from_transition_point - incoming_start_offset
                
                # BPM change happens 8s before transition
                bpm_change_point = last_start_sec + from_transition_point - 8.0
                
                transition_type = "Transition Overlap"
                transition_point = from_transition_point
                incoming_intro = to_intro_duration
                
                # Enhanced comment with key info
                key_info = ""
                if key_score >= 1.0:
                    key_info = f"✓ Keys match ({from_key}→{to_key})"
                elif key_score < 0:
                    key_info = f"⚠ Key clash ({from_key}→{to_key})"
                
                comment = (
                    f"{last_track['title']} (BPM {last_track.get('bpm', 120)}) -> {track['title']} "
                    f"(BPM {track.get('bpm', 120)}). {overlap_comment}. {key_info}. "
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
