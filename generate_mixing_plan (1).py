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


# ================= SMART TRANSITION SELECTOR =================
# Threshold for choosing chorus-beatmatch vs echo-transition
BEATMATCH_THRESHOLD = 60  # Score >= 60 = chorus-beatmatch, < 60 = echo-transition

def calculate_transition_compatibility(from_track: dict, to_track: dict) -> tuple:
    """
    Calculate how well two tracks can be beatmatched together.
    
    Returns: (score 0-100, transition_type, reasons_list)
    
    Score >= 60: Use chorus-beatmatch (professional DJ blend)
    Score < 60: Use echo-transition (safer, avoids awkward overlap)
    """
    score = 100  # Start perfect, subtract for issues
    reasons = []
    bonuses = []
    
    # Get track properties
    from_key = from_track.get("key", "")
    to_key = to_track.get("key", "")
    from_bpm = from_track.get("bpm", 120)
    to_bpm = to_track.get("bpm", 120)
    from_genre = from_track.get("genre", "").lower()
    to_genre = to_track.get("genre", "").lower()
    
    # Get structure data (check both locations for backward compatibility)
    from_structure = from_track.get("structure", from_track)
    to_structure = to_track.get("structure", to_track)
    
    # Check for vocals
    from_has_vocals_at_transition = True  # Assume yes unless we know otherwise
    candidates = from_structure.get("transition_candidates", [])
    for c in candidates:
        if c.get("type") == "chorus_end":
            from_has_vocals_at_transition = c.get("has_vocals_after", True)
            break
    
    to_has_vocals_in_intro = to_structure.get("has_vocals_in_first_8s", True)
    
    # ===== FACTOR 1: KEY COMPATIBILITY (-30 points for clash) =====
    key_score = calculate_key_compatibility(from_key, to_key)
    if key_score < 0:  # Key clash
        score -= 30
        reasons.append(f"Key clash ({from_key}→{to_key})")
    elif key_score >= 1.0:
        bonuses.append(f"Keys compatible ({from_key}→{to_key})")
    
    # ===== FACTOR 2: VOCAL OVERLAP (-25 points) =====
    if from_has_vocals_at_transition and to_has_vocals_in_intro:
        score -= 25
        reasons.append("Vocal overlap risk (both tracks have vocals)")
    elif not from_has_vocals_at_transition and not to_has_vocals_in_intro:
        score += 5  # Bonus for clean instrumental blend
        bonuses.append("Clean instrumental transition")
    
    # ===== FACTOR 3: BPM DIFFERENCE (-20 points if >8%) =====
    if from_bpm > 0:
        bpm_diff_percent = abs(from_bpm - to_bpm) / from_bpm * 100
        if bpm_diff_percent > 8:
            score -= 20
            reasons.append(f"BPM diff {bpm_diff_percent:.1f}% (time-stretch artifacts)")
        elif bpm_diff_percent > 5:
            score -= 10
            reasons.append(f"BPM diff {bpm_diff_percent:.1f}%")
        elif bpm_diff_percent < 2:
            score += 5  # Bonus for matching BPM
            bonuses.append(f"BPMs nearly identical ({from_bpm}→{to_bpm})")
    
    # ===== FACTOR 4: GENRE COMPATIBILITY (-15 points) =====
    # Define genre families that mix well together
    genre_families = {
        "electronic": ["edm", "house", "techno", "trance", "dubstep", "drum and bass"],
        "urban": ["hip hop", "hip-hop", "rap", "trap", "r&b", "rnb"],
        "caribbean": ["afrobeats", "dancehall", "reggae", "soca", "amapiano"],
        "pop": ["pop", "dance pop", "electropop"],
        "latin": ["reggaeton", "latin", "bachata", "salsa"]
    }
    
    from_family = None
    to_family = None
    for family, genres in genre_families.items():
        if any(g in from_genre for g in genres):
            from_family = family
        if any(g in to_genre for g in genres):
            to_family = family
    
    if from_family and to_family and from_family != to_family:
        score -= 15
        reasons.append(f"Genre mismatch ({from_genre}→{to_genre})")
    elif from_genre == to_genre:
        score += 5
        bonuses.append(f"Same genre ({from_genre})")
    
    # ===== FACTOR 5: ENERGY FLOW (-10 points for bad flow) =====
    from_energy = from_track.get("energy", 0.5)
    to_energy = to_track.get("energy", 0.5)
    
    # Big energy drop is awkward for beatmatch
    if isinstance(from_energy, (int, float)) and isinstance(to_energy, (int, float)):
        energy_diff = from_energy - to_energy
        if energy_diff > 0.3:  # Significant energy drop
            score -= 10
            reasons.append("Large energy drop")
        elif energy_diff < -0.2:  # Energy increase is good
            score += 5
            bonuses.append("Energy builds up")
    
    # ===== DETERMINE TRANSITION TYPE =====
    if score >= BEATMATCH_THRESHOLD:
        transition_type = "chorus-beatmatch"
    else:
        transition_type = "echo-transition"
    
    return score, transition_type, reasons, bonuses


def find_chorus_times(track: dict) -> tuple:
    """
    Find BOTH the START and END of the FIRST CHORUS for DJ transitions.
    
    For beatmatch transitions:
    - At chorus_start: incoming song begins (overlap starts)
    - At chorus_end: outgoing song fades out (overlap ends)
    
    Args:
        track: Track dictionary with structure data
    
    Returns: (chorus_start, chorus_end) tuple in seconds
    """
    # Check both locations: track["structure"] and track directly (backward compatibility)
    structure = track.get("structure", track)
    
    # Get chorus_start_time and chorus_end_time from GPT analysis
    chorus_start = structure.get("chorus_start_time")
    chorus_end = structure.get("chorus_end_time")
    
    if chorus_start and chorus_end:
        print(f"    → Found chorus: {chorus_start:.1f}s - {chorus_end:.1f}s ({chorus_end - chorus_start:.1f}s)")
        first_line = structure.get("first_chorus_line", "")
        last_line = structure.get("last_chorus_line", "")
        if first_line:
            print(f"    → Start: \"{first_line[:40]}...\"" if len(first_line) > 40 else f"    → Start: \"{first_line}\"")
        if last_line:
            print(f"    → End: \"{last_line[:40]}...\"" if len(last_line) > 40 else f"    → End: \"{last_line}\"")
        return float(chorus_start), float(chorus_end)
    
    # FALLBACK: Only chorus_end available (old format)
    if chorus_end:
        # Estimate chorus_start as 15 seconds before chorus_end
        estimated_start = max(chorus_end - 15.0, 20.0)
        print(f"    → Estimated chorus: {estimated_start:.1f}s - {chorus_end:.1f}s")
        return float(estimated_start), float(chorus_end)
    
    # FALLBACK: Check transition_candidates for chorus_end type
    candidates = structure.get("transition_candidates", [])
    for candidate in candidates:
        if candidate.get("type") == "chorus_end":
            end_time = candidate["time"]
            start_time = max(end_time - 15.0, 20.0)
            print(f"    → From candidate: {start_time:.1f}s - {end_time:.1f}s")
            return start_time, end_time
    
    # Fallback to recommended_transition or transition_point
    end_time = structure.get("recommended_transition") or structure.get("transition_point") or 60.0
    start_time = max(end_time - 15.0, 20.0)
    print(f"    → Using fallback: {start_time:.1f}s - {end_time:.1f}s")
    return start_time, float(end_time)


def find_first_chorus_end(track: dict) -> float:
    """
    Find the END of the FIRST CHORUS - where to echo out.
    (Backward compatible wrapper around find_chorus_times)
    """
    _, chorus_end = find_chorus_times(track)
    return chorus_end


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
                transition_type = "fade-in"
                transition_point = None
                first_chorus_end = None
                chorus_start = None
                incoming_intro = None
                bpm_change_point = None
                compatibility_score = None
                comment = f"Start with {track['title']} (BPM {track.get('bpm', 'N/A')})"
            else:
                # === SMART TRANSITION SELECTION ===
                print(f"\n  🎛️  Analyzing: {last_track['title']} → {track['title']}")
                
                # Calculate compatibility score
                compat_score, recommended_type, issues, bonuses = calculate_transition_compatibility(last_track, track)
                compatibility_score = compat_score
                
                print(f"    📊 Compatibility Score: {compat_score}/100")
                if bonuses:
                    print(f"    ✅ Bonuses: {', '.join(bonuses)}")
                if issues:
                    print(f"    ⚠️  Issues: {', '.join(issues)}")
                print(f"    🎵 Recommended: {recommended_type.upper()}")
                
                # Get BOTH chorus_start and chorus_end from structure analysis
                chorus_start, chorus_end = find_chorus_times(last_track)
                to_intro_duration = track.get("intro_duration", 8.0)
                
                # Key info for comment
                from_key = last_track.get("key", "")
                to_key = track.get("key", "")
                key_score = calculate_key_compatibility(from_key, to_key)
                
                if recommended_type == "chorus-beatmatch":
                    # === CHORUS BEATMATCH: Professional DJ blend ===
                    # At chorus_start: incoming song begins playing
                    # Both tracks play together from chorus_start to chorus_end
                    # At chorus_end: outgoing fades out, incoming continues solo
                    transition_type = "chorus-beatmatch"
                    
                    # Overlap duration is the entire chorus length
                    overlap_duration = chorus_end - chorus_start
                    
                    # Ensure minimum overlap of 8 seconds
                    if overlap_duration < 8.0:
                        # Extend chorus_start backwards to get at least 8s overlap
                        chorus_start = max(chorus_end - 12.0, 15.0)
                        overlap_duration = chorus_end - chorus_start
                    
                    # Safety check: chorus_start must be before chorus_end
                    if chorus_start >= chorus_end:
                        # If chorus is too early or invalid, use echo-transition instead
                        print(f"    ⚠️ Invalid chorus timing ({chorus_start:.1f}s >= {chorus_end:.1f}s), switching to echo-transition")
                        transition_type = "echo-transition"
                        transition_point = chorus_end
                        first_chorus_end = chorus_end
                        chorus_start = None
                        
                        echo_duration = 3.0
                        incoming_start_sec = last_start_sec + chorus_end + echo_duration
                        bpm_change_point = incoming_start_sec
                        incoming_intro = to_intro_duration
                        
                        comment = (
                            f"ECHO-OUT (invalid timing): {last_track['title']} → {track['title']}. "
                            f"Echo at {chorus_end:.1f}s, 3s echo, then incoming."
                        )
                        print(f"    🔊 Echo-out: Transition at {chorus_end:.1f}s + 3s echo")
                    else:
                        transition_point = chorus_start
                        first_chorus_end = chorus_end
                        
                        incoming_start_sec = last_start_sec + chorus_start
                        bpm_change_point = incoming_start_sec
                        incoming_intro = to_intro_duration
                        
                        key_info = f"✓ Keys: {from_key}→{to_key}" if key_score >= 0.5 else f"⚠ Keys: {from_key}→{to_key}"
                        
                        comment = (
                            f"BEATMATCH (Score:{compat_score}): {last_track['title']} → {track['title']}. "
                            f"Chorus: {chorus_start:.1f}s-{chorus_end:.1f}s ({overlap_duration:.0f}s overlap). "
                            f"{key_info}. BPM {last_track.get('bpm', 120)}→{track.get('bpm', 120)}."
                        )
                        print(f"    🎧 Beatmatch: Chorus {chorus_start:.1f}s → {chorus_end:.1f}s ({overlap_duration:.0f}s overlap)")
                    
                else:
                    # === ECHO TRANSITION: Safe fallback ===
                    # Echo out first song at chorus_end, then start second
                    transition_type = "echo-transition"
                    transition_point = chorus_end
                    first_chorus_end = chorus_end
                    chorus_start = None
                    
                    echo_duration = 3.0
                    incoming_start_sec = last_start_sec + chorus_end + echo_duration
                    bpm_change_point = incoming_start_sec
                    incoming_intro = to_intro_duration
                    
                    issue_text = f" Issues: {', '.join(issues[:2])}" if issues else ""
                    comment = (
                        f"ECHO-OUT (Score:{compat_score}): {last_track['title']} → {track['title']}. "
                        f"Echo at {first_chorus_end:.1f}s, 3s echo, then incoming.{issue_text}"
                    )
                    print(f"    🔊 Echo-out: Transition at {first_chorus_end:.1f}s + 3s echo")

            start_str = format_time(incoming_start_sec)

            # Build mixing plan entry
            entry = {
                "from_track": last_track["title"] if last_track else None,
                "to_track": track["title"],
                "incoming_start_sec": incoming_start_sec,
                "start_time": start_str,
                "transition_point": transition_point,
                "first_chorus_end_sec": first_chorus_end if last_track else None,
                "incoming_intro_duration": incoming_intro,
                "bpm_change_point_sec": bpm_change_point,
                "compatibility_score": compatibility_score,
                "transition_type": transition_type,
                "to_bpm": track.get("bpm", 120),
                "from_bpm": last_track.get("bpm", 120) if last_track else None,
                "comment": comment,
            }
            
            # Add type-specific fields
            if transition_type == "chorus-beatmatch":
                entry["chorus_start_sec"] = chorus_start
                entry["overlap_duration_sec"] = overlap_duration
            elif transition_type == "echo-transition":
                entry["echo_duration_sec"] = 3.0
            
            mixing_plan.append(entry)

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
