# mixing_engine.py
"""
DJ Mixing Engine: Chorus Beatmatch + Crossfade + Minor Time-Stretch

- Reads mixing_plan.json + structure_data.json
- Applies chorus beatmatch: second song starts at first chorus start of first
- Fade out first song at first chorus end
- Minor time-stretching on incoming song to match BPM of outgoing (±2%)
- Every 5th song: full-song + crossfade
- Outputs normalized mix.mp3
"""

import os
import json
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import librosa
import scipy.signal
from scipy.signal import butter, sosfilt

SONGS_DIR = "./songs"

# ================= GENRE-SPECIFIC DJ MIXING RULES =================
GENRE_MIXING_RULES = {
    "edm": {
        "name": "EDM/Electronic",
        "overlap_multiplier": 2.0,  # Longer blends (16s instead of 8s)
        "use_breakdown": True,
        "eq_filter_strength": 1.5,  # Stronger filtering
        "description": "Extended blends, use breakdowns for mixing"
    },
    "house": {
        "name": "House",
        "overlap_multiplier": 1.5,  # 12s overlaps
        "isolate_drums": True,
        "eq_filter_strength": 1.2,
        "description": "Longer blends, drum-focused transitions"
    },
    "hip-hop": {
        "name": "Hip-Hop/Rap",
        "overlap_multiplier": 0.5,  # Quick cuts (4s)
        "cut_style": "quick",
        "eq_filter_strength": 0.8,  # Less filtering
        "description": "Quick cuts, beat juggling style"
    },
    "rap": {
        "name": "Rap",
        "overlap_multiplier": 0.5,
        "cut_style": "quick",
        "eq_filter_strength": 0.8,
        "description": "Quick cuts, minimal overlap"
    },
    "pop": {
        "name": "Pop",
        "overlap_multiplier": 1.0,  # Standard 8s
        "eq_filter_strength": 1.0,
        "description": "Standard transitions"
    },
    "rock": {
        "name": "Rock",
        "overlap_multiplier": 0.75,  # Slightly shorter (6s)
        "eq_filter_strength": 0.9,
        "description": "Moderate overlaps, energy-focused"
    }
}

def get_genre_rules(genre):
    """
    Get mixing rules for a specific genre.
    Returns default if genre not found.
    """
    genre_lower = genre.lower() if genre else "unknown"
    
    # Check for exact match
    for key, rules in GENRE_MIXING_RULES.items():
        if key in genre_lower:
            return rules
    
    # Default rules
    return {
        "name": "Default",
        "overlap_multiplier": 1.0,
        "eq_filter_strength": 1.0,
        "description": "Standard transitions"
    }

# ================= SAFE CONVERSIONS =================
def ms(seconds) -> int:
    if seconds is None or seconds == "":
        return 0
    try:
        return int(round(float(seconds) * 1000))
    except (ValueError, TypeError):
        return 0

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except:
        return default

# ================= AUDIO UTILITIES =================
def audio_segment_to_np(seg: AudioSegment):
    samples = np.array(seg.get_array_of_samples())
    if seg.channels == 2:
        samples = samples.reshape((-1,2)).mean(axis=1)
    return samples.astype(np.float32) / 32768.0

def np_to_audio_segment(y: np.ndarray, sr: int = 44100):
    y = np.clip(y, -1.0, 1.0)
    y_int16 = (y * 32767).astype(np.int16)
    return AudioSegment(y_int16.tobytes(), frame_rate=sr, sample_width=2, channels=1)

# ================= EQ FILTERING (PROFESSIONAL DJ MIXING) =================
def apply_lowpass_filter(audio: AudioSegment, cutoff_hz: float = 8000):
    """
    Apply low-pass filter to audio (cuts high frequencies).
    Used on OUTGOING track during fadeout to prevent frequency clash.
    """
    y = audio_segment_to_np(audio)
    sr = audio.frame_rate
    
    # Ensure cutoff is below Nyquist frequency (sr/2)
    nyquist = sr / 2.0
    cutoff_hz = min(cutoff_hz, nyquist * 0.95)  # 95% of Nyquist for safety
    
    # Design Butterworth low-pass filter
    sos = butter(4, cutoff_hz, btype='lowpass', fs=sr, output='sos')
    y_filtered = sosfilt(sos, y)
    
    return np_to_audio_segment(y_filtered, sr=sr)

def apply_highpass_filter(audio: AudioSegment, cutoff_hz: float = 200):
    """
    Apply high-pass filter to audio (cuts low frequencies/bass).
    Used on INCOMING track during intro to prevent bass clash.
    """
    y = audio_segment_to_np(audio)
    sr = audio.frame_rate
    
    # Ensure cutoff is valid (must be > 0 and < Nyquist)
    nyquist = sr / 2.0
    cutoff_hz = max(20, min(cutoff_hz, nyquist * 0.95))  # Between 20Hz and 95% Nyquist
    
    # Design Butterworth high-pass filter
    sos = butter(4, cutoff_hz, btype='highpass', fs=sr, output='sos')
    y_filtered = sosfilt(sos, y)
    
    return np_to_audio_segment(y_filtered, sr=sr)

def apply_progressive_eq(audio: AudioSegment, filter_type: str = "lowpass"):
    """
    Apply EQ filter that gradually increases in strength.
    Creates smooth transition instead of sudden frequency cut.
    """
    duration_ms = len(audio)
    if duration_ms < 100:
        return audio
    
    sr = audio.frame_rate
    nyquist = sr / 2.0
    num_chunks = 10
    chunk_size = duration_ms // num_chunks
    
    result = AudioSegment.empty()
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, duration_ms)
        chunk = audio[start:end]
        
        # Gradually increase filter strength
        progress = (i + 1) / num_chunks
        
        if filter_type == "lowpass":
            # Start at 12kHz (or safe limit), end at 4kHz (or safe limit)
            max_cutoff = min(12000, nyquist * 0.95)
            min_cutoff = min(4000, nyquist * 0.5)
            cutoff = max_cutoff - ((max_cutoff - min_cutoff) * progress)
            filtered = apply_lowpass_filter(chunk, cutoff)
        else:  # highpass
            # Start at 100Hz, end at 300Hz
            cutoff = 100 + (200 * progress)
            filtered = apply_highpass_filter(chunk, cutoff)
        
        result += filtered
    
    return result


# ================= BEAT ALIGNMENT =================
def detect_beats(audio_seg: AudioSegment, sr=44100, min_interval_ms=200):
    """Detect beat positions in audio segment using energy peaks."""
    y = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
    if audio_seg.channels == 2:
        y = y.reshape((-1,2)).mean(axis=1)
    y = y / (np.max(np.abs(y))+1e-8)
    
    # Use RMS energy for beat detection
    energy = np.abs(y)
    energy_smooth = scipy.signal.medfilt(energy, kernel_size=101)
    
    # Find peaks with proper spacing
    min_distance = int(sr * min_interval_ms / 1000)
    peaks, properties = scipy.signal.find_peaks(
        energy_smooth, 
        height=0.2, 
        distance=min_distance,
        prominence=0.1
    )
    
    # Convert to milliseconds
    beat_times_ms = (peaks / sr) * 1000
    return beat_times_ms

def find_best_lag(outgoing: AudioSegment, incoming: AudioSegment):
    """Find optimal lag to align beats between two audio segments."""
    outgoing_beats = detect_beats(outgoing)
    incoming_beats = detect_beats(incoming)
    
    if len(outgoing_beats) == 0 or len(incoming_beats) == 0:
        return 0.0
    
    # Use first strong beat as reference
    target_start = outgoing_beats[0]
    incoming_start = incoming_beats[0]
    lag_ms = target_start - incoming_start
    
    # Limit lag to ±1 second for safety
    return np.clip(lag_ms/1000.0, -1.0, 1.0)

# ================= TIME-STRETCH =================
def time_stretch_audio(audio: AudioSegment, factor: float):
    """Time-stretch AudioSegment without changing pitch using librosa."""
    y = audio_segment_to_np(audio)
    y_stretch = librosa.effects.time_stretch(y, rate=factor)
    return np_to_audio_segment(y_stretch, sr=audio.frame_rate)

# ================= TRANSITIONS =================
def apply_chorus_beatmatch(current_track: AudioSegment, incoming_track: AudioSegment, 
                           chorus_start_ms: int, chorus_end_ms: int, 
                           fade_duration_ms: int, bpm_from: float, bpm_to: float):
    """
    Apply chorus beatmatch transition between two INDIVIDUAL tracks:
    1. Play current_track up to chorus_start (solo)
    2. At chorus_start, introduce incoming_track - BOTH play together
    3. Current track continues playing normally during overlap
    4. At chorus_end, fade out current_track over fade_duration
    5. Continue with incoming track solo after fade completes
    
    Returns: Combined audio with transition
    """
    # Minor time-stretch to match BPM (limit to ±2% for quality)
    if bpm_from > 0 and bpm_to > 0:
        stretch_factor = safe_float(bpm_to/bpm_from, 1.0)
        stretch_factor = np.clip(stretch_factor, 0.98, 1.02)
        if abs(stretch_factor - 1.0) > 0.01:
            print(f"  Time-stretching incoming by {stretch_factor:.3f}x ({bpm_to:.1f}/{bpm_from:.1f} BPM)")
            incoming_track = time_stretch_audio(incoming_track, stretch_factor)

    # Beat alignment: align incoming with chorus section of current
    chorus_section = current_track[chorus_start_ms:min(chorus_end_ms, len(current_track))]
    incoming_start = incoming_track[:min(10000, len(incoming_track))]
    
    lag_sec = find_best_lag(chorus_section, incoming_start)
    lag_ms = ms(lag_sec)
    
    if abs(lag_ms) > 50:
        print(f"  Beat alignment: {lag_ms}ms lag adjustment")
        if lag_ms > 0:
            incoming_track = AudioSegment.silent(lag_ms) + incoming_track
        elif lag_ms < 0:
            incoming_track = incoming_track[-lag_ms:]

    # Part 1: Current track BEFORE chorus starts (solo outgoing)
    before_chorus = current_track[:chorus_start_ms]
    
    # Part 2: Overlap section - both tracks play together
    # Current track continues from chorus_start to chorus_end + fade_duration
    overlap_end = min(chorus_end_ms + fade_duration_ms, len(current_track))
    overlap_duration = overlap_end - chorus_start_ms
    
    current_overlap = current_track[chorus_start_ms:overlap_end]
    incoming_overlap = incoming_track[:overlap_duration]
    
    # === PROFESSIONAL DJ EQ MIXING ===
    # Apply low-pass filter to outgoing track (reduce highs to prevent clash)
    print(f"  Applying EQ: Low-pass on outgoing, High-pass on incoming")
    current_overlap_filtered = apply_progressive_eq(current_overlap, filter_type="lowpass")
    
    # Apply high-pass filter to incoming track intro (reduce bass initially)
    incoming_overlap_filtered = apply_progressive_eq(incoming_overlap, filter_type="highpass")
    
    # Fade out current track at chorus_end (during the fade_duration)
    fade_start_in_overlap = chorus_end_ms - chorus_start_ms
    if fade_duration_ms > 0 and fade_start_in_overlap < len(current_overlap_filtered):
        # Keep current playing normally until chorus_end, then fade
        before_fade = current_overlap_filtered[:fade_start_in_overlap]
        fade_section = current_overlap_filtered[fade_start_in_overlap:].fade_out(
            min(fade_duration_ms, len(current_overlap_filtered) - fade_start_in_overlap)
        )
        current_overlap_filtered = before_fade + fade_section
    
    # Overlay both tracks during overlap period (now with EQ filtering)
    overlap_mixed = current_overlap_filtered.overlay(incoming_overlap_filtered)
    
    # Part 3: After overlap - incoming track continues solo
    # Ensure incoming plays for at least 30 seconds total
    remaining_incoming = incoming_track[overlap_duration:]
    min_play_time_ms = 30000  # 30 seconds minimum
    current_incoming_duration = overlap_duration + len(remaining_incoming)
    
    if current_incoming_duration < min_play_time_ms:
        # This shouldn't happen if chorus detection is correct, but safeguard
        print(f"  Warning: Incoming track plays for only {current_incoming_duration/1000:.1f}s")
    
    # Combine all parts sequentially
    result = before_chorus + overlap_mixed + remaining_incoming
    
    print(f"  Transition: {len(before_chorus)/1000:.1f}s solo → "
          f"{len(overlap_mixed)/1000:.1f}s overlap → "
          f"{len(remaining_incoming)/1000:.1f}s solo incoming")
    
    return normalize(result)

def apply_crossfade(pre_mix: AudioSegment, next_audio: AudioSegment, fade_duration_ms: int = 5000):
    return pre_mix.append(next_audio, crossfade=fade_duration_ms)

# ================= MAIN MIX =================
def generate_mix(mixing_plan_json: str = "output/mixing_plan.json", 
                structure_json: str = "output/structure_data.json",
                output_path: str = "output/mix.mp3"):
    print("Loading data...")
    with open(mixing_plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f).get("mixing_plan", [])
    with open(structure_json, "r", encoding="utf-8") as f:
        structure_data = json.load(f)

    tracks_db = {}
    for section in structure_data.get("analyzed_setlist", []):
        for track in section.get("analyzed_tracks", []):
            tracks_db[track["title"]] = track

    if not plan:
        print("No mixing plan found!")
        return

    mix = AudioSegment.empty()
    previous_track_audio = None
    previous_track_start_in_mix = 0

    for idx, entry in enumerate(plan):
        to_title = entry.get("to_track")
        if not to_title or to_title not in tracks_db:
            print(f"  [SKIP] Missing track: {to_title}")
            continue

        to_meta = tracks_db[to_title]
        to_path = os.path.join(SONGS_DIR, to_meta.get("file",""))
        if not os.path.exists(to_path):
            print(f"  [ERROR] File not found: {to_path}")
            continue

        to_audio = AudioSegment.from_file(to_path)
        bpm_to = safe_float(to_meta.get("bpm", 0))

        # First track: just add it with fade in
        if idx == 0:
            print(f"\n1. {to_title} [Intro]")
            print(f"   Track duration: {len(to_audio)/1000:.1f}s")
            mix = to_audio.fade_in(ms(2))
            previous_track_audio = to_audio
            previous_track_start_in_mix = 0
            continue

        # Get transition parameters from mixing plan
        from_title = entry.get("from_track")
        if not from_title or from_title not in tracks_db:
            print(f"\n  [WARN] from_track missing: {from_title}. Appending with crossfade.")
            mix = mix.append(to_audio, crossfade=2000)
            previous_track_audio = to_audio
            previous_track_start_in_mix = len(mix) - len(to_audio)
            continue

        from_meta = tracks_db[from_title]
        bpm_from = safe_float(from_meta.get("bpm", 0))
        
        # NEW STRUCTURE: Get timing from mixing plan
        transition_point_sec = safe_float(entry.get("transition_point"), 90.0)
        incoming_intro_sec = safe_float(entry.get("incoming_intro_duration"), 8.0)
        bpm_change_point_sec = safe_float(entry.get("bpm_change_point_sec"), 82.0)
        overlap_duration_sec = safe_float(entry.get("overlap_duration"), 8.0)
        fade_duration_sec = safe_float(entry.get("fade_duration"), 1.0)
        
        transition_point_ms = ms(transition_point_sec)
        overlap_duration_ms = ms(overlap_duration_sec)
        fade_duration_ms = ms(fade_duration_sec)
        
        print(f"\n{idx+1}. → {to_title}")
        print(f"   Mix length before: {len(mix)/1000:.1f}s")
        print(f"   Transition at: {transition_point_sec:.1f}s")
        print(f"   Incoming intro: {incoming_intro_sec:.1f}s")
        print(f"   BPM change at: {bpm_change_point_sec:.1f}s")
        print(f"   Overlap: {overlap_duration_sec:.1f}s, Fade: {fade_duration_sec:.1f}s")

        # Calculate timing positions in the mix
        transition_point_in_mix = previous_track_start_in_mix + transition_point_ms
        bpm_change_point_in_mix = previous_track_start_in_mix + ms(bpm_change_point_sec)
        
        # Calculate when incoming track should start (based on intro duration)
        if incoming_intro_sec > 8.0:
            incoming_start_offset_sec = 8.0
        else:
            incoming_start_offset_sec = incoming_intro_sec
        
        incoming_start_in_mix = transition_point_in_mix - ms(incoming_start_offset_sec)
        
        print(f"   Incoming starts at: {incoming_start_in_mix/1000:.1f}s in mix")
        print(f"   Transition point: {transition_point_in_mix/1000:.1f}s in mix")
        
        # STEP 1: BPM change on previous track 8 seconds before transition
        # Apply time-stretching to the entire previous track if needed
        if previous_track_audio and bpm_from > 0 and bpm_to > 0:
            stretch_factor = safe_float(bpm_to/bpm_from, 1.0)
            stretch_factor = np.clip(stretch_factor, 0.90, 1.10)  # Allow wider range for DJ mixing
            if abs(stretch_factor - 1.0) > 0.01:
                print(f"   BPM change: {bpm_from:.1f} → {bpm_to:.1f} (stretch: {stretch_factor:.3f}x)")
                
                # Get the part that needs BPM change (from bpm_change_point to end)
                bpm_change_offset_in_track = ms(bpm_change_point_sec) - previous_track_start_in_mix
                if bpm_change_offset_in_track > 0:
                    # Keep beginning unchanged, stretch the rest
                    before_bpm_change = mix[:bpm_change_point_in_mix]
                    after_bpm_change = mix[bpm_change_point_in_mix:]
                    stretched_section = time_stretch_audio(after_bpm_change, stretch_factor)
                    mix = before_bpm_change + stretched_section
        
        # STEP 2: Beat alignment for incoming track
        # Get a section from previous track at transition point for beat matching
        if previous_track_audio:
            match_section_start = max(0, transition_point_ms - 5000)
            match_section = previous_track_audio[match_section_start:min(transition_point_ms + 5000, len(previous_track_audio))]
            incoming_start_section = to_audio[:min(10000, len(to_audio))]
            
            lag_sec = find_best_lag(match_section, incoming_start_section)
            lag_ms = ms(lag_sec)
            
            if abs(lag_ms) > 50:
                print(f"   Beat alignment: {lag_ms}ms adjustment")
                if lag_ms > 0:
                    to_audio = AudioSegment.silent(lag_ms) + to_audio
                elif lag_ms < 0:
                    to_audio = to_audio[-lag_ms:]
        
        # STEP 3: Overlay incoming track starting at calculated position
        # Pad incoming to start at the right position in the mix
        incoming_padded = AudioSegment.silent(incoming_start_in_mix) + to_audio
        
        # STEP 4: Create the overlap section (8 seconds at transition point)
        overlap_end_in_mix = transition_point_in_mix + overlap_duration_ms
        
        # Keep mix up to where overlap starts
        mix_before_overlap = mix[:transition_point_in_mix]
        
        # Get the overlap section from previous track (8 seconds after transition)
        overlap_from_previous = mix[transition_point_in_mix:overlap_end_in_mix]
        
        # Get corresponding section from incoming track
        incoming_at_transition = incoming_start_in_mix
        overlap_start_in_incoming = transition_point_in_mix - incoming_at_transition
        overlap_from_incoming = to_audio[int(overlap_start_in_incoming):int(overlap_start_in_incoming + overlap_duration_ms)]
        
        # Overlay both tracks during overlap
        overlapped_section = overlap_from_previous.overlay(overlap_from_incoming)
        
        # STEP 5: Fade out previous track during last 1 second of overlap
        fade_start_in_mix = overlap_end_in_mix - fade_duration_ms
        
        # Split overlapped section into non-fade and fade parts
        non_fade_duration_ms = overlap_duration_ms - fade_duration_ms
        overlap_before_fade = overlapped_section[:int(non_fade_duration_ms)]
        overlap_with_fade = overlapped_section[int(non_fade_duration_ms):]
        
        # Get incoming audio during fade (no fade on incoming)
        fade_start_in_incoming = int(overlap_start_in_incoming + non_fade_duration_ms)
        incoming_during_fade = to_audio[fade_start_in_incoming:fade_start_in_incoming + int(fade_duration_ms)]
        
        # Fade out previous from the overlapped section, keep incoming clear
        # Apply fade to the previous track component only
        previous_during_fade = overlap_with_fade.fade_out(int(fade_duration_ms))
        faded_section = incoming_during_fade.overlay(previous_during_fade)
        
        # STEP 6: Continue with incoming track only after overlap
        remaining_incoming_start = int(overlap_start_in_incoming + overlap_duration_ms)
        remaining_incoming = to_audio[remaining_incoming_start:]
        
        # STEP 7: Combine all sections
        mix = mix_before_overlap + overlap_before_fade + faded_section + remaining_incoming
        
        # Update tracking variables
        previous_track_audio = to_audio
        previous_track_start_in_mix = incoming_start_in_mix
        
        print(f"   Mix length after: {len(mix)/1000:.1f}s")

    print("\n" + "="*60)
    print("Normalizing & exporting final mix...")
    final_mix = normalize(mix)
    final_mix.export(output_path, format="mp3", bitrate="320k",
                     tags={"artist":"AI DJ","title":"Full-Auto Chorus Mix"})
    print(f"✅ MIX READY → {output_path} ({len(final_mix)/60000:.1f} minutes)")
    print("="*60)

if __name__ == "__main__":
    generate_mix()
