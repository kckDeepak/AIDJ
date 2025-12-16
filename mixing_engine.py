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

# Import waveform visualization module
try:
    from waveform_visualizer import (
        plot_waveform_alignment,
        plot_beat_alignment,
        plot_phase_cancellation_check,
        plot_mix_overview
    )
    VISUALIZATION_ENABLED = True
except ImportError as e:
    print(f"Warning: Waveform visualization disabled - {e}")
    VISUALIZATION_ENABLED = False

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


# ================= WAVEFORM PHASE ALIGNMENT (PROFESSIONAL DJ TECHNIQUE) =================
def align_waveform_phase(outgoing: AudioSegment, incoming: AudioSegment, max_shift_ms=50):
    """
    Align waveforms at the sample level using cross-correlation.
    This prevents phase cancellation when overlaying tracks.
    
    Professional DJs use this technique to ensure waveforms constructively interfere
    rather than cancel each other out (which causes volume drops/hollow sound).
    
    Process:
    1. Convert both audio segments to numpy arrays
    2. Use cross-correlation to find optimal phase alignment
    3. Shift incoming waveform by optimal offset
    4. Check for phase coherence after alignment
    
    Returns: (phase_aligned_incoming, shift_samples, coherence_score)
    """
    # Convert to numpy arrays for processing
    y_out = audio_segment_to_np(outgoing)
    y_in = audio_segment_to_np(incoming)
    sr = outgoing.frame_rate
    
    # Limit analysis to first few seconds for speed
    analysis_samples = min(sr * 5, len(y_out), len(y_in))  # 5 seconds max
    y_out_segment = y_out[:analysis_samples]
    y_in_segment = y_in[:analysis_samples]
    
    # Calculate max shift in samples
    max_shift_samples = int(sr * max_shift_ms / 1000)
    
    # Use cross-correlation to find optimal alignment
    # This finds where the two waveforms match best
    correlation = np.correlate(y_out_segment, y_in_segment, mode='same')
    
    # Find the lag that gives maximum correlation
    center = len(correlation) // 2
    search_range = min(max_shift_samples, center)
    search_start = center - search_range
    search_end = center + search_range
    
    search_correlation = correlation[search_start:search_end]
    optimal_lag_idx = np.argmax(np.abs(search_correlation))
    optimal_lag_samples = optimal_lag_idx - search_range
    
    # Apply shift to incoming audio
    if optimal_lag_samples > 0:
        # Add silence to beginning
        silence_duration_ms = int(optimal_lag_samples * 1000 / sr)
        aligned_incoming = AudioSegment.silent(silence_duration_ms) + incoming
    elif optimal_lag_samples < 0:
        # Trim from beginning
        trim_duration_ms = int(abs(optimal_lag_samples) * 1000 / sr)
        aligned_incoming = incoming[trim_duration_ms:]
    else:
        aligned_incoming = incoming
    
    # Calculate phase coherence score (0-1, higher is better)
    # This measures how well the waveforms align
    y_in_aligned = audio_segment_to_np(aligned_incoming)[:analysis_samples]
    if len(y_in_aligned) >= analysis_samples:
        # Normalize both signals for comparison
        y_out_norm = y_out_segment / (np.max(np.abs(y_out_segment)) + 1e-8)
        y_in_norm = y_in_aligned / (np.max(np.abs(y_in_aligned)) + 1e-8)
        
        # Calculate correlation coefficient (coherence)
        coherence = np.abs(np.corrcoef(y_out_norm, y_in_norm)[0, 1])
    else:
        coherence = 0.0
    
    shift_ms = optimal_lag_samples * 1000 / sr
    
    return aligned_incoming, shift_ms, coherence

def detect_zero_crossings(audio_seg: AudioSegment, window_ms=100):
    """
    Detect zero-crossing points in audio for clean transition points.
    Zero crossings are where the waveform crosses the zero amplitude line.
    Mixing at zero crossings prevents clicks and pops.
    
    Returns: Array of zero-crossing positions in milliseconds
    """
    y = audio_segment_to_np(audio_seg)
    sr = audio_seg.frame_rate
    
    # Find where signal crosses zero
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    
    # Convert to milliseconds
    zero_crossing_times_ms = (zero_crossings / sr) * 1000
    
    return zero_crossing_times_ms

def check_phase_cancellation(outgoing: AudioSegment, incoming: AudioSegment, overlap_start_ms=0):
    """
    Check if overlaying two audio segments will cause phase cancellation.
    Phase cancellation occurs when two waveforms are out of phase and cancel each other.
    
    Returns: (has_cancellation: bool, cancellation_severity: float)
    """
    # Extract overlap sections
    overlap_duration_ms = min(len(outgoing) - overlap_start_ms, len(incoming), 5000)  # Max 5s check
    if overlap_duration_ms <= 0:
        return False, 0.0
    
    y_out = audio_segment_to_np(outgoing[overlap_start_ms:overlap_start_ms + overlap_duration_ms])
    y_in = audio_segment_to_np(incoming[:overlap_duration_ms])
    
    # Normalize for fair comparison
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-8)
    y_in = y_in / (np.max(np.abs(y_in)) + 1e-8)
    
    # Check if signals are approximately opposite (phase cancellation)
    # Negative correlation indicates phase opposition
    min_len = min(len(y_out), len(y_in))
    correlation = np.corrcoef(y_out[:min_len], y_in[:min_len])[0, 1]
    
    # Cancellation severity: -1 = complete opposition, 0 = no correlation, 1 = perfect alignment
    has_cancellation = correlation < -0.3  # Threshold for significant cancellation
    severity = abs(min(correlation, 0))  # Only negative correlations matter
    
    return has_cancellation, severity

# ================= ADVANCED BEAT ALIGNMENT SYSTEM =================
def detect_beat_grid(audio_seg: AudioSegment, bpm=None):
    """
    Detect precise beat grid with downbeat detection using librosa.
    Returns beat times in milliseconds and downbeat positions.
    """
    y = audio_segment_to_np(audio_seg)
    sr = audio_seg.frame_rate
    
    # Use librosa's advanced beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Detect downbeats (first beat of each bar - every 4 beats typically)
    # Use onset strength to find stronger beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # Downbeats are typically every 4 beats in 4/4 time
    if len(beat_times) >= 4:
        # Find downbeats by looking at onset strength at beat positions
        beat_strengths = [onset_env[int(librosa.time_to_frames(bt, sr=sr))] for bt in beat_times]
        
        # Group into bars of 4 beats
        downbeat_indices = []
        for i in range(0, len(beat_times) - 3, 4):
            # Find strongest beat in this group of 4
            bar_beats = beat_strengths[i:i+4]
            strongest_in_bar = i + np.argmax(bar_beats)
            downbeat_indices.append(strongest_in_bar)
        
        downbeat_times = beat_times[downbeat_indices]
    else:
        downbeat_times = beat_times[::4] if len(beat_times) >= 4 else beat_times[:1]
    
    # Convert to milliseconds
    beat_times_ms = beat_times * 1000
    downbeat_times_ms = downbeat_times * 1000
    
    return beat_times_ms, downbeat_times_ms, tempo

def align_beats_perfect(outgoing: AudioSegment, incoming: AudioSegment, 
                        overlap_duration_ms: int, bpm_from: float, bpm_to: float,
                        track_names: tuple = ("Track A", "Track B")):
    """
    Perfect beat-to-beat alignment during overlap.
    
    Process:
    1. Detect beat grids in both tracks
    2. Align first downbeat of incoming to outgoing's beat grid
    3. Apply micro time-stretching per beat to maintain sync throughout overlap
    4. Generate waveform visualizations for DJ analysis
    
    Args:
        outgoing: Outgoing track audio
        incoming: Incoming track audio
        overlap_duration_ms: Duration of overlap
        bpm_from: BPM of outgoing track
        bpm_to: BPM of incoming track
        track_names: Tuple of (outgoing_name, incoming_name) for visualization
    
    Returns: (aligned_incoming, shift_ms)
    """
    print("   🎯 Applying perfect beat alignment...")
    
    # Detect beat grids
    try:
        outgoing_beats, outgoing_downbeats, _ = detect_beat_grid(outgoing, bpm_from)
        incoming_beats, incoming_downbeats, _ = detect_beat_grid(incoming, bpm_to)
    except Exception as e:
        print(f"   ⚠️  Beat detection failed: {e}, using basic alignment")
        return incoming, 0
    
    if len(outgoing_downbeats) == 0 or len(incoming_downbeats) == 0:
        print("   ⚠️  No downbeats detected, using basic alignment")
        return incoming, 0
    
    # === VISUALIZATION: Beat Grid Alignment ===
    if VISUALIZATION_ENABLED and len(outgoing_beats) > 0 and len(incoming_beats) > 0:
        try:
            plot_beat_alignment(outgoing, incoming, 
                              outgoing_beats, incoming_beats,
                              outgoing_downbeats, incoming_downbeats,
                              track_names)
        except Exception as e:
            print(f"   ⚠️  Beat visualization failed: {e}")
    
    # STEP 1: Align first downbeat
    # Find the first downbeat in outgoing (should be near start of overlap section)
    target_downbeat = outgoing_downbeats[0]
    incoming_downbeat = incoming_downbeats[0]
    
    # Calculate shift needed to align downbeats
    shift_ms = target_downbeat - incoming_downbeat
    
    print(f"   → Downbeat alignment: {shift_ms:.1f}ms shift")
    
    # Apply initial shift
    if shift_ms > 0:
        incoming = AudioSegment.silent(int(shift_ms)) + incoming
        # Update beat positions
        incoming_beats = incoming_beats + shift_ms
        incoming_downbeats = incoming_downbeats + shift_ms
    elif shift_ms < 0:
        incoming = incoming[int(-shift_ms):]
        # Update beat positions
        incoming_beats = incoming_beats + shift_ms
        incoming_downbeats = incoming_downbeats + shift_ms
    
    # STEP 2: Beat grid warping for continuous sync
    # Only warp beats within the overlap duration
    overlap_beats_out = outgoing_beats[outgoing_beats < overlap_duration_ms]
    overlap_beats_in = incoming_beats[incoming_beats < overlap_duration_ms]
    
    if len(overlap_beats_out) < 2 or len(overlap_beats_in) < 2:
        print(f"   → Basic alignment applied (shift: {shift_ms:.1f}ms)")
        return incoming, shift_ms
    
    # Match the number of beats to process
    num_beats = min(len(overlap_beats_out), len(overlap_beats_in))
    
    # Calculate beat-by-beat drift and apply micro corrections
    total_corrections = 0
    corrected_audio = incoming
    
    for i in range(1, num_beats):
        target_beat_time = overlap_beats_out[i]
        actual_beat_time = overlap_beats_in[i]
        drift_ms = actual_beat_time - target_beat_time
        
        # Apply micro time-stretch if drift exceeds threshold (10ms)
        if abs(drift_ms) > 10:
            # Calculate the segment to correct (from previous beat to this beat)
            segment_start = int(overlap_beats_in[i-1])
            segment_end = int(overlap_beats_in[i])
            
            if segment_end > segment_start and segment_end <= len(corrected_audio):
                segment = corrected_audio[segment_start:segment_end]
                
                # Calculate correction ratio
                target_duration = overlap_beats_out[i] - overlap_beats_out[i-1]
                actual_duration = segment_end - segment_start
                correction_ratio = target_duration / actual_duration
                
                # Limit correction to prevent artifacts (±5%)
                correction_ratio = np.clip(correction_ratio, 0.95, 1.05)
                
                # Apply micro time-stretch
                if abs(correction_ratio - 1.0) > 0.01:
                    try:
                        stretched_segment = time_stretch_audio(segment, correction_ratio)
                        
                        # Reconstruct audio with corrected segment
                        before = corrected_audio[:segment_start]
                        after = corrected_audio[segment_end:]
                        corrected_audio = before + stretched_segment + after
                        
                        total_corrections += 1
                    except:
                        pass  # Skip if stretch fails
    
    if total_corrections > 0:
        print(f"   → Beat grid warping: {total_corrections} micro-corrections applied")
    else:
        print(f"   → Beats already aligned (drift < 10ms)")
    
    # STEP 3: WAVEFORM PHASE ALIGNMENT (Professional DJ technique)
    # After beat alignment, align waveforms at sample level to prevent phase cancellation
    print(f"   🌊 Applying waveform phase alignment...")
    
    # Use a section from both tracks for phase analysis
    phase_analysis_duration = min(overlap_duration_ms, 3000, len(outgoing), len(corrected_audio))
    outgoing_phase_section = outgoing[:phase_analysis_duration]
    incoming_phase_section = corrected_audio[:phase_analysis_duration]
    
    # Apply phase alignment
    phase_aligned, phase_shift_ms, coherence = align_waveform_phase(
        outgoing_phase_section,
        corrected_audio,
        max_shift_ms=20  # Max 20ms micro-adjustment for phase
    )
    
    # Check for phase cancellation
    has_cancellation, severity = check_phase_cancellation(
        outgoing,
        phase_aligned,
        overlap_start_ms=0
    )
    
    if has_cancellation:
        print(f"   ⚠️  Phase cancellation detected (severity: {severity:.2f})")
        print(f"   → Inverting phase to fix cancellation...")
        # Invert the phase of incoming track to fix cancellation
        y_inverted = audio_segment_to_np(phase_aligned) * -1
        phase_aligned = np_to_audio_segment(y_inverted, sr=phase_aligned.frame_rate)
        # Re-check
        has_cancellation, severity = check_phase_cancellation(outgoing, phase_aligned)
        if not has_cancellation:
            print(f"   ✅ Phase cancellation fixed!")
    
    print(f"   → Phase alignment: {phase_shift_ms:.2f}ms shift, coherence: {coherence:.3f}")
    
    # === VISUALIZATION: Waveform Phase Alignment ===
    if VISUALIZATION_ENABLED:
        try:
            plot_waveform_alignment(
                outgoing_phase_section, 
                phase_aligned[:phase_analysis_duration] if len(phase_aligned) >= phase_analysis_duration else phase_aligned,
                phase_analysis_duration, 
                track_names,
                phase_shift_ms, 
                coherence
            )
        except Exception as e:
            print(f"   ⚠️  Waveform visualization failed: {e}")
    
    # === VISUALIZATION: Phase Cancellation Check ===
    if VISUALIZATION_ENABLED and (has_cancellation or severity > 0.2):
        try:
            plot_phase_cancellation_check(
                outgoing,
                phase_aligned,
                has_cancellation, 
                severity,
                track_names
            )
        except Exception as e:
            print(f"   ⚠️  Phase check visualization failed: {e}")
    
    # Use phase-aligned version if coherence improved significantly
    if abs(phase_shift_ms) > 1.0:  # Only apply if shift is meaningful
        corrected_audio = phase_aligned
    
    return corrected_audio, shift_ms

def apply_gradual_tempo_sync(outgoing_audio: AudioSegment, overlap_duration_ms: int, 
                             stretch_factor: float):
    """
    Gradually adjust tempo during overlap only - not entire song.
    This mimics how Pioneer CDJs and Traktor handle tempo sync.
    
    Process:
    1. Keep most of outgoing track at original tempo
    2. Gradually ramp tempo in the overlap section only
    3. Smooth transition from 1.0x to target stretch_factor
    
    Args:
        outgoing_audio: The outgoing track audio
        overlap_duration_ms: Duration of overlap in milliseconds
        stretch_factor: Target tempo multiplier (e.g., 1.05 = 5% faster)
    
    Returns:
        AudioSegment with gradual tempo adjustment
    """
    if abs(stretch_factor - 1.0) < 0.01:
        return outgoing_audio  # No sync needed
    
    print(f"   🎛️  Applying gradual tempo sync: 1.00x → {stretch_factor:.3f}x over {overlap_duration_ms/1000:.1f}s")
    
    # Calculate ramp start point (2x overlap duration before end for smooth ramp)
    ramp_duration_ms = min(overlap_duration_ms * 2, 16000)  # Max 16s ramp
    ramp_start = len(outgoing_audio) - ramp_duration_ms
    
    if ramp_start < 0:
        # Audio too short, stretch entire thing
        return time_stretch_audio(outgoing_audio, stretch_factor)
    
    # Split audio: stable section + ramp section
    stable_section = outgoing_audio[:ramp_start]
    ramp_section = outgoing_audio[ramp_start:]
    
    # Divide ramp section into small chunks for smooth gradual change
    num_chunks = 32  # 32 chunks = very smooth transition
    chunk_duration_ms = len(ramp_section) // num_chunks
    
    if chunk_duration_ms < 100:
        # Chunks too small, reduce number
        num_chunks = max(8, len(ramp_section) // 100)
        chunk_duration_ms = len(ramp_section) // num_chunks
    
    ramped_chunks = []
    total_stretch_applied = 0
    
    for i in range(num_chunks):
        chunk_start = i * chunk_duration_ms
        chunk_end = (i + 1) * chunk_duration_ms if i < num_chunks - 1 else len(ramp_section)
        chunk = ramp_section[chunk_start:chunk_end]
        
        # Calculate progressive stretch factor (linear interpolation)
        progress = i / (num_chunks - 1)  # 0.0 → 1.0
        current_stretch = 1.0 + (stretch_factor - 1.0) * progress
        
        # Apply micro-stretch to this chunk
        try:
            stretched_chunk = time_stretch_audio(chunk, current_stretch)
            ramped_chunks.append(stretched_chunk)
            total_stretch_applied += abs(current_stretch - 1.0)
        except Exception as e:
            # If stretch fails, use original chunk
            ramped_chunks.append(chunk)
    
    # Recombine: stable section + gradually ramped section
    result = stable_section
    for chunk in ramped_chunks:
        result += chunk
    
    avg_stretch = total_stretch_applied / num_chunks if num_chunks > 0 else 0
    print(f"   → Gradual ramp: {num_chunks} micro-adjustments (avg {avg_stretch:.4f}x per segment)")
    
    return result

def detect_beats(audio_seg: AudioSegment, sr=44100, min_interval_ms=200):
    """Detect beat positions in audio segment using energy peaks (legacy method)."""
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
    """Find optimal lag to align beats between two audio segments (legacy method)."""
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
    
    # Track mix overview data for visualization
    mix_overview_data = []

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
            
            # Add to mix overview
            mix_overview_data.append({
                'name': to_title,
                'start_ms': 0,
                'duration_ms': len(to_audio),
                'bpm': bpm_to
            })
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
        
        # STEP 1: GRADUAL BPM SYNC - Apply smooth tempo transition BEFORE overlap starts
        # The outgoing track must be fully synced to incoming BPM by the time overlap begins
        # This ensures both tracks are at the same tempo during the overlap for perfect alignment
        if previous_track_audio and bpm_from > 0 and bpm_to > 0:
            stretch_factor = safe_float(bpm_to/bpm_from, 1.0)
            stretch_factor = np.clip(stretch_factor, 0.90, 1.10)  # Allow wider range for DJ mixing
            if abs(stretch_factor - 1.0) > 0.01:
                print(f"   BPM sync: {bpm_from:.1f} → {bpm_to:.1f} (stretch: {stretch_factor:.3f}x)")
                print(f"   → Sync completes BEFORE overlap (at {transition_point_in_mix/1000:.1f}s)")
                
                # Calculate tempo ramp duration (typically 8-16 seconds before transition)
                ramp_duration_ms = min(overlap_duration_ms * 2, 16000)  # Max 16s ramp
                ramp_start_in_mix = max(0, transition_point_in_mix - ramp_duration_ms)
                
                # Split mix into: before ramp, ramp section, overlap section
                before_ramp = mix[:ramp_start_in_mix]
                ramp_section = mix[ramp_start_in_mix:transition_point_in_mix]
                overlap_section = mix[transition_point_in_mix:]
                
                # Apply gradual tempo sync to ramp section (completes at transition point)
                synced_ramp = apply_gradual_tempo_sync(
                    ramp_section, 
                    ramp_duration_ms, 
                    stretch_factor
                )
                
                # Reconstruct mix: before ramp + synced ramp + overlap (both now at same BPM)
                mix = before_ramp + synced_ramp + overlap_section
                
                print(f"   → Tempo ramp: {ramp_start_in_mix/1000:.1f}s to {transition_point_in_mix/1000:.1f}s ({ramp_duration_ms/1000:.1f}s)")
        
        # STEP 2: PERFECT BEAT-GRID ALIGNMENT for incoming track
        # Get a section from previous track at transition point for beat matching
        if previous_track_audio:
            match_section_start = max(0, transition_point_ms - 5000)
            match_section_end = min(transition_point_ms + overlap_duration_ms + 5000, len(previous_track_audio))
            match_section = previous_track_audio[match_section_start:match_section_end]
            
            # Get incoming section for alignment (overlap duration + buffer)
            incoming_alignment_section = to_audio[:min(overlap_duration_ms + 10000, len(to_audio))]
            
            # Apply perfect beat-to-beat alignment (with visualization)
            track_names = (from_title, to_title)
            aligned_incoming, shift_ms = align_beats_perfect(
                match_section, 
                incoming_alignment_section, 
                overlap_duration_ms, 
                bpm_from, 
                bpm_to,
                track_names=track_names
            )
            
            # Update the full incoming track with aligned version
            if len(aligned_incoming) > 0:
                remainder = to_audio[len(incoming_alignment_section):]
                to_audio = aligned_incoming + remainder
            
            if abs(shift_ms) > 50:
                print(f"   ✅ Perfect beat alignment: {shift_ms:.1f}ms initial shift + grid warping")
        
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
        
        # Add to mix overview
        mix_overview_data.append({
            'name': to_title,
            'start_ms': incoming_start_in_mix,
            'duration_ms': len(to_audio),
            'bpm': bpm_to
        })
        
        print(f"   Mix length after: {len(mix)/1000:.1f}s")

    # === GENERATE MIX OVERVIEW VISUALIZATION ===
    if VISUALIZATION_ENABLED and len(mix_overview_data) > 0:
        try:
            print("\n📊 Generating mix overview visualization...")
            plot_mix_overview(mix_overview_data)
        except Exception as e:
            print(f"⚠️  Mix overview visualization failed: {e}")

    print("\n" + "="*60)
    print("Normalizing & exporting final mix...")
    final_mix = normalize(mix)
    final_mix.export(output_path, format="mp3", bitrate="320k",
                     tags={"artist":"AI DJ","title":"Full-Auto Chorus Mix"})
    print(f"✅ MIX READY → {output_path} ({len(final_mix)/60000:.1f} minutes)")
    print("="*60)

if __name__ == "__main__":
    generate_mix()
