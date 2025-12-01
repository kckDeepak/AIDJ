# generate_mix.py

"""
This module generates a continuous DJ mix by applying transitions based on an analyzed setlist and a mixing plan.
It supports 'Chorus Beatmatch' and other transitions, with 'Chorus Beatmatch' using a short crossfade logic at the exact transition point.

Key features:
- Uses the globally sorted track order from 'mixing_plan.json'.
- Converts audio between pydub's AudioSegment and NumPy arrays for processing.
- **Chorus Beatmatch:** Short 2s crossfade (fade out outgoing, fade in incoming) at exact transition point, with tempo stretching to match BPMs. Uses same building logic as crossfade (extracts tail from current mix).
- **Fallback/Other:** Standard crossfade with configurable overlap.
- Handles missing files gracefully and normalizes the final mix for consistent loudness.

Dependencies:
- os: For file path operations.
- json: For parsing input setlist and mixing plan JSON files.
- numpy: For numerical operations on audio data.
- librosa: For audio feature extraction (e.g., time stretch).
- pydub: For loading, manipulating, and exporting audio files.
"""

import os 
import json 
import numpy as np 
import librosa 
from pydub import AudioSegment 
from pydub.effects import normalize 

# Define the directory path where local MP3 song files are stored.
SONGS_DIR = "./songs"


# ---------------------------
# Utility conversions
# ---------------------------
def audio_segment_to_np(segment: AudioSegment):
    """Converts a pydub AudioSegment to a mono float32 NumPy array with values in [-1, 1]."""
    samples = np.array(segment.get_array_of_samples()) 
    if segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1) 
    sr = segment.frame_rate 
    y = samples.astype(np.float32) / 32768.0 
    return y, sr


def np_to_audio_segment(y: np.ndarray, sr: int):
    """Converts a mono float32 NumPy array in [-1, 1] to a pydub AudioSegment (mono, 16-bit)."""
    y_clipped = np.clip(y, -1.0, 1.0) 
    y_int16 = (y_clipped * 32767.0).astype(np.int16) 
    return AudioSegment(
        y_int16.tobytes(), 
        frame_rate=sr, 
        sample_width=2, 
        channels=1 
    )


# ---------------------------
# Standard crossfade transition
# ---------------------------
def apply_transition(segment1: AudioSegment,
                     segment2: AudioSegment,
                     transition_type: str,
                     duration_ms: int = 8000,
                     early_ms: int = 5500,
                     otac: float = 0.0,
                     eq_match_duration_ms: int = 15000):
    """
    Applies a crossfade transition between outgoing tail and incoming track.
    Stretches incoming based on OTAC if provided.
    """
    # Compute overlap
    overlap_ms = duration_ms + early_ms
    overlap_ms = max(overlap_ms, duration_ms)  # Ensure at least duration
    overlap_ms = min(overlap_ms, len(segment1), len(segment2))  # Safety

    # Convert to numpy for stretch
    y1, sr1 = audio_segment_to_np(segment1[-overlap_ms:])
    y2_full, sr2 = audio_segment_to_np(segment2)

    # Apply OTAC tempo stretch to incoming track if otac != 0
    if abs(otac) > 0.01:
        stretch_duration_sec = max(duration_ms, eq_match_duration_ms) / 1000.0
        rate = 1.0 + otac * stretch_duration_sec / 60.0
        y2_full = librosa.effects.time_stretch(y2_full, rate=rate)

    segment2_stretched = np_to_audio_segment(y2_full, sr2)
    segment2_stretched = segment2_stretched.set_frame_rate(segment1.frame_rate).set_channels(1)

    # Extract overlap portions
    outgoing_tail = segment1[-overlap_ms:]
    incoming_head = segment2_stretched[:overlap_ms]

    # Standard crossfade
    faded_out = outgoing_tail.fade_out(overlap_ms)
    faded_in = incoming_head.fade_in(overlap_ms)
    crossed = faded_out.overlay(faded_in)
    return crossed + segment2_stretched[overlap_ms:]


# ---------------------------
# Mix generator
# ---------------------------
def generate_mix(analyzed_setlist_json, mixing_plan_json, first_fade_in_ms=5000, crossfade_early_ms=5500, eq_match_ms=15000):
    """
    Generates a continuous DJ mix by processing an analyzed setlist and applying transitions from a mixing plan.
    """
    try:
        # Parse the analyzed setlist JSON string to get all metadata.
        analyzed_data = json.loads(analyzed_setlist_json)
        # Load the mixing plan from the specified JSON file.
        mixing_plan = json.load(open(mixing_plan_json, 'r')).get("mixing_plan", [])
        
        full_mix = AudioSegment.empty()  # Initialize an empty AudioSegment for the mix.
        last_track_start_mix_sec = None  # Track the mix start time (seconds) of the last added track.
        last_track_meta = None  # Track last track metadata for transitions
        
        # Flatten all track metadata into a single dictionary for easy lookup by title.
        all_tracks_metadata = {}
        for segment in analyzed_data.get("analyzed_setlist", []):
            for track in segment.get("analyzed_tracks", []):
                all_tracks_metadata[track['title']] = track

        # Iterate over the mixing plan entries (each entry represents the start of a new track).
        for track_index, plan_entry in enumerate(mixing_plan):
            to_track_title = plan_entry.get("to_track")
            
            # --- Get Incoming Track Metadata ---
            if to_track_title not in all_tracks_metadata:
                print(f"[generate_mix] Missing metadata for incoming track: {to_track_title}. Skipping.")
                continue
            track = all_tracks_metadata[to_track_title]  # to_meta

            transition_type = plan_entry.get("transition_type", "Crossfade") 
            otac = plan_entry.get("otac", 0.0) 
            outgoing_cut_sec = plan_entry.get("outgoing_cut_sec") 
            overlap_sec = plan_entry.get("overlap_sec", 8.0) 

            file_path = os.path.join(SONGS_DIR, track["file"]) 
            if not os.path.exists(file_path):
                print(f"[generate_mix] Missing file: {file_path}. Skipping track.")
                continue

            # Load the audio file (incoming track).
            to_audio = AudioSegment.from_file(file_path)

            if track_index == 0:
                # --- FIRST TRACK ---
                fade_dur = int(min(first_fade_in_ms, len(to_audio)))
                full_mix += to_audio.fade_in(fade_dur)
                last_track_start_mix_sec = 0.0
                last_track_meta = track
            else:
                # --- SUBSEQUENT TRACKS (TRANSITIONS) ---
                from_title = plan_entry.get("from_track")
                from_meta = all_tracks_metadata.get(from_title)
                if from_meta is None:
                    print(f"[generate_mix] Missing from metadata for {from_title}. Falling back to crossfade.")
                    transition_type = "Crossfade"
                
                # 1. Calculate the exact mix time of the outgoing track's cut point.
                trans_start_mix_sec = last_track_start_mix_sec + outgoing_cut_sec
                trans_start_ms = int(trans_start_mix_sec * 1000 + 0.5)

                # Prepare parameters based on transition type
                if transition_type == "Chorus Beatmatch":
                    # Short 2s crossfade at exact point, with BPM matching
                    fade_ms = 2000
                    overlap_ms = fade_ms
                    duration_ms = fade_ms
                    early_ms = 0
                    effective_otac = 0.0  # Will handle BPM stretch separately

                    # Stretch incoming to match outgoing BPM
                    bpm_from = last_track_meta.get('bpm', 120)
                    bpm_to = track.get('bpm', 120)
                    if abs(bpm_from - bpm_to) > 0.5:
                        y_to, sr_to = audio_segment_to_np(to_audio)
                        rate = float(bpm_from) / float(bpm_to)
                        y_st = librosa.effects.time_stretch(y_to, rate=rate)
                        to_audio = np_to_audio_segment(y_st, sr_to)
                        print(f"[generate_mix] Stretched '{to_track_title}' to match BPM {bpm_from} from {bpm_to}")
                else:
                    # Standard crossfade parameters
                    duration_ms = 8000
                    early_ms = crossfade_early_ms
                    overlap_ms = int(overlap_sec * 1000)  # Use from plan or default
                    overlap_ms = duration_ms + early_ms  # Override with standard
                    effective_otac = otac

                # 2. Determine the start of the overlap/tail segment in the current mix.
                tail_start_ms = max(0, trans_start_ms - overlap_ms)

                # 3. Trim the mix and extract the necessary segments.
                pre_transition = full_mix[:tail_start_ms]  # Audio before the overlap starts.
                tail = full_mix[tail_start_ms:trans_start_ms]  # The overlap segment of the outgoing track.
                
                if len(tail) < 500:
                    print(f"[generate_mix] Warning: Tail for '{plan_entry.get('from_track')}' is too short ({len(tail)}ms). Appending fully.")
                    full_mix += to_audio
                    last_track_start_mix_sec = len(full_mix) / 1000.0 - len(to_audio) / 1000.0
                    last_track_meta = track
                    continue

                # 4. Apply the transition.
                trans_audio = apply_transition(
                    tail, to_audio, transition_type,
                    duration_ms=duration_ms,
                    early_ms=early_ms,
                    otac=effective_otac,
                    eq_match_duration_ms=eq_match_ms
                )
                
                # 5. Rebuild full_mix.
                full_mix = pre_transition + trans_audio
                
                # 6. Update last start time for the *newly added* track.
                last_track_start_mix_sec = trans_start_mix_sec - (overlap_ms / 1000.0)
                
                last_track_meta = track

        # Normalize the final mix for consistent loudness and export as MP3.
        full_mix = normalize(full_mix)
        full_mix.export("mix.mp3", format="mp3")
        print("Mix exported to 'mix.mp3'")

    except Exception as e:
        print(f"[generate_mix] Error: {e}")
        raise


# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    """
    Entry point for testing the mix generator with a sample analyzed setlist and mixing plan.
    """
    # Sample setlist with chorus times added for proper Chorus Beatmatch testing.
    sample_analyzed_setlist_json = '''
    {
        "analyzed_setlist": [
            {
                "time": "19:00–20:00",
                "analyzed_tracks": [
                    {
                        "title": "Tum Hi Ho",
                        "artist": "Arijit Singh",
                        "file": "Arijit Singh - Tum Hi Ho.mp3",
                        "bpm": 94,
                        "key_semitone": 9,
                        "scale": "major",
                        "genre": "bollywood",
                        "energy": 0.45,
                        "valence": 0.32,
                        "danceability": 0.52,
                        "has_vocals": true,
                        "segments": [{"label": "verse", "start": 30.0, "end": 60.0}],
                        "choruses": [{"start": 60.0, "end": 90.0}], 
                        "chroma_matrix": null,
                        "transition": "Fade In",
                        "notes": "Balanced Vibe track. Genre: bollywood."
                    },
                    {
                        "title": "No Scrubs",
                        "artist": "TLC",
                        "file": "TLC - No Scrubs.mp3",
                        "bpm": 93,
                        "key_semitone": 8,
                        "scale": "minor",
                        "genre": "r&b",
                        "energy": 0.7,
                        "valence": 0.6,
                        "danceability": 0.8,
                        "has_vocals": true,
                        "segments": [{"label": "verse", "start": 20.0, "end": 50.0}],
                        "choruses": [{"start": 45.0, "end": 75.0}], 
                        "chroma_matrix": null,
                        "transition": "Crossfade",
                        "notes": "Dance Floor Filler track. Genre: r&b."
                    }
                ]
            }
        ]
    }
    '''
    generate_mix(sample_analyzed_setlist_json, "mixing_plan.json")