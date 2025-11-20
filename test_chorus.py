import librosa, json, os
from track_analysis_identification_engine import _detect_choruses, analyze_track

file_path = os.path.join("./songs", "Needle.mp3")  # Replace with a real file
y, sr = librosa.load(file_path, sr=None)
choruses = _detect_choruses(y, sr)
print(json.dumps(choruses, indent=2))