"""
Songs Router - Song Management API
==================================

Endpoints for:
- Listing songs with metadata
- Uploading new MP3 files
- Deleting songs
- Getting waveform data
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import librosa
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
SONGS_DIR = BASE_DIR / "songs"
NOTES_DIR = BASE_DIR / "notes"

router = APIRouter()


# ==================== MODELS ====================

class SongMetadata(BaseModel):
    """Song metadata model"""
    filename: str
    title: str
    artist: str
    bpm: Optional[float] = None
    key: Optional[str] = None
    genre: Optional[str] = None
    energy: Optional[float] = None
    duration: Optional[float] = None


class SongListResponse(BaseModel):
    """Response model for song list"""
    songs: List[SongMetadata]
    total: int


class UploadResponse(BaseModel):
    """Response model for upload"""
    filename: str
    success: bool
    message: str
    metadata: Optional[SongMetadata] = None


# ==================== HELPERS ====================

def parse_filename(filename: str) -> tuple:
    """Parse artist and title from filename"""
    name = Path(filename).stem
    
    # Try "Artist - Title" format
    if " - " in name:
        parts = name.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    
    # Fallback
    return "Unknown Artist", name


def load_cached_metadata(filename: str) -> Optional[dict]:
    """Load cached metadata from notes folder"""
    cache_name = Path(filename).stem.replace(" ", "_").lower()
    cache_path = NOTES_DIR / f"{cache_name}_metadata.json"
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_audio_duration(filepath: Path) -> Optional[float]:
    """Get audio duration in seconds"""
    try:
        y, sr = librosa.load(str(filepath), sr=None, duration=10)  # Just load 10s to get duration fast
        # Get full duration from file info
        import audioread
        with audioread.audio_open(str(filepath)) as f:
            return f.duration
    except Exception:
        return None


def generate_waveform_data(filepath: Path, num_points: int = 100) -> List[float]:
    """Generate simplified waveform data for visualization"""
    try:
        y, sr = librosa.load(str(filepath), sr=22050, mono=True)
        
        # Downsample to desired number of points
        chunk_size = len(y) // num_points
        waveform = []
        
        for i in range(num_points):
            start = i * chunk_size
            end = start + chunk_size
            chunk = y[start:end]
            # Use RMS of chunk
            rms = np.sqrt(np.mean(chunk ** 2))
            waveform.append(float(rms))
        
        # Normalize to 0-1
        max_val = max(waveform) if waveform else 1
        waveform = [v / max_val for v in waveform]
        
        return waveform
    except Exception:
        return []


# ==================== ENDPOINTS ====================

@router.get("", response_model=SongListResponse)
async def list_songs():
    """List all songs with metadata from both Supabase and local cache"""
    from backend.services.supabase_storage import list_files
    
    songs = []
    
    # Get files from Supabase Storage
    supabase_result = list_files()
    
    if supabase_result.get("success") and supabase_result.get("files"):
        # Process songs from Supabase
        for file_info in supabase_result["files"]:
            filename = file_info["name"]
            artist, title = parse_filename(filename)
            
            # Load cached metadata if available
            cached = load_cached_metadata(filename)
            
            song = SongMetadata(
                filename=filename,
                title=title,
                artist=artist,
                bpm=cached.get("bpm") if cached else None,
                key=cached.get("key") if cached else None,
                genre=cached.get("genre") if cached else None,
                energy=cached.get("energy") if cached else None,
                duration=cached.get("duration") if cached else None
            )
            songs.append(song)
    else:
        # Fallback to local directory scan
        for mp3_file in SONGS_DIR.glob("*.mp3"):
            artist, title = parse_filename(mp3_file.name)
            
            # Load cached metadata if available
            cached = load_cached_metadata(mp3_file.name)
            
            song = SongMetadata(
                filename=mp3_file.name,
                title=title,
                artist=artist,
                bpm=cached.get("bpm") if cached else None,
                key=cached.get("key") if cached else None,
                genre=cached.get("genre") if cached else None,
                energy=cached.get("energy") if cached else None,
                duration=cached.get("duration") if cached else None
            )
            songs.append(song)
    
    # Sort by title
    songs.sort(key=lambda s: s.title.lower())
    
    return SongListResponse(songs=songs, total=len(songs))


@router.get("/{filename}/metadata", response_model=SongMetadata)
async def get_song_metadata(filename: str):
    """Get detailed metadata for a specific song"""
    filepath = SONGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Song not found: {filename}")
    
    artist, title = parse_filename(filename)
    cached = load_cached_metadata(filename)
    
    # Get duration if not cached
    duration = cached.get("duration") if cached else None
    if duration is None:
        duration = get_audio_duration(filepath)
    
    return SongMetadata(
        filename=filename,
        title=title,
        artist=artist,
        bpm=cached.get("bpm") if cached else None,
        key=cached.get("key") if cached else None,
        genre=cached.get("genre") if cached else None,
        energy=cached.get("energy") if cached else None,
        duration=duration
    )


@router.get("/{filename}/waveform")
async def get_song_waveform(filename: str, points: int = Query(default=100, le=500)):
    """Get waveform data for visualization"""
    filepath = SONGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Song not found: {filename}")
    
    waveform = generate_waveform_data(filepath, points)
    
    return {
        "filename": filename,
        "waveform": waveform,
        "points": len(waveform)
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_song(file: UploadFile = File(...)):
    """Upload a new MP3 file"""
    # Validate file type
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files are allowed")
    
    # Check if file already exists
    filepath = SONGS_DIR / file.filename
    if filepath.exists():
        raise HTTPException(status_code=400, detail=f"Song already exists: {file.filename}")
    
    try:
        # Save file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse metadata
        artist, title = parse_filename(file.filename)
        
        return UploadResponse(
            filename=file.filename,
            success=True,
            message=f"Successfully uploaded {file.filename}",
            metadata=SongMetadata(
                filename=file.filename,
                title=title,
                artist=artist
            )
        )
    except Exception as e:
        # Clean up on failure
        if filepath.exists():
            filepath.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.put("/{filename}/rename")
async def rename_song(filename: str, new_name: str):
    """Rename a song file"""
    filepath = SONGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Song not found: {filename}")
    
    try:
        # Create new filename with .mp3 extension
        new_filename = new_name if new_name.endswith('.mp3') else f"{new_name}.mp3"
        new_filepath = SONGS_DIR / new_filename
        
        # Check if new name already exists
        if new_filepath.exists() and new_filepath != filepath:
            raise HTTPException(status_code=400, detail=f"A song with name '{new_filename}' already exists")
        
        # Rename the file
        filepath.rename(new_filepath)
        
        # Rename cached metadata if exists
        old_cache_name = Path(filename).stem.replace(" ", "_").lower()
        new_cache_name = Path(new_filename).stem.replace(" ", "_").lower()
        
        for pattern in ["_metadata.json", "_structure.json"]:
            old_cache_path = NOTES_DIR / f"{old_cache_name}{pattern}"
            new_cache_path = NOTES_DIR / f"{new_cache_name}{pattern}"
            if old_cache_path.exists():
                old_cache_path.rename(new_cache_path)
        
        artist, title = parse_filename(new_filename)
        
        return {
            "success": True, 
            "old_filename": filename,
            "new_filename": new_filename,
            "title": title,
            "artist": artist
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rename failed: {str(e)}")


@router.delete("/{filename}")
async def delete_song(filename: str):
    """Delete a song from both local storage and Supabase"""
    from backend.services.supabase_storage import delete_file
    
    filepath = SONGS_DIR / filename
    
    # Track errors but don't fail if one deletion fails
    errors = []
    
    try:
        # Delete from Supabase Storage
        result = delete_file(filename)
        if not result.get("success"):
            errors.append(f"Supabase: {result.get('error')}")
        else:
            print(f"✅ Deleted from Supabase: {filename}")
    except Exception as e:
        errors.append(f"Supabase: {str(e)}")
    
    try:
        # Delete local file if exists
        if filepath.exists():
            filepath.unlink()
            print(f"✅ Deleted locally: {filename}")
        
        # Delete cached metadata if exists
        cache_name = Path(filename).stem.replace(" ", "_").lower()
        for pattern in ["_metadata.json", "_structure.json"]:
            cache_path = NOTES_DIR / f"{cache_name}{pattern}"
            if cache_path.exists():
                cache_path.unlink()
        
    except Exception as e:
        errors.append(f"Local: {str(e)}")
    
    # Return success if at least Supabase deletion worked (local is ephemeral anyway)
    if "Supabase:" not in str(errors) or not errors:
        return {"success": True, "message": f"Deleted {filename}"}
    else:
        raise HTTPException(status_code=500, detail=f"Delete failed: {'; '.join(errors)}")
