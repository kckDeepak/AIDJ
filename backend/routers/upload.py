"""
Upload Router - Local Storage Upload API
=========================================

Simple file upload that saves MP3 files directly to the local songs/ folder.

Endpoints:
- POST /upload-audio - Upload MP3 file to local storage
- GET /api/upload/files - List all uploaded files
- DELETE /api/upload/{filename} - Delete a file
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["upload"])

# Get base directory
BASE_DIR = Path(__file__).parent.parent.parent
SONGS_DIR = BASE_DIR / "songs"
SONGS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MODELS ====================

class UploadResponse(BaseModel):
    """Response for file upload"""
    success: bool
    url: str
    filename: str
    message: str


class FileInfo(BaseModel):
    """Info about an uploaded file"""
    name: str
    url: str
    size: int
    created_at: Optional[str] = None


class FileListResponse(BaseModel):
    """Response for file listing"""
    success: bool
    files: List[FileInfo]
    total: int


# ==================== ENDPOINTS ====================

@router.post("/upload-audio", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload an MP3 file to local storage
    
    Flow:
    1. Client sends file to this endpoint
    2. Backend validates file (MP3 only, max 50MB)
    3. Backend saves to local songs/ folder
    4. Backend returns local URL for playback
    
    Returns:
        {
            "success": true,
            "url": "/static/songs/song.mp3",
            "filename": "song.mp3",
            "message": "File uploaded successfully"
        }
    """
    
    print(f"🎯 UPLOAD: {file.filename}")
    
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith('.mp3'):
        raise HTTPException(
            status_code=400,
            detail="Only MP3 files are allowed. Please upload a .mp3 file."
        )
    
    # Validate content type
    allowed_types = ["audio/mpeg", "audio/mp3", "application/octet-stream"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected audio/mpeg or audio/mp3."
        )
    
    # Read file content
    try:
        file_data = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
    
    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB in bytes
    if len(file_data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is 50MB. Your file: {len(file_data) / 1024 / 1024:.2f}MB"
        )
    
    if len(file_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )
    
    # Save file locally
    try:
        local_path = SONGS_DIR / file.filename
        with open(local_path, 'wb') as f:
            f.write(file_data)
        print(f"✅ Saved: {local_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    return UploadResponse(
        success=True,
        url=f"/static/songs/{file.filename}",
        filename=file.filename,
        message="File uploaded successfully"
    )


@router.get("/api/upload/files", response_model=FileListResponse)
async def get_uploaded_files():
    """
    List all uploaded audio files in local storage
    
    Returns:
        List of files with their URLs and sizes
    """
    files = []
    
    try:
        for mp3_file in SONGS_DIR.glob("*.mp3"):
            stat = mp3_file.stat()
            files.append(FileInfo(
                name=mp3_file.name,
                url=f"/static/songs/{mp3_file.name}",
                size=stat.st_size,
                created_at=None
            ))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )
    
    return FileListResponse(
        success=True,
        files=files,
        total=len(files)
    )


@router.delete("/api/upload/{filename}")
async def delete_uploaded_file(filename: str):
    """
    Delete a file from local storage
    
    Args:
        filename: The name of the file to delete (e.g., "song.mp3")
        
    Returns:
        Success message or error
    """
    if not filename.lower().endswith('.mp3'):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename"
        )
    
    file_path = SONGS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )
    
    try:
        file_path.unlink()
        print(f"🗑️ Deleted: {file_path}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )
    
    return {
        "success": True,
        "message": f"Successfully deleted {filename}"
    }
