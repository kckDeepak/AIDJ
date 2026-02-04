"""
Upload Router - Cloud Storage Upload API
=========================================

Endoint to upload files to Supabase Storage.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.services.supabase_client import supabase_service

router = APIRouter(tags=["upload"])

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
    Upload an MP3 file to Supabase Storage
    """
    
    print(f"🎯 UPLOAD: {file.filename}")
    
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith('.mp3'):
        raise HTTPException(
            status_code=400,
            detail="Only MP3 files are allowed. Please upload a .mp3 file."
        )
    
    filename = file.filename
    
    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
            
        # Upload to Supabase
        public_url = supabase_service.upload_file("songs_bucket_1", tmp_path, filename)
        
        # Clean up
        if tmp_path.exists():
            tmp_path.unlink()
            
        if not public_url:
             raise HTTPException(status_code=500, detail="Failed to get public URL from storage")

        return UploadResponse(
            success=True,
            url=public_url,
            filename=filename,
            message="File uploaded successfully"
        )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload file: {str(e)}"
        )


@router.get("/api/upload/files", response_model=FileListResponse)
async def get_uploaded_files():
    """
    List all uploaded audio files in storage
    """
    files = []
    
    try:
        storage_files = supabase_service.list_files("songs_bucket_1")
        for f in storage_files:
            name = f.get('name')
            if name and name.endswith('.mp3'):
                files.append(FileInfo(
                    name=name,
                    url="", # We don't fetch full URL here to save time, or we could
                    size=f.get('metadata', {}).get('size', 0),
                    created_at=f.get('created_at')
                ))

    except Exception as e:
        print(f"List error: {e}")
        # Return empty list instead of crashing
        pass
    
    return FileListResponse(
        success=True,
        files=files,
        total=len(files)
    )


@router.delete("/api/upload/{filename}")
async def delete_uploaded_file(filename: str):
    """
    Delete a file from storage
    """
    # Placeholder for delete
    return {
        "success": False,
        "message": "Delete not fully implemented for cloud storage in this router"
    }
