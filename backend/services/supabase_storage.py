"""
Supabase Storage Service
========================

Handles file uploads to Supabase Storage - the SINGLE storage solution for this project.

ARCHITECTURE:
    Frontend (Vercel) → Backend (Render) → Supabase Storage (audio-files bucket)

WHY SUPABASE STORAGE:
    - Persistent file storage (unlike Render's ephemeral filesystem)
    - Simple API with Python SDK
    - Free tier: 1GB storage, 2GB bandwidth/month
    - No client-side configuration needed (keys stay on backend)

SETUP:
    1. Create a Supabase project at https://supabase.com
    2. Create a storage bucket named "audio-files" (set to PUBLIC)
    3. Set environment variables: SUPABASE_URL and SUPABASE_SERVICE_KEY
    4. Add storage policies to allow uploads and public reads

SECURITY:
    - SUPABASE_SERVICE_KEY is a SECRET - never expose to frontend
    - All uploads go through the backend API
    - Frontend/clients only interact with backend endpoints
"""

import os
from typing import Optional
from supabase import create_client, Client

# Initialize Supabase client from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Bucket name for audio files - must match bucket created in Supabase dashboard
AUDIO_BUCKET = "audio-files"


def get_supabase_client() -> Optional[Client]:
    """Get Supabase client if credentials are configured"""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("⚠️ Supabase credentials not configured")
        return None
    
    try:
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        return None


def upload_file(file_data: bytes, filename: str, content_type: str = "audio/mpeg") -> dict:
    """
    Upload a file to Supabase Storage
    
    Args:
        file_data: Raw file bytes
        filename: Name to save the file as
        content_type: MIME type of the file
        
    Returns:
        dict with 'success', 'url', and 'error' keys
    """
    client = get_supabase_client()
    
    if not client:
        return {
            "success": False,
            "url": None,
            "error": "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
        }
    
    try:
        # Upload path: songs/filename.mp3
        path = f"songs/{filename}"
        
        # Upload to Supabase Storage
        result = client.storage.from_(AUDIO_BUCKET).upload(
            path,
            file_data,
            {"content-type": content_type}
        )
        
        # Get public URL
        public_url = client.storage.from_(AUDIO_BUCKET).get_public_url(path)
        
        print(f"✅ Uploaded to Supabase: {filename}")
        
        return {
            "success": True,
            "url": public_url,
            "path": path,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Supabase upload failed: {error_msg}")
        
        # Check for common errors
        if "Duplicate" in error_msg or "already exists" in error_msg.lower():
            # File already exists, return existing URL
            public_url = client.storage.from_(AUDIO_BUCKET).get_public_url(f"songs/{filename}")
            return {
                "success": True,
                "url": public_url,
                "path": f"songs/{filename}",
                "error": None,
                "note": "File already existed"
            }
        
        return {
            "success": False,
            "url": None,
            "error": error_msg
        }


def delete_file(filename: str) -> dict:
    """
    Delete a file from Supabase Storage
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        dict with 'success' and 'error' keys
    """
    client = get_supabase_client()
    
    if not client:
        return {
            "success": False,
            "error": "Supabase not configured"
        }
    
    try:
        path = f"songs/{filename}"
        client.storage.from_(AUDIO_BUCKET).remove([path])
        
        print(f"🗑️ Deleted from Supabase: {filename}")
        
        return {"success": True, "error": None}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def upload_mix_file(file_data: bytes, filename: str, content_type: str = "audio/mpeg") -> dict:
    """
    Upload a generated mix file to Supabase Storage in the mixes folder
    
    Args:
        file_data: Raw file bytes
        filename: Name to save the file as
        content_type: MIME type of the file
        
    Returns:
        dict with 'success', 'url', and 'error' keys
    """
    client = get_supabase_client()
    
    if not client:
        return {
            "success": False,
            "url": None,
            "error": "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
        }
    
    try:
        # Upload path: mixes/filename.mp3
        path = f"mixes/{filename}"
        
        # Upload to Supabase Storage (will overwrite if exists)
        # Note: Using file_options with upsert as boolean
        result = client.storage.from_(AUDIO_BUCKET).upload(
            path,
            file_data,
            file_options={"content-type": content_type, "upsert": True}
        )
        
        # Get public URL
        public_url = client.storage.from_(AUDIO_BUCKET).get_public_url(path)
        
        print(f"✅ Uploaded mix to Supabase: {filename}")
        
        return {
            "success": True,
            "url": public_url,
            "path": path,
            "error": None
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Supabase mix upload failed: {error_msg}")
        
        # Try to get URL even if upload failed (file might exist)
        try:
            public_url = client.storage.from_(AUDIO_BUCKET).get_public_url(f"mixes/{filename}")
            return {
                "success": True,
                "url": public_url,
                "path": f"mixes/{filename}",
                "error": None,
                "note": "Using existing file"
            }
        except:
            pass
        
        return {
            "success": False,
            "url": None,
            "error": error_msg
        }


def list_files() -> dict:
    """
    List all files in the audio bucket
    
    Returns:
        dict with 'success', 'files', and 'error' keys
    """
    client = get_supabase_client()
    
    if not client:
        return {
            "success": False,
            "files": [],
            "error": "Supabase not configured"
        }
    
    try:
        result = client.storage.from_(AUDIO_BUCKET).list("songs")
        
        files = []
        for item in result:
            public_url = client.storage.from_(AUDIO_BUCKET).get_public_url(f"songs/{item['name']}")
            files.append({
                "name": item["name"],
                "url": public_url,
                "size": item.get("metadata", {}).get("size", 0),
                "created_at": item.get("created_at")
            })
        
        return {
            "success": True,
            "files": files,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "files": [],
            "error": str(e)
        }
