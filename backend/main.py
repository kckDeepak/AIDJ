"""
FastAPI Backend for AI DJ Mixing System
========================================

Main entry point providing:
- REST API for song management and mix generation
- WebSocket for real-time pipeline progress
- Static file serving for audio files
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# Add parent directory to path for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.routers import songs, mix, upload
from backend.services.websocket_manager import manager

# Paths - detect base directory properly for both local and Render
import os
cwd = Path.cwd()
if os.environ.get('RENDER'):
    # On Render, use the project src directory
    BASE_DIR = Path('/opt/render/project/src')
elif (cwd / 'songs').exists():
    BASE_DIR = cwd
elif (cwd.parent / 'songs').exists():
    BASE_DIR = cwd.parent
else:
    BASE_DIR = Path(__file__).parent.parent

SONGS_DIR = BASE_DIR / "songs"
OUTPUT_DIR = BASE_DIR / "output"
NOTES_DIR = BASE_DIR / "notes"

# Ensure directories exist with parents=True for Render deployment
SONGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

# Create placeholder files if needed
(OUTPUT_DIR / ".gitkeep").touch(exist_ok=True)
(SONGS_DIR / ".gitkeep").touch(exist_ok=True)

print(f"[STARTUP] BASE_DIR: {BASE_DIR}")
print(f"[STARTUP] SONGS_DIR: {SONGS_DIR} (exists: {SONGS_DIR.exists()})")
print(f"[STARTUP] OUTPUT_DIR: {OUTPUT_DIR} (exists: {OUTPUT_DIR.exists()})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Re-ensure directories exist at startup
    SONGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🎧 AI DJ Mixing System Backend Starting...")
    print(f"   Songs directory: {SONGS_DIR} (exists: {SONGS_DIR.exists()})")
    print(f"   Output directory: {OUTPUT_DIR} (exists: {OUTPUT_DIR.exists()})")
    yield
    print("🎧 AI DJ Mixing System Backend Shutting Down...")


# Create FastAPI app
app = FastAPI(
    title="AI DJ Mixing System API",
    description="Professional AI-powered DJ mixing pipeline with real-time progress tracking",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers for audio playback
)

# Mount static directories for serving audio files
app.mount("/static/songs", StaticFiles(directory=str(SONGS_DIR)), name="songs")
app.mount("/static/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")

# Include routers
app.include_router(songs.router, prefix="/api/songs", tags=["Songs"])
app.include_router(mix.router, prefix="/api/mix", tags=["Mix Generation"])
app.include_router(upload.router)  # Supabase Storage - POST /upload-audio

print("✅ Routers registered:")
print("   - /api/songs/* (songs)")
print("   - /api/mix/* (mix)")
print("   - /upload-audio (upload)")
print("   - /api/upload/files (list)")
print("   - /api/upload/{filename} (delete)")


@app.get("/static/output/mix.mp3")
async def serve_mix():
    """Serve generated mix with proper audio headers and CORS"""
    mix_file = OUTPUT_DIR / "mix.mp3"
    
    if not mix_file.exists():
        return Response(
            content="Mix not found. Generate a mix first.",
            status_code=404,
            media_type="text/plain"
        )
    
    return FileResponse(
        path=str(mix_file),
        media_type="audio/mpeg",
        headers={
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cache-Control": "no-cache",
        }
    )


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "AI DJ Mixing System",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    songs_count = len(list(SONGS_DIR.glob("*.mp3")))
    return {
        "status": "healthy",
        "songs_available": songs_count,
        "output_ready": (OUTPUT_DIR / "mix.mp3").exists()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
