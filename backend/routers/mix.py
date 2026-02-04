"""
Mix Router - Mix Generation API
================================

Endpoints for:
- Starting mix generation with natural language prompt
- Checking job status
- Downloading completed mix
- WebSocket for real-time updates
"""

import asyncio
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from backend.services.pipeline_runner import (
    create_job, get_job, run_pipeline_job,
    pause_job, resume_job, cancel_job,
    JobStatus, PIPELINE_STAGES
)
from backend.services.websocket_manager import manager

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"

router = APIRouter()


# ==================== MODELS ====================

class MixRequest(BaseModel):
    """Request to generate a mix"""
    prompt: str
    song_ids: Optional[List[str]] = None  # Optional: specific songs to include


class MixResponse(BaseModel):
    """Response after starting mix generation"""
    job_id: str
    status: str
    message: str
    websocket_url: str


class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    current_stage: int
    current_stage_name: str
    progress_percent: float
    logs: list
    error: Optional[str] = None
    mix_url: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.post("/generate", response_model=MixResponse)
async def generate_mix(request: MixRequest, background_tasks: BackgroundTasks):
    """
    Start generating a DJ mix from natural language prompt.
    
    Example prompts:
    - "Mix all songs"
    - "Create a 5-song mix with upbeat energy"
    - "Start with love me jeje, then sensational, work me out"
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Create job
    job = create_job(request.prompt)
    
    # Start pipeline in background
    background_tasks.add_task(run_pipeline_job, job)
    
    return MixResponse(
        job_id=job.job_id,
        status=job.status.value,
        message=f"Mix generation started! Connect to WebSocket for live updates.",
        websocket_url=f"/ws/mix/{job.job_id}"
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_mix_status(job_id: str):
    """Get current status of a mix generation job"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    stage_name = ""
    if 1 <= job.current_stage <= 5:
        stage_name = PIPELINE_STAGES[job.current_stage - 1].name
    
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        current_stage=job.current_stage,
        current_stage_name=stage_name,
        progress_percent=job.progress_percent,
        logs=job.logs[-10:],  # Last 10 logs
        error=job.error,
        mix_url=job.mix_url
    )


@router.get("/download/{job_id}")
async def download_mix(job_id: str):
    """Download the generated mix file"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=400, detail=f"Mix not ready. Status: {job.status.value}")
    
    # If we have a cloud URL, redirect to it
    if job.mix_url and job.mix_url.startswith("http"):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=job.mix_url)
    
    # Fallback to local file if path is relative
    mix_path = OUTPUT_DIR / "mix.mp3"
    if not mix_path.exists():
        raise HTTPException(status_code=404, detail="Mix file not found locally or in cloud")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(mix_path),
        media_type="audio/mpeg",
        filename=f"ai_dj_mix_{job_id}.mp3"
    )


@router.post("/pause/{job_id}")
async def pause_mix_generation(job_id: str):
    """Pause mix generation"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if pause_job(job_id):
        await manager.send_paused(job_id)
        return {"success": True, "message": "Mix generation paused", "status": "paused"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot pause job. Current status: {job.status.value}")


@router.post("/resume/{job_id}")
async def resume_mix_generation(job_id: str):
    """Resume paused mix generation"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if resume_job(job_id):
        await manager.send_resumed(job_id)
        return {"success": True, "message": "Mix generation resumed", "status": "running"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot resume job. Current status: {job.status.value}")


@router.post("/cancel/{job_id}")
async def cancel_mix_generation(job_id: str):
    """Cancel mix generation"""
    job = get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    if cancel_job(job_id):
        await manager.send_cancelled(job_id)
        return {"success": True, "message": "Mix generation cancelled", "status": "cancelled"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job. Current status: {job.status.value}")


@router.get("/stages")
async def get_pipeline_stages():
    """Get list of pipeline stages"""
    return {
        "stages": [
            {
                "number": stage.number,
                "name": stage.name,
                "description": stage.description
            }
            for stage in PIPELINE_STAGES
        ]
    }


# ==================== WEBSOCKET ====================

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time mix progress updates.
    
    Connect: ws://localhost:8000/api/mix/ws/{job_id}
    
    Messages sent:
    - {"type": "stage_update", "stage": 1-5, "name": "...", "status": "running|complete"}
    - {"type": "log", "message": "...", "level": "info|warning|error"}
    - {"type": "progress", "percent": 0-100}
    - {"type": "complete", "mix_url": "/static/output/mix.mp3"}
    - {"type": "error", "message": "..."}
    """
    # Validate job exists
    job = get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return
    
    await manager.connect(websocket, job_id)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "job_id": job_id,
            "status": job.status.value,
            "current_stage": job.current_stage,
            "progress_percent": job.progress_percent
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong or close)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Echo ping
                if data == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception:
        manager.disconnect(websocket, job_id)
