"""
Pipeline Runner Service
=======================

Wraps the existing run_pipeline.py as an async task
with progress tracking and WebSocket updates.
"""

import asyncio
import os
import uuid
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
from enum import Enum

# Add parent directory for importing existing modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.services.websocket_manager import manager


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETE = "complete"
    FAILED = "failed"


class PipelineStage:
    """Pipeline stage information"""
    def __init__(self, number: int, name: str, description: str):
        self.number = number
        self.name = name
        self.description = description


# Define pipeline stages
PIPELINE_STAGES = [
    PipelineStage(1, "Song Selection", "Selecting songs based on user request"),
    PipelineStage(2, "BPM Analysis", "Analyzing BPM and metadata"),
    PipelineStage(3, "Structure Detection", "Detecting transition points and vocals"),
    PipelineStage(4, "Mix Planning", "Generating professional mixing plan"),
    PipelineStage(5, "Mix Generation", "Creating final audio mix"),
]


class MixJob:
    """Represents a mix generation job"""
    def __init__(self, job_id: str, prompt: str):
        self.job_id = job_id
        self.prompt = prompt
        self.status = JobStatus.PENDING
        self.current_stage = 0
        self.progress_percent = 0
        self.logs: list = []
        self.error: Optional[str] = None
        self.mix_url: Optional[str] = None
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.is_paused = False
        self.is_cancelled = False
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "progress_percent": self.progress_percent,
            "logs": self.logs[-20:],  # Last 20 logs
            "error": self.error,
            "mix_url": self.mix_url,
            "is_paused": self.is_paused,
            "is_cancelled": self.is_cancelled,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


# Global job store
jobs: Dict[str, MixJob] = {}


class PipelineRunner:
    """Runs the DJ mixing pipeline with progress tracking"""
    
    def __init__(self, job: MixJob):
        self.job = job
        # Detect base directory - use current working directory approach
        # This works both locally and on Render
        cwd = Path.cwd()
        
        # Check if we're running from backend directory or project root
        if (cwd / 'songs').exists():
            self.base_dir = cwd
        elif (cwd.parent / 'songs').exists():
            self.base_dir = cwd.parent
        elif os.environ.get('RENDER'):
            # Fallback for Render
            self.base_dir = Path('/opt/render/project/src')
        else:
            # Default fallback
            self.base_dir = Path(__file__).parent.parent.parent
        
        # Create directories immediately
        (self.base_dir / 'output').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'songs').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'notes').mkdir(parents=True, exist_ok=True)
    
    async def log(self, message: str, level: str = "info"):
        """Log message and broadcast via WebSocket"""
        self.job.logs.append({"message": message, "level": level, "time": datetime.now().isoformat()})
        await manager.send_log(self.job.job_id, message, level)
        print(f"[{self.job.job_id}] {level.upper()}: {message}")
    
    async def update_stage(self, stage_num: int, status: str = "running"):
        """Update current stage and broadcast"""
        self.job.current_stage = stage_num
        stage = PIPELINE_STAGES[stage_num - 1]
        await manager.send_stage_update(self.job.job_id, stage_num, stage.name, status)
        
        # Update progress percent based on stage
        if status == "running":
            self.job.progress_percent = (stage_num - 1) * 20
        elif status == "complete":
            self.job.progress_percent = stage_num * 20
        
        await manager.send_progress(self.job.job_id, self.job.progress_percent, stage_num)
    
    async def download_songs_from_supabase(self, songs_dir: Path):
        """Download all songs from Supabase to local directory for processing"""
        try:
            from backend.services.supabase_storage import list_files, get_supabase_client
            
            await self.log("Downloading songs from Supabase Storage...")
            
            result = list_files()
            if not result["success"]:
                await self.log(f"Warning: Could not list files from Supabase: {result.get('error')}", "warning")
                return
            
            files = result["files"]
            await self.log(f"Found {len(files)} songs in Supabase Storage")
            
            client = get_supabase_client()
            if not client:
                await self.log("Warning: Supabase client not available", "warning")
                return
            
            # Download each song
            for file_info in files:
                filename = file_info["name"]
                local_path = songs_dir / filename
                
                # Skip if already exists and is not empty
                if local_path.exists() and local_path.stat().st_size > 0:
                    await self.log(f"Skipping {filename} (already exists locally)")
                    continue
                
                try:
                    # Download from Supabase
                    file_data = client.storage.from_("audio-files").download(f"songs/{filename}")
                    
                    # Save locally
                    with open(local_path, 'wb') as f:
                        f.write(file_data)
                    
                    await self.log(f"Downloaded {filename} ({len(file_data)} bytes)")
                except Exception as e:
                    await self.log(f"Failed to download {filename}: {e}", "error")
            
            await self.log("✅ All songs downloaded from Supabase")
        except Exception as e:
            await self.log(f"Warning: Error downloading songs from Supabase: {e}", "warning")
    
    async def run(self) -> bool:
        """Execute the full pipeline"""
        
        # Import existing pipeline modules
        try:
            from track_analysis_openai_approach import combined_engine
            from bpm_lookup import process_bpm_lookup
            from structure_detector import process_structure_detection
            from generate_mixing_plan import generate_mixing_plan
            from mixing_engine import generate_mix
        except ImportError as e:
            await self.log(f"Failed to import pipeline modules: {e}", "error")
            return False
        
        output_dir = self.base_dir / "output"
        songs_dir = self.base_dir / "songs"
        notes_dir = self.base_dir / "notes"
        
        # Ensure all required directories exist with explicit creation
        for dir_path in [output_dir, songs_dir, notes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            await self.log(f"Directory {dir_path}: exists={dir_path.exists()}")
        
        await self.log(f"Base directory: {self.base_dir}")
        await self.log(f"Output directory: {output_dir}")
        await self.log(f"Songs directory: {songs_dir}")
        
        # Download all songs from Supabase before processing
        await self.download_songs_from_supabase(songs_dir)
        
        try:
            self.job.status = JobStatus.RUNNING
            
            # Stage 1: Song Selection
            await self.update_stage(1, "running")
            await self.log("Starting song selection based on your request...")
            
            await asyncio.to_thread(
                combined_engine,
                self.job.prompt,
                output_path=str(output_dir / "analyzed_setlist.json"),
                songs_dir=str(songs_dir)
            )
            await self.update_stage(1, "complete")
            await self.log("Song selection complete ✓")
            
            # Stage 2: BPM Analysis
            await self.update_stage(2, "running")
            await self.log("Analyzing BPM and metadata...")
            
            await asyncio.to_thread(
                process_bpm_lookup,
                str(output_dir / "analyzed_setlist.json"),
                str(output_dir / "basic_setlist.json")
            )
            await self.update_stage(2, "complete")
            await self.log("BPM analysis complete ✓")
            
            # Stage 3: Structure Detection
            await self.update_stage(3, "running")
            await self.log("Detecting song structures and transition points...")
            
            await asyncio.to_thread(
                process_structure_detection,
                str(output_dir / "basic_setlist.json"),
                str(output_dir / "structure_data.json")
            )
            await self.update_stage(3, "complete")
            await self.log("Structure detection complete ✓")
            
            # Stage 4: Mix Planning
            await self.update_stage(4, "running")
            await self.log("Generating professional mixing plan...")
            
            await asyncio.to_thread(
                generate_mixing_plan,
                basic_setlist_path=str(output_dir / "basic_setlist.json"),
                structure_json_path=str(output_dir / "structure_data.json"),
                output_path=str(output_dir / "mixing_plan.json")
            )
            await self.update_stage(4, "complete")
            await self.log("Mixing plan ready ✓")
            
            # Stage 5: Final Mix Generation
            await self.update_stage(5, "running")
            await self.log("Creating final audio mix... This may take a few minutes.")
            
            mix_path = output_dir / "mix.mp3"
            await asyncio.to_thread(
                generate_mix,
                mixing_plan_json=str(output_dir / "mixing_plan.json"),
                structure_json=str(output_dir / "structure_data.json"),
                output_path=str(mix_path)
            )
            await self.update_stage(5, "complete")
            await self.log("Mix generation complete! 🎧")
            
            # Upload final mix to Supabase for persistent storage
            mix_url = "/static/output/mix.mp3"  # Default fallback
            try:
                from backend.services.supabase_storage import upload_mix_file
                
                if mix_path.exists():
                    await self.log("Uploading final mix to Supabase...")
                    
                    with open(mix_path, "rb") as f:
                        mix_data = f.read()
                    
                    # Upload with timestamp to avoid conflicts
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    mix_filename = f"mix_{timestamp}.mp3"
                    
                    result = await asyncio.to_thread(
                        upload_mix_file,
                        mix_data,
                        mix_filename,
                        "audio/mpeg"
                    )
                    
                    if result.get("url"):
                        mix_url = result["url"]
                        await self.log(f"✅ Mix uploaded to Supabase: {mix_filename}")
                        await self.log(f"🔗 Mix URL: {mix_url}")
                        
                        # Validate URL format
                        if not mix_url.startswith(("http://", "https://")):
                            await self.log(f"⚠️ Invalid URL format: {mix_url}", "warning")
                    else:
                        error_detail = result.get('error', 'Unknown error')
                        await self.log(f"⚠️ Supabase upload failed: {error_detail}", "error")
                        await self.log(f"⚠️ Upload result: {result}", "error")
                        
            except Exception as e:
                await self.log(f"⚠️ Could not upload to Supabase: {e}")
                import traceback
                await self.log(traceback.format_exc(), "error")
                # Continue anyway, use local URL
            
            # Complete
            self.job.status = JobStatus.COMPLETE
            self.job.mix_url = mix_url
            self.job.completed_at = datetime.now()
            self.job.progress_percent = 100
            
            await manager.send_complete(self.job.job_id, self.job.mix_url)
            return True
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            await self.log(error_msg, "error")
            await self.log(traceback.format_exc(), "error")
            
            self.job.status = JobStatus.FAILED
            self.job.error = error_msg
            self.job.completed_at = datetime.now()
            
            await manager.send_error(self.job.job_id, error_msg)
            return False


def create_job(prompt: str) -> MixJob:
    """Create a new mix generation job"""
    job_id = str(uuid.uuid4())[:8]
    job = MixJob(job_id, prompt)
    jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[MixJob]:
    """Get job by ID"""
    return jobs.get(job_id)


def pause_job(job_id: str) -> bool:
    """Pause a running job"""
    job = jobs.get(job_id)
    if job and job.status == JobStatus.RUNNING:
        job.is_paused = True
        job.status = JobStatus.PAUSED
        return True
    return False


def resume_job(job_id: str) -> bool:
    """Resume a paused job"""
    job = jobs.get(job_id)
    if job and job.status == JobStatus.PAUSED:
        job.is_paused = False
        job.status = JobStatus.RUNNING
        return True
    return False


def cancel_job(job_id: str) -> bool:
    """Cancel a running or paused job"""
    job = jobs.get(job_id)
    if job and job.status in [JobStatus.RUNNING, JobStatus.PAUSED, JobStatus.PENDING]:
        job.is_cancelled = True
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()
        return True
    return False


async def run_pipeline_job(job: MixJob):
    """Run pipeline in background"""
    runner = PipelineRunner(job)
    await runner.run()
