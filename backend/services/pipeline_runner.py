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
from backend.services.supabase_client import supabase_service


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
        cwd = Path.cwd()
        
        # Check if we're running from backend directory or project root
        if (cwd / 'songs').exists():
            self.base_dir = cwd
        elif (cwd.parent / 'songs').exists():
            self.base_dir = cwd.parent
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
    
    async def run(self) -> bool:
        """Execute the full pipeline"""
        
        # Import existing pipeline modules
        try:
            await self.log(f"📦 Importing pipeline modules...")
            await self.log(f"   sys.path[0]: {sys.path[0]}")
            await self.log(f"   Current working directory: {os.getcwd()}")
            
            from track_analysis_openai_approach import combined_engine
            await self.log("   ✓ track_analysis_openai_approach")
            
            from bpm_lookup import process_bpm_lookup
            await self.log("   ✓ bpm_lookup")
            
            from structure_detector import process_structure_detection
            await self.log("   ✓ structure_detector")
            
            from generate_mixing_plan import generate_mixing_plan
            await self.log("   ✓ generate_mixing_plan")
            
            from mixing_engine import generate_mix
            await self.log("   ✓ mixing_engine")
            
            await self.log("✅ All pipeline modules imported successfully")
        except ImportError as e:
            await self.log(f"❌ Failed to import pipeline modules: {e}", "error")
            import traceback
            await self.log(traceback.format_exc(), "error")
            
            self.job.status = JobStatus.FAILED
            self.job.error = f"Import failed: {str(e)}"
            self.job.completed_at = datetime.now()
            await manager.send_error(self.job.job_id, self.job.error)
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
        
        # Check if songs exist locally
        song_files = list(songs_dir.glob("*.mp3"))
        await self.log(f"Found {len(song_files)} song(s) in local storage")
        
        if len(song_files) == 0:
            error_msg = "No songs found! Please upload MP3 files first."
            await self.log(error_msg, "error")
            self.job.status = JobStatus.FAILED
            self.job.error = error_msg
            self.job.completed_at = datetime.now()
            await manager.send_error(self.job.job_id, error_msg)
            return False
        
        try:
            self.job.status = JobStatus.RUNNING
            
            # Sync songs from Supabase to ensure we have all files locally for processing
            await self.log("☁️ Syncing songs from Supabase Storage...")
            files = supabase_service.list_files("songs_bucket_1")
            synced_count = 0
            for file in files:
                fname = file.get('name')
                if fname and fname.endswith('.mp3'):
                    local_fpath = songs_dir / fname
                    if not local_fpath.exists():
                        if supabase_service.download_file("songs_bucket_1", fname, local_fpath):
                            synced_count += 1
            if synced_count > 0:
                await self.log(f"   ✓ Downloaded {synced_count} songs from cloud")
            
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
            
            try:
                await asyncio.to_thread(
                    generate_mixing_plan,
                    basic_setlist_path=str(output_dir / "basic_setlist.json"),
                    structure_json_path=str(output_dir / "structure_data.json"),
                    output_path=str(output_dir / "mixing_plan.json"),
                    songs_dir=str(songs_dir)
                )
            except Exception as plan_error:
                error_msg = f"Mixing plan generation failed: {str(plan_error)}"
                await self.log(error_msg, "error")
                import traceback
                await self.log(traceback.format_exc(), "error")
                
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            # Verify mixing plan was created
            mixing_plan_path = output_dir / "mixing_plan.json"
            if not mixing_plan_path.exists():
                error_msg = "Mixing plan file was not created"
                await self.log(error_msg, "error")
                
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                
                await manager.send_error(self.job.job_id, error_msg)
                return False
                
            await self.update_stage(4, "complete")
            await self.log("Mixing plan ready ✓")
            
            # Stage 5: Final Mix Generation
            await self.update_stage(5, "running")
            await self.log("Creating final audio mix... This may take a few minutes.")
            
            mix_path = output_dir / "mix.mp3"
            mixing_plan_path = output_dir / "mixing_plan.json"
            structure_data_path = output_dir / "structure_data.json"
            
            # Verify all required files exist
            await self.log(f"📋 Verifying input files...")
            if not mixing_plan_path.exists():
                error_msg = f"Mixing plan not found: {mixing_plan_path}"
                await self.log(error_msg, "error")
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            if not structure_data_path.exists():
                error_msg = f"Structure data not found: {structure_data_path}"
                await self.log(error_msg, "error")
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            await self.log(f"   ✓ {mixing_plan_path.name} ({mixing_plan_path.stat().st_size} bytes)")
            await self.log(f"   ✓ {structure_data_path.name} ({structure_data_path.stat().st_size} bytes)")
            
            # Verify songs directory has files
            song_files = list(songs_dir.glob("*.mp3"))
            await self.log(f"   ✓ Found {len(song_files)} song(s) in {songs_dir}")
            for song_file in song_files:
                await self.log(f"      - {song_file.name} ({song_file.stat().st_size / 1024 / 1024:.2f} MB)")
            
            # Remove old mix if exists
            if mix_path.exists():
                mix_path.unlink()
                await self.log("🗑️ Removed old mix file")
            
            await self.log("🎵 Starting mix generation...")
            try:
                await asyncio.to_thread(
                    generate_mix,
                    mixing_plan_json=str(mixing_plan_path),
                    structure_json=str(structure_data_path),
                    output_path=str(mix_path)
                )
            except Exception as mix_error:
                error_msg = f"Mix generation failed: {str(mix_error)}"
                await self.log(error_msg, "error")
                import traceback
                await self.log(traceback.format_exc(), "error")
                
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            # Verify mix was actually created
            if not mix_path.exists():
                error_msg = "Mix generation reported success but file was not created"
                await self.log(error_msg, "error")
                
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            file_size = mix_path.stat().st_size
            if file_size == 0:
                error_msg = "Mix file was created but is empty (0 bytes)"
                await self.log(error_msg, "error")
                
                self.job.status = JobStatus.FAILED
                self.job.error = error_msg
                self.job.completed_at = datetime.now()
                
                await manager.send_error(self.job.job_id, error_msg)
                return False
            
            await self.update_stage(5, "complete")
            await self.log(f"Mix generation complete! 🎧 ({file_size / 1024 / 1024:.2f} MB)")
            
            # Upload Mix to Supabase
            await self.log("☁️ Uploading mix to Supabase Storage...")
            mix_filename = f"mix_{self.job.job_id}.mp3"
            mix_url = supabase_service.upload_file("mixes_bucket_1", mix_path, mix_filename)
            
            if not mix_url:
                await self.log("⚠️ Upload failed, falling back to local URL", "warning")
                mix_url = "/static/output/mix.mp3"
            else:
                await self.log(f"   ✓ Uploaded: {mix_url}")
            
            await self.log(f"🔗 Mix URL: {mix_url}")
            
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
