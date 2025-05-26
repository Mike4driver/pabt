import threading
import uuid
from datetime import timedelta, datetime
from typing import Optional, Dict, Any

# --- Background Job Tracking System ---
background_jobs: Dict[str, 'BackgroundJob'] = {}
job_lock = threading.Lock()

class BackgroundJob:
    def __init__(self, job_id: str, job_type: str, description: str):
        self.job_id = job_id
        self.job_type = job_type
        self.description = description
        self.status = "running"  # running, completed, failed
        self.progress = 0
        self.total = 0
        self.current_item = ""
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.result: Dict[str, Any] = {}
        self.error: Optional[str] = None

    def update_progress(self, current: int, total: int, current_item: str = ""):
        self.progress = current
        self.total = total
        self.current_item = current_item
        # Ensure progress does not exceed total if completion is handled separately
        if self.status == "running" and self.progress == self.total and self.total > 0 :
             # If progress reaches total, but not explicitly completed, consider it 99% to avoid confusion
             # Or, rely on explicit complete/fail calls. For now, let's allow progress to reach total.
             pass


    def complete(self, result: Optional[Dict[str, Any]] = None):
        self.status = "completed"
        self.end_time = datetime.now()
        self.result = result if result is not None else {}
        self.progress = self.total # Ensure progress shows 100%

    def fail(self, error: str):
        self.status = "failed"
        self.end_time = datetime.now()
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "description": self.description,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "current_item": self.current_item,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "result": self.result,
            "error": self.error
        }

def create_background_job(job_type: str, description: str) -> BackgroundJob:
    job_id = str(uuid.uuid4())
    job = BackgroundJob(job_id, job_type, description)
    with job_lock:
        background_jobs[job_id] = job
    return job

def get_background_job(job_id: str) -> Optional[BackgroundJob]:
    with job_lock:
        return background_jobs.get(job_id)

def cleanup_old_jobs():
    """Remove jobs older than 1 hour that are completed or failed"""
    cutoff = datetime.now() - timedelta(hours=1)
    with job_lock:
        to_remove = []
        for job_id, job in background_jobs.items():
            if job.status in ["completed", "failed"] and job.end_time and job.end_time < cutoff:
                to_remove.append(job_id)
        for job_id in to_remove:
            del background_jobs[job_id] 