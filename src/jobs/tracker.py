"""
Job tracking module.
Handles job state management for background processing.
"""

import hashlib
import threading
from datetime import datetime
from typing import Optional, List, Dict

# In-memory job storage (can be replaced with Redis/database for production)
jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()


def generate_job_id(user_input_id: Optional[str], source_ids: Optional[List[str]]) -> str:
    """Generate a unique job ID based on filters."""
    filter_str = f"{user_input_id}_{sorted(source_ids) if source_ids else 'all'}"
    return hashlib.md5(filter_str.encode()).hexdigest()


def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status by job_id."""
    with jobs_lock:
        return jobs.get(job_id)


def create_job(job_id: str, user_input_id: Optional[str], source_ids: Optional[List[str]]) -> Dict:
    """Create a new job entry."""
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "user_input_id": user_input_id,
            "source_ids": source_ids,
            "created_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress": 0,
            "estimated_time_remaining": None
        }
        return jobs[job_id]


def update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
    progress: int = 0
):
    """Update job status."""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            if result:
                jobs[job_id]["result"] = result
            if error:
                jobs[job_id]["error"] = error
            jobs[job_id]["progress"] = progress
            if status in ["completed", "failed"]:
                jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()


def clear_job(job_id: str):
    """Remove a job from the tracker."""
    with jobs_lock:
        if job_id in jobs:
            del jobs[job_id]


def get_all_jobs() -> Dict[str, Dict]:
    """Get all jobs (copy)."""
    with jobs_lock:
        return dict(jobs)
