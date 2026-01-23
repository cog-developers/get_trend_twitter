"""Job tracking system using Redis for persistence."""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from src.infra.redis_client import (
    set_job_data,
    get_job_data,
    delete_job_data,
    exists_job
)
from src.logging.logger import get_logger

logger = get_logger(__name__)

# Job TTL: 7 days (604800 seconds)
JOB_TTL = 604800

# Stale job timeout: Consider a job stale if it's been processing for more than 2 hours
STALE_JOB_TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours


def generate_job_id(user_input_id: Optional[str], source_ids: Optional[List[str]]) -> str:
    """Generate a unique job ID based on filters"""
    filter_str = f"{user_input_id}_{sorted(source_ids) if source_ids else 'all'}"
    return hashlib.md5(filter_str.encode()).hexdigest()


def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status by job_id from Redis"""
    return get_job_data(job_id)


def create_job(job_id: str, user_input_id: Optional[str], source_ids: Optional[List[str]]) -> Dict:
    """Create a new job entry in Redis"""
    now = datetime.now(timezone.utc)
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "user_input_id": user_input_id,
        "source_ids": source_ids,
        "created_at": now.isoformat(),
        "last_updated_at": now.isoformat(),  # Track last update time
        "finished_at": None,
        "result": None,
        "error": None,
        "progress": 0,
        "estimated_time_remaining": None
    }
    
    set_job_data(job_id, job_data, ttl=JOB_TTL)
    logger.info(f"📝 Created job {job_id} in Redis")
    return job_data


def update_job_status(
    job_id: str, 
    status: str, 
    result: Optional[Dict] = None, 
    error: Optional[str] = None, 
    progress: int = 0
):
    """Update job status in Redis"""
    job_data = get_job_data(job_id)
    now = datetime.utcnow()
    
    if job_data is None:
        logger.warning(f"⚠️ Job {job_id} not found in Redis, creating new entry")
        # Create a basic job entry if it doesn't exist
        job_data = {
            "job_id": job_id,
            "status": status,
            "user_input_id": None,
            "source_ids": None,
            "created_at": now.isoformat(),
            "last_updated_at": now.isoformat(),
            "finished_at": None,
            "result": None,
        }
    
    # Update fields
    job_data["status"] = status
    job_data["progress"] = progress
    job_data["last_updated_at"] = now.isoformat()  # Always update timestamp
    
    if result is not None:
        job_data["result"] = result
    
    if error is not None:
        job_data["error"] = error
    
    if status in ["completed", "failed"]:
        job_data["finished_at"] = now.isoformat()
    
    # Save back to Redis
    set_job_data(job_id, job_data, ttl=JOB_TTL)
    logger.debug(f"📝 Updated job {job_id} status: {status} (progress: {progress}%)")


def is_job_stale(job_data: Dict) -> bool:
    """
    Check if a job is stale (has been processing for too long without updates).
    
    Args:
        job_data: Job data dictionary from Redis
    
    Returns:
        True if job is stale, False otherwise
    """
    if job_data is None:
        return False
    
    status = job_data.get("status")
    if status not in ["pending", "processing"]:
        return False  # Only check active jobs
    
    # Check last_updated_at timestamp
    last_updated_str = job_data.get("last_updated_at")
    if not last_updated_str:
        # Fallback to created_at if last_updated_at doesn't exist (old jobs)
        last_updated_str = job_data.get("created_at")
        if not last_updated_str:
            return True  # No timestamp = consider stale
    
    try:
        # Parse ISO format timestamp
        if 'Z' in last_updated_str:
            last_updated_str = last_updated_str.replace('Z', '+00:00')
        
        last_updated = datetime.fromisoformat(last_updated_str)
        
        # Ensure timezone-aware
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        age_seconds = (now - last_updated).total_seconds()
        
        return age_seconds > STALE_JOB_TIMEOUT_SECONDS
    except (ValueError, TypeError) as e:
        logger.warning(f"⚠️ Error parsing timestamp for job {job_data.get('job_id')}: {e}")
        return True  # If we can't parse, consider stale


def reset_stale_job(job_id: str) -> bool:
    """
    Reset a stale job so it can be reprocessed.
    
    Args:
        job_id: Job ID to reset
    
    Returns:
        True if job was reset, False otherwise
    """
    job_data = get_job_data(job_id)
    if not job_data:
        return False
    
    if not is_job_stale(job_data):
        return False
    
    logger.warning(f"🔄 Resetting stale job {job_id} (was {job_data.get('status')} since {job_data.get('last_updated_at')})")
    
    # Delete the stale job so it can be recreated
    delete_job_data(job_id)
    return True
