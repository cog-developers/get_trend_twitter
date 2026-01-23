"""Redis client for caching and job tracking."""

import json
from typing import Optional
import redis
from redis.connection import ConnectionPool
from redis.exceptions import ConnectionError, RedisError
from src.config.settings import REDIS_URL, REDIS_DB
from src.logging.logger import get_logger

logger = get_logger(__name__)

# Global Redis connection pool
_redis_pool: Optional[ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Get or create Redis client with connection pooling."""
    global _redis_client, _redis_pool
    
    if _redis_client is None:
        try:
            # Parse Redis URL and create connection pool
            _redis_pool = ConnectionPool.from_url(
                REDIS_URL,
                db=REDIS_DB,
                decode_responses=False,  # We'll handle encoding ourselves for JSON
                max_connections=50,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            _redis_client = redis.Redis(connection_pool=_redis_pool)
            
            # Test connection
            _redis_client.ping()
            logger.info(f"✅ Connected to Redis (DB: {REDIS_DB})")
        except ConnectionError as e:
            logger.error(f"❌ Redis connection error: {e}")
            logger.error(f"   URL: {REDIS_URL.split('@')[-1] if '@' in REDIS_URL else REDIS_URL}")  # Hide password
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    return _redis_client


def validate_redis_connection() -> bool:
    """Validate Redis connection and return True if successful."""
    try:
        client = get_redis_client()
        client.ping()
        return True
    except Exception as e:
        logger.error(f"❌ Redis validation failed: {e}")
        return False


def set_job_data(job_id: str, data: dict, ttl: int = 86400):
    """
    Store job data in Redis with TTL (default 24 hours).
    
    Args:
        job_id: Unique job identifier
        data: Job data dictionary
        ttl: Time to live in seconds (default: 86400 = 24 hours)
    """
    try:
        client = get_redis_client()
        key = f"job:{job_id}"
        # Serialize to JSON
        value = json.dumps(data, ensure_ascii=False)
        client.setex(key, ttl, value)
    except RedisError as e:
        logger.error(f"❌ Redis error storing job data: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error storing job data in Redis: {e}")
        raise


def get_job_data(job_id: str) -> Optional[dict]:
    """
    Retrieve job data from Redis.
    
    Args:
        job_id: Unique job identifier
    
    Returns:
        Job data dictionary or None if not found
    """
    try:
        client = get_redis_client()
        key = f"job:{job_id}"
        value = client.get(key)
        
        if value is None:
            return None
        
        # Deserialize from JSON
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        return json.loads(value)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decoding job data from Redis: {e}")
        return None
    except RedisError as e:
        logger.error(f"❌ Redis error retrieving job data: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error retrieving job data from Redis: {e}")
        return None


def delete_job_data(job_id: str):
    """Delete job data from Redis."""
    try:
        client = get_redis_client()
        key = f"job:{job_id}"
        client.delete(key)
    except RedisError as e:
        logger.error(f"❌ Redis error deleting job data: {e}")
    except Exception as e:
        logger.error(f"❌ Error deleting job data from Redis: {e}")


def exists_job(job_id: str) -> bool:
    """Check if job exists in Redis."""
    try:
        client = get_redis_client()
        key = f"job:{job_id}"
        return client.exists(key) > 0
    except RedisError as e:
        logger.error(f"❌ Redis error checking job existence: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking job existence in Redis: {e}")
        return False
