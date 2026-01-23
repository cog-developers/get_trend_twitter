"""
Redis caching module for embeddings.
Handles storing and retrieving embeddings from Redis.
"""

import json
from typing import List, Dict, Optional

import numpy as np
import redis

from ..config.settings import (
    REDIS_URL,
    REDIS_DB,
    REDIS_EMBEDDING_PREFIX,
    REDIS_EMBEDDING_TTL,
    EMBEDDING_DIM,
    get_logger
)

logger = get_logger(__name__)

# Global Redis client
_redis_client: Optional[redis.Redis] = None


def create_redis_client() -> redis.Redis:
    """Create and return Redis client."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            db=REDIS_DB,
            decode_responses=False,  # We need bytes for numpy arrays
            socket_timeout=10,
            socket_connect_timeout=10,
            retry_on_timeout=True
        )
        # Test connection
        _redis_client.ping()
        logger.info("Connected to Redis")
        return _redis_client
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise


def get_redis_client() -> redis.Redis:
    """Get the Redis client (creates if not exists)."""
    global _redis_client
    if _redis_client is None:
        return create_redis_client()
    return _redis_client


def _make_key(doc_id: str) -> str:
    """Create Redis key for a document embedding."""
    return f"{REDIS_EMBEDDING_PREFIX}{doc_id}"


def get_embedding(doc_id: str) -> Optional[np.ndarray]:
    """
    Get cached embedding for a document.

    Args:
        doc_id: Document ID

    Returns:
        numpy array of embedding or None if not cached
    """
    try:
        client = get_redis_client()
        key = _make_key(doc_id)
        data = client.get(key)

        if data is None:
            return None

        # Deserialize numpy array
        embedding = np.frombuffer(data, dtype=np.float32)

        if len(embedding) != EMBEDDING_DIM:
            logger.warning(f"Cached embedding has wrong dimension: {len(embedding)} vs {EMBEDDING_DIM}")
            return None

        return embedding
    except Exception as e:
        logger.warning(f"Error getting cached embedding for {doc_id}: {e}")
        return None


def get_embeddings_batch(doc_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
    """
    Get cached embeddings for multiple documents.

    Args:
        doc_ids: List of document IDs

    Returns:
        Dict mapping doc_id to embedding (or None if not cached)
    """
    if not doc_ids:
        return {}

    try:
        client = get_redis_client()
        keys = [_make_key(doc_id) for doc_id in doc_ids]

        # Use pipeline for efficiency
        pipe = client.pipeline()
        for key in keys:
            pipe.get(key)
        results = pipe.execute()

        embeddings = {}
        cached_count = 0

        for doc_id, data in zip(doc_ids, results):
            if data is not None:
                embedding = np.frombuffer(data, dtype=np.float32)
                if len(embedding) == EMBEDDING_DIM:
                    embeddings[doc_id] = embedding
                    cached_count += 1
                else:
                    embeddings[doc_id] = None
            else:
                embeddings[doc_id] = None

        logger.info(f"Retrieved {cached_count}/{len(doc_ids)} embeddings from Redis cache")
        return embeddings
    except Exception as e:
        logger.warning(f"Error getting batch embeddings from Redis: {e}")
        return {doc_id: None for doc_id in doc_ids}


def save_embedding(doc_id: str, embedding: np.ndarray) -> bool:
    """
    Save embedding to Redis cache.

    Args:
        doc_id: Document ID
        embedding: numpy array of embedding

    Returns:
        True if saved successfully
    """
    try:
        client = get_redis_client()
        key = _make_key(doc_id)

        # Serialize numpy array to bytes
        data = embedding.astype(np.float32).tobytes()

        client.setex(key, REDIS_EMBEDDING_TTL, data)
        return True
    except Exception as e:
        logger.warning(f"Error saving embedding for {doc_id}: {e}")
        return False


def save_embeddings_batch(embeddings: Dict[str, np.ndarray]) -> int:
    """
    Save multiple embeddings to Redis cache.

    Args:
        embeddings: Dict mapping doc_id to embedding

    Returns:
        Number of embeddings saved successfully
    """
    if not embeddings:
        return 0

    try:
        client = get_redis_client()
        pipe = client.pipeline()

        for doc_id, embedding in embeddings.items():
            if embedding is not None:
                key = _make_key(doc_id)
                data = embedding.astype(np.float32).tobytes()
                pipe.setex(key, REDIS_EMBEDDING_TTL, data)

        pipe.execute()
        count = len(embeddings)
        logger.info(f"Saved {count} embeddings to Redis cache")
        return count
    except Exception as e:
        logger.error(f"Error saving batch embeddings to Redis: {e}")
        return 0


def save_embeddings_from_docs(docs: List[Dict]) -> int:
    """
    Save new embeddings from processed documents to Redis.

    Args:
        docs: List of document dicts with 'id' and 'new_embedding' fields

    Returns:
        Number of embeddings saved
    """
    embeddings = {}
    for doc in docs:
        if "new_embedding" in doc and doc["new_embedding"] is not None:
            embeddings[doc["id"]] = doc["new_embedding"]

    if not embeddings:
        logger.info("No new embeddings to save to Redis")
        return 0

    return save_embeddings_batch(embeddings)


def delete_embedding(doc_id: str) -> bool:
    """Delete a cached embedding."""
    try:
        client = get_redis_client()
        key = _make_key(doc_id)
        client.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Error deleting embedding for {doc_id}: {e}")
        return False


def get_cache_stats() -> Dict:
    """Get Redis cache statistics."""
    try:
        client = get_redis_client()
        info = client.info("memory")

        # Count embedding keys
        cursor = 0
        count = 0
        while True:
            cursor, keys = client.scan(cursor, match=f"{REDIS_EMBEDDING_PREFIX}*", count=1000)
            count += len(keys)
            if cursor == 0:
                break

        return {
            "embedding_count": count,
            "used_memory": info.get("used_memory_human", "unknown"),
            "used_memory_peak": info.get("used_memory_peak_human", "unknown"),
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"error": str(e)}


# ====== LAST FETCH TIMESTAMP TRACKING ======

LAST_FETCH_PREFIX = "last_fetch:"


def _make_fetch_key(user_input_id: Optional[str], source_ids: Optional[List[str]]) -> str:
    """Create Redis key for last fetch timestamp."""
    import hashlib
    filter_str = f"{user_input_id or 'all'}_{sorted(source_ids) if source_ids else 'all'}"
    filter_hash = hashlib.md5(filter_str.encode()).hexdigest()[:12]
    return f"{LAST_FETCH_PREFIX}{filter_hash}"


def get_last_fetch_timestamp(
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None
) -> Optional[int]:
    """
    Get the last fetch timestamp for a filter combination.

    Args:
        user_input_id: Optional user input ID filter
        source_ids: Optional source IDs filter

    Returns:
        Timestamp in milliseconds or None if never fetched
    """
    try:
        client = get_redis_client()
        key = _make_fetch_key(user_input_id, source_ids)
        data = client.get(key)

        if data is None:
            return None

        return int(data.decode('utf-8'))
    except Exception as e:
        logger.warning(f"Error getting last fetch timestamp: {e}")
        return None


def save_last_fetch_timestamp(
    timestamp_ms: int,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None
) -> bool:
    """
    Save the last fetch timestamp for a filter combination.

    Args:
        timestamp_ms: Timestamp in milliseconds
        user_input_id: Optional user input ID filter
        source_ids: Optional source IDs filter

    Returns:
        True if saved successfully
    """
    try:
        client = get_redis_client()
        key = _make_fetch_key(user_input_id, source_ids)
        # No TTL - keep forever
        client.set(key, str(timestamp_ms))
        logger.info(f"Saved last fetch timestamp: {timestamp_ms}")
        return True
    except Exception as e:
        logger.warning(f"Error saving last fetch timestamp: {e}")
        return False


def clear_last_fetch_timestamp(
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None
) -> bool:
    """Clear the last fetch timestamp (force full refetch next time)."""
    try:
        client = get_redis_client()
        key = _make_fetch_key(user_input_id, source_ids)
        client.delete(key)
        logger.info("Cleared last fetch timestamp")
        return True
    except Exception as e:
        logger.warning(f"Error clearing last fetch timestamp: {e}")
        return False
