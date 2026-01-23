"""
Post fetching service.
Handles fetching posts from OpenSearch with optional filters.
Uses Redis for embedding cache and incremental fetching.
"""

import time
from typing import List, Dict, Optional, Tuple

import numpy as np
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import TransportError

from ..config.settings import (
    SOURCE_INDEX,
    EMBEDDING_DIM,
    OPENSEARCH_SCAN_SIZE,
    OPENSEARCH_SCAN_RETRIES,
    OPENSEARCH_RETRY_DELAY_SECONDS,
    USE_REDIS_CACHE,
    get_logger
)
from ..processing.text import clean_text

logger = get_logger(__name__)


def _load_cached_embeddings_from_redis(docs: List[Dict]) -> int:
    """
    Load cached embeddings from Redis and attach to docs.

    Args:
        docs: List of document dicts with 'id' field

    Returns:
        Number of cached embeddings found
    """
    if not USE_REDIS_CACHE or not docs:
        return 0

    try:
        from ..database.redis_cache import get_embeddings_batch

        doc_ids = [doc["id"] for doc in docs]
        cached = get_embeddings_batch(doc_ids)

        cached_count = 0
        for doc in docs:
            embedding = cached.get(doc["id"])
            if embedding is not None:
                doc["cached_embedding"] = embedding
                cached_count += 1

        return cached_count
    except Exception as e:
        logger.warning(f"Failed to load embeddings from Redis: {e}")
        return 0


def _get_max_timestamp(docs: List[Dict]) -> Optional[int]:
    """Get the maximum timestamp from docs (in milliseconds)."""
    max_ts = None
    for doc in docs:
        ts = doc.get("timestamp")
        if ts is not None:
            # Handle different timestamp formats
            if isinstance(ts, (int, float)):
                ts_ms = int(ts) if ts > 10_000_000_000 else int(ts * 1000)
            elif isinstance(ts, str):
                try:
                    from datetime import datetime, timezone
                    if ts.endswith("Z"):
                        ts = ts[:-1] + "+00:00"
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts_ms = int(dt.timestamp() * 1000)
                except Exception:
                    continue
            else:
                continue

            if max_ts is None or ts_ms > max_ts:
                max_ts = ts_ms

    return max_ts


def fetch_posts(
    client: OpenSearch,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    index: str = None,
    incremental: bool = True
) -> List[Dict]:
    """
    Fetch posts from OpenSearch with optional filters.
    Supports incremental fetching (only new posts since last fetch).
    Loads cached embeddings from Redis.

    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by (None = all sources)
        index: Optional index name (defaults to SOURCE_INDEX)
        incremental: If True, only fetch posts newer than last fetch (default: True)

    Returns:
        List of posts with cached embeddings if available
    """
    index = index or SOURCE_INDEX

    filter_info = []
    if user_input_id:
        filter_info.append(f"user_input_id={user_input_id}")
    if source_ids:
        filter_info.append(f"source_ids={source_ids}")

    # Get last fetch timestamp for incremental fetching
    last_fetch_ts = None
    if incremental and USE_REDIS_CACHE:
        try:
            from ..database.redis_cache import get_last_fetch_timestamp
            last_fetch_ts = get_last_fetch_timestamp(user_input_id, source_ids)
            if last_fetch_ts:
                filter_info.append(f"timestamp>{last_fetch_ts}")
                logger.info(f"Incremental fetch: only posts after timestamp {last_fetch_ts}")
        except Exception as e:
            logger.warning(f"Could not get last fetch timestamp: {e}")

    logger.info(f"Fetching posts from index: {index}" + (f" with filters: {', '.join(filter_info)}" if filter_info else ""))

    # Build query with filters
    must_clauses = []

    if user_input_id:
        must_clauses.append({"term": {"user_input_id": user_input_id}})

    if source_ids:
        must_clauses.append({"terms": {"source_id": source_ids}})

    # Add timestamp filter for incremental fetch
    if last_fetch_ts:
        must_clauses.append({
            "range": {
                "timestamp": {
                    "gt": last_fetch_ts
                }
            }
        })

    scan_size = OPENSEARCH_SCAN_SIZE
    max_retries = OPENSEARCH_SCAN_RETRIES
    retry_delay = OPENSEARCH_RETRY_DELAY_SECONDS

    # Only fetch necessary fields from OpenSearch (no embeddings)
    base_source_fields = [
        "post_text", "text", "content", "created_at", "timestamp",
        "author", "likes", "retweets", "replies", "user_input_id", "source_id",
        "post_created_at"
    ]

    attempt = 0
    while True:
        docs = []

        query = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "_source": base_source_fields
        }

        try:
            for hit in helpers.scan(
                client,
                query=query,
                index=index,
                size=scan_size,
                scroll="10m",
                raise_on_error=False
            ):
                try:
                    src = hit.get("_source", {})

                    # Try multiple field names for post text
                    text = (
                        src.get("post_text") or
                        src.get("text") or
                        src.get("content") or
                        ""
                    )

                    if not text or len(text.strip()) < 10:
                        continue

                    cleaned = clean_text(text)
                    if cleaned and len(cleaned) > 10:
                        docs.append({
                            "id": hit["_id"],
                            "text": text.strip(),
                            "cleaned": cleaned,
                            "author": src.get("author"),
                            "created_at": src.get("created_at") or src.get("timestamp"),
                            "timestamp": src.get("timestamp"),
                            "post_created_at": src.get("post_created_at"),
                            "likes": src.get("likes", 0) or 0,
                            "retweets": src.get("retweets", 0) or 0,
                            "replies": src.get("replies", 0) or 0,
                            "user_input_id": src.get("user_input_id"),
                            "source_id": src.get("source_id"),
                            "cached_embedding": None,  # Will be loaded from Redis
                        })
                except Exception as e:
                    logger.warning(f"Error processing document {hit.get('_id')}: {e}")
                    continue

            # Load cached embeddings from Redis
            cached_count = _load_cached_embeddings_from_redis(docs)

            # Save the max timestamp for next incremental fetch
            if docs and USE_REDIS_CACHE:
                try:
                    from ..database.redis_cache import save_last_fetch_timestamp
                    max_ts = _get_max_timestamp(docs)
                    if max_ts:
                        save_last_fetch_timestamp(max_ts, user_input_id, source_ids)
                except Exception as e:
                    logger.warning(f"Could not save last fetch timestamp: {e}")

            fetch_type = "incremental" if last_fetch_ts else "full"
            logger.info(f"Loaded {len(docs)} valid posts ({fetch_type} fetch, {cached_count} with cached embeddings from Redis, {len(docs) - cached_count} need embedding)")
            return docs

        except TransportError as e:
            if e.status_code == 429 and "circuit_breaking_exception" in str(e):
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"Error fetching posts after {max_retries} retries: {e}")
                    raise
                logger.warning(
                    "OpenSearch circuit breaker hit; retrying in %ss (attempt %s/%s) with size=%s",
                    retry_delay, attempt, max_retries, scan_size
                )
                time.sleep(retry_delay)
                scan_size = max(50, scan_size // 2)
                continue
            logger.error(f"Error fetching posts: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
            raise


def fetch_all_posts(
    client: OpenSearch,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    index: str = None
) -> List[Dict]:
    """
    Fetch ALL posts (ignores incremental, does full fetch).
    Use this to force a complete refetch.

    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by
        index: Optional index name

    Returns:
        List of all posts
    """
    return fetch_posts(
        client,
        user_input_id=user_input_id,
        source_ids=source_ids,
        index=index,
        incremental=False
    )
