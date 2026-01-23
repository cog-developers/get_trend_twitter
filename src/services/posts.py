"""
Post fetching service.
Handles fetching posts from OpenSearch with optional filters.
"""

import time
from typing import List, Dict, Optional

import numpy as np
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import TransportError

from ..config.settings import (
    SOURCE_INDEX,
    EMBEDDING_DIM,
    OPENSEARCH_SCAN_SIZE,
    OPENSEARCH_SCAN_RETRIES,
    OPENSEARCH_RETRY_DELAY_SECONDS,
    INCLUDE_CACHED_EMBEDDINGS,
    get_logger
)
from ..processing.text import clean_text

logger = get_logger(__name__)


def fetch_posts(
    client: OpenSearch,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    index: str = None
) -> List[Dict]:
    """
    Fetch posts from OpenSearch with optional filters.
    Includes cached embeddings when available.

    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by (None = all sources)
        index: Optional index name (defaults to SOURCE_INDEX)

    Returns:
        List of posts with cached embeddings if available
    """
    index = index or SOURCE_INDEX

    filter_info = []
    if user_input_id:
        filter_info.append(f"user_input_id={user_input_id}")
    if source_ids:
        filter_info.append(f"source_ids={source_ids}")

    logger.info(f"Fetching posts from index: {index}" + (f" with filters: {', '.join(filter_info)}" if filter_info else ""))

    # Build query with filters
    must_clauses = []

    if user_input_id:
        must_clauses.append({"term": {"user_input_id": user_input_id}})

    if source_ids:
        must_clauses.append({"terms": {"source_id": source_ids}})

    include_cached = INCLUDE_CACHED_EMBEDDINGS
    scan_size = OPENSEARCH_SCAN_SIZE
    max_retries = OPENSEARCH_SCAN_RETRIES
    retry_delay = OPENSEARCH_RETRY_DELAY_SECONDS

    base_source_fields = [
        "post_text", "text", "content", "created_at", "timestamp",
        "author", "likes", "retweets", "replies", "user_input_id", "source_id",
        "post_created_at"
    ]

    attempt = 0
    while True:
        docs = []
        cached_count = 0

        query = {
            "query": {
                "bool": {
                    "must": must_clauses if must_clauses else [{"match_all": {}}]
                }
            },
            "_source": base_source_fields + (["embedding", "embedding_updated_at"] if include_cached else [])
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
                        # Get cached embedding if available
                        cached_embedding = src.get("embedding") if include_cached else None
                        has_cached = cached_embedding is not None and len(cached_embedding) == EMBEDDING_DIM

                        if has_cached:
                            cached_count += 1

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
                            "cached_embedding": np.array(cached_embedding) if has_cached else None,
                        })
                except Exception as e:
                    logger.warning(f"Error processing document {hit.get('_id')}: {e}")
                    continue

            logger.info(f"Loaded {len(docs)} valid posts ({cached_count} with cached embeddings, {len(docs) - cached_count} need embedding)")
            return docs

        except TransportError as e:
            if e.status_code == 429 and "circuit_breaking_exception" in str(e):
                attempt += 1
                if attempt > max_retries:
                    logger.error(f"Error fetching posts after {max_retries} retries: {e}")
                    raise
                logger.warning(
                    "OpenSearch circuit breaker hit; retrying in %ss (attempt %s/%s) with size=%s, include_cached=%s",
                    retry_delay, attempt, max_retries, scan_size, include_cached
                )
                time.sleep(retry_delay)
                scan_size = max(50, scan_size // 2)
                include_cached = False
                continue
            logger.error(f"Error fetching posts: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
            raise
