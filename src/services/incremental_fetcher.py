"""Incremental data fetching - only fetch new posts since last run."""

import numpy as np
from typing import List, Dict, Optional
from opensearchpy import OpenSearch, helpers
from src.config.settings import SOURCE_INDEX, EMBEDDING_DIM
from src.common.text_utils import clean_text, parse_datetime_to_epoch_millis
from src.logging.logger import get_logger

logger = get_logger(__name__)


def fetch_new_posts(
    client: OpenSearch, 
    user_input_id: str = None, 
    source_ids: List[str] = None,
    since_timestamp_ms: Optional[int] = None
) -> List[Dict]:
    """
    Fetch only NEW posts since the last topic generation.
    
    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by
        since_timestamp_ms: Only fetch posts after this timestamp (epoch milliseconds)
    
    Returns:
        List of new posts with cached embeddings if available
    """
    if since_timestamp_ms:
        logger.info(f"📥 Fetching NEW posts since timestamp: {since_timestamp_ms}")
    else:
        logger.info(f"📥 Fetching ALL posts (no timestamp filter)")

    docs = []
    cached_count = 0
    
    # Build query with filters
    must_clauses = []
    if user_input_id:
        must_clauses.append({"term": {"user_input_id": user_input_id}})
    if source_ids:
        must_clauses.append({"terms": {"source_id": source_ids}})
    
    # Add timestamp filter for incremental fetching
    if since_timestamp_ms:
        # Try multiple timestamp fields
        timestamp_filter = {
            "bool": {
                "should": [
                    {"range": {"timestamp": {"gt": since_timestamp_ms}}},
                    {"range": {"created_at": {"gt": since_timestamp_ms}}},
                    {"range": {"post_created_at": {"gt": since_timestamp_ms}}}
                ],
                "minimum_should_match": 1
            }
        }
        must_clauses.append(timestamp_filter)
    
    query = {
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}]
            }
        },
        "_source": [
            "post_text", "text", "content", "created_at", "timestamp",
            "author", "likes", "retweets", "replies", "user_input_id", "source_id",
            "embedding", "embedding_updated_at"  # Include cached embeddings
        ]
    }

    try:
        for hit in helpers.scan(
            client,
            query=query,
            index=SOURCE_INDEX,
            size=500,
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

                if not text or len(text.strip()) < 5:
                    continue

                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 5:
                    # Get cached embedding if available
                    cached_embedding = src.get("embedding")
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
                logger.warning(f"⚠️ Error processing document {hit.get('_id')}: {e}")
                continue

        logger.info(f"✅ Loaded {len(docs)} NEW posts ({cached_count} with cached embeddings, {len(docs) - cached_count} need embedding)")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error fetching new posts: {e}")
        raise
