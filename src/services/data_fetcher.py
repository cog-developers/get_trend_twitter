"""Data fetching service for OpenSearch."""

import numpy as np
from typing import List, Dict
from opensearchpy import OpenSearch, helpers
from src.config.settings import SOURCE_INDEX, EMBEDDING_DIM
from src.common.text_utils import clean_text
from src.logging.logger import get_logger

logger = get_logger(__name__)


def fetch_posts(client: OpenSearch, user_input_id: str = None, source_ids: List[str] = None) -> List[Dict]:
    """
    Fetch posts from user-input-posts index, including cached embeddings.
    
    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by (None = all sources)
    
    Returns:
        List of posts with cached embeddings if available
    """
    logger.info(f"📥 Fetching posts from index: {SOURCE_INDEX}")

    docs = []
    cached_count = 0
    
    # Build query with filters
    must_clauses = []
    if user_input_id:
        must_clauses.append({"term": {"user_input_id": user_input_id}})
    if source_ids:
        must_clauses.append({"terms": {"source_id": source_ids}})
    
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

                if not text or len(text.strip()) < 10:
                    continue

                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 10:
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

        logger.info(f"✅ Loaded {len(docs)} valid posts ({cached_count} with cached embeddings, {len(docs) - cached_count} need embedding)")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error fetching posts: {e}")
        raise
