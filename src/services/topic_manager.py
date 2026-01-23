"""Incremental topic management - load existing topics and merge with new ones."""

from typing import List, Dict, Optional
from opensearchpy import OpenSearch
from src.config.settings import TRENDING_INDEX, MAX_TOPICS
from src.common.text_utils import parse_datetime_to_epoch_millis
from src.logging.logger import get_logger

logger = get_logger(__name__)


def get_existing_topics(
    client: OpenSearch,
    user_input_id: str,
    source_ids: Optional[List[str]] = None
) -> List[Dict]:
    """
    Load existing trending topics for a user_input_id from OpenSearch.
    
    Args:
        client: OpenSearch client
        user_input_id: User input ID to filter by
        source_ids: Optional source IDs filter
    
    Returns:
        List of existing topics sorted by trending_score (descending)
    """
    logger.info(f"📥 Loading existing topics for user_input_id={user_input_id}")
    
    must_clauses = [{"term": {"user_input_id": user_input_id}}]
    
    if source_ids:
        must_clauses.append({"terms": {"filtered_sources": source_ids}})
    
    query = {
        "size": MAX_TOPICS * 2,  # Get more than MAX_TOPICS to be safe
        "sort": [{"trending_score": {"order": "desc"}}],
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }
    
    try:
        response = client.search(index=TRENDING_INDEX, body=query)
        hits = response.get('hits', {}).get('hits', [])
        
        existing_topics = []
        for hit in hits:
            topic = hit['_source']
            # Ensure member_ids is a list
            if isinstance(topic.get('member_ids'), str):
                topic['member_ids'] = [topic['member_ids']]
            existing_topics.append(topic)
        
        logger.info(f"✅ Found {len(existing_topics)} existing topics")
        return existing_topics
        
    except Exception as e:
        logger.warning(f"⚠️ Error loading existing topics: {e}")
        return []


def get_last_processed_timestamp(
    existing_topics: List[Dict]
) -> Optional[int]:
    """
    Get the maximum last_post_timestamp from existing topics.
    This tells us the timestamp of the newest post that was already processed.
    
    Args:
        existing_topics: List of existing topics
    
    Returns:
        Maximum timestamp in milliseconds, or None if no topics
    """
    if not existing_topics:
        return None
    
    timestamps = []
    for topic in existing_topics:
        ts = topic.get('last_post_timestamp')
        if ts:
            # Handle both epoch millis and ISO strings
            ts_ms = parse_datetime_to_epoch_millis(ts)
            if ts_ms:
                timestamps.append(ts_ms)
    
    return max(timestamps) if timestamps else None


def get_existing_member_ids(existing_topics: List[Dict]) -> set:
    """
    Get all member IDs from existing topics to avoid duplicates.
    
    Args:
        existing_topics: List of existing topics
    
    Returns:
        Set of all member IDs already in topics
    """
    member_ids = set()
    for topic in existing_topics:
        ids = topic.get('member_ids', [])
        if isinstance(ids, list):
            member_ids.update(ids)
        elif isinstance(ids, str):
            member_ids.add(ids)
    return member_ids
