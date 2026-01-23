"""Incremental topic merging - intelligently merge new topics with existing ones."""

import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.config.settings import MAX_TOPICS
from src.common.text_utils import parse_datetime_to_epoch_millis
from src.logging.logger import get_logger

logger = get_logger(__name__)


def merge_topics_incremental(
    existing_topics: List[Dict],
    new_topics: List[Dict],
    max_topics: int = MAX_TOPICS
) -> List[Dict]:
    """
    Intelligently merge new topics with existing ones.
    
    Strategy:
    1. If existing < max_topics: Add new topics (up to max_topics)
    2. If existing = max_topics: Merge new with existing (update member_ids)
    3. Always keep exactly max_topics (or fewer if not enough)
    
    Args:
        existing_topics: Existing topics from previous run
        new_topics: Newly generated topics
        max_topics: Maximum number of topics to keep
    
    Returns:
        Merged list of topics (up to max_topics)
    """
    logger.info(f"🔄 Merging topics: {len(existing_topics)} existing + {len(new_topics)} new → max {max_topics}")
    
    if not existing_topics:
        logger.info("📝 No existing topics, returning new topics")
        return new_topics[:max_topics] if new_topics else []
    
    if not new_topics:
        logger.info("📝 No new topics, returning existing topics")
        return existing_topics[:max_topics]
    
    # Convert existing topics to dict for easier lookup
    existing_dict = {topic.get('cluster_id'): topic for topic in existing_topics}
    existing_count = len(existing_topics)
    
    # Case 1: Existing topics < max_topics → Add new topics
    if existing_count < max_topics:
        logger.info(f"➕ Adding new topics (existing: {existing_count} < max: {max_topics})")
        
        # Merge by similarity: if new topic is similar to existing, update existing
        # Otherwise, add as new topic
        merged = list(existing_topics)
        remaining_slots = max_topics - existing_count
        
        # Try to merge similar topics first
        for new_topic in new_topics:
            if len(merged) >= max_topics:
                break
            
            # Check if new topic is similar to any existing topic
            similar_existing = find_similar_topic(new_topic, merged)
            
            if similar_existing:
                # Merge: update existing topic with new posts
                logger.info(f"🔗 Merging new topic '{new_topic.get('topic', '')[:50]}...' with existing '{similar_existing.get('topic', '')[:50]}...'")
                merged_topic = merge_topic_posts(similar_existing, new_topic)
                # Replace in merged list
                merged = [t if t.get('cluster_id') != similar_existing.get('cluster_id') else merged_topic for t in merged]
            else:
                # Add as new topic
                merged.append(new_topic)
        
        # Sort by trending score
        merged.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        return merged[:max_topics]
    
    # Case 2: Existing topics = max_topics → Merge/Update existing
    elif existing_count >= max_topics:
        logger.info(f"🔄 Updating existing topics (existing: {existing_count} >= max: {max_topics})")
        
        merged = list(existing_topics)
        
        # For each new topic, try to merge with most similar existing topic
        for new_topic in new_topics:
            similar_existing = find_similar_topic(new_topic, merged)
            
            if similar_existing:
                # Merge: update existing topic with new posts
                logger.info(f"🔗 Updating existing topic '{similar_existing.get('topic', '')[:50]}...' with new posts")
                merged_topic = merge_topic_posts(similar_existing, new_topic)
                # Replace in merged list
                merged = [t if t.get('cluster_id') != similar_existing.get('cluster_id') else merged_topic for t in merged]
            else:
                # New topic doesn't match any existing - replace lowest scoring existing topic
                if new_topic.get('trending_score', 0) > min([t.get('trending_score', 0) for t in merged]):
                    logger.info(f"🆕 Replacing lowest scoring topic with new topic '{new_topic.get('topic', '')[:50]}...'")
                    # Find and replace lowest scoring topic
                    merged.sort(key=lambda x: x.get('trending_score', 0))
                    merged[0] = new_topic
        
        # Sort by trending score
        merged.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        return merged[:max_topics]
    
    return existing_topics[:max_topics]


def find_similar_topic(new_topic: Dict, existing_topics: List[Dict], threshold: float = 0.90) -> Optional[Dict]:
    """
    Find the most similar existing topic to a new topic using centroid embeddings.
    
    Args:
        new_topic: New topic to match
        existing_topics: List of existing topics
        threshold: Cosine similarity threshold (default: 0.90 - stricter to ensure content correlation)
    
    Returns:
        Most similar existing topic or None if below threshold
    """
    new_centroid = new_topic.get('centroid_embedding')
    if not new_centroid or not isinstance(new_centroid, list):
        return None
    
    new_centroid = np.array(new_centroid).reshape(1, -1)
    best_match = None
    best_similarity = threshold
    
    for existing in existing_topics:
        existing_centroid = existing.get('centroid_embedding')
        if not existing_centroid or not isinstance(existing_centroid, list):
            continue
        
        existing_centroid = np.array(existing_centroid).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(new_centroid, existing_centroid)[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = existing
    
    if best_match:
        logger.debug(f"🔍 Found similar topic: similarity={best_similarity:.3f} (threshold={threshold})")
    
    return best_match


def merge_topic_posts(existing_topic: Dict, new_topic: Dict) -> Dict:
    """
    Merge two topics by combining their posts and recalculating metrics.
    
    Args:
        existing_topic: Existing topic
        new_topic: New topic to merge
    
    Returns:
        Merged topic with updated metrics
    """
    # Combine member IDs (avoid duplicates)
    existing_ids = set(existing_topic.get('member_ids', []))
    new_ids = set(new_topic.get('member_ids', []))
    merged_ids = list(existing_ids | new_ids)
    
    # Update post count
    merged_post_count = len(merged_ids)
    
    # Recalculate trending score based only on post count (content similarity)
    merged_trending_score = merged_post_count
    
    # Update last_post_timestamp (use the newer one)
    existing_ts = existing_topic.get('last_post_timestamp')
    new_ts = new_topic.get('last_post_timestamp')
    
    # Handle timestamp comparison (can be epoch millis or ISO string)
    existing_ts_ms = parse_datetime_to_epoch_millis(existing_ts) if existing_ts else None
    new_ts_ms = parse_datetime_to_epoch_millis(new_ts) if new_ts else None
    
    if existing_ts_ms and new_ts_ms:
        merged_ts = max(existing_ts_ms, new_ts_ms)
    elif new_ts_ms:
        merged_ts = new_ts_ms
    elif existing_ts_ms:
        merged_ts = existing_ts_ms
    else:
        merged_ts = None
    
    # Combine keywords (keep unique, most common)
    existing_keywords = existing_topic.get('keywords', [])
    new_keywords = new_topic.get('keywords', [])
    # Simple merge: combine and deduplicate
    merged_keywords = list(dict.fromkeys(existing_keywords + new_keywords))[:5]
    
    # Keep existing topic label (or use new if existing is generic)
    existing_label = existing_topic.get('topic', '')
    new_label = new_topic.get('topic', '')
    merged_label = existing_label if existing_label and existing_label != "غير قابل للتحديد" else new_label
    
    # Create merged topic
    merged_topic = existing_topic.copy()
    merged_topic.update({
        'member_ids': merged_ids,
        'post_count': merged_post_count,
        'engagement_score': 0,  # Not used - topics based on content only
        'trending_score': merged_trending_score,
        'last_post_timestamp': merged_ts,
        'keywords': merged_keywords,
        'topic': merged_label,
        # Update timestamp to now
        'generated_at': new_topic.get('generated_at'),
        'timestamp': new_topic.get('timestamp')
    })
    
    logger.info(f"✅ Merged topic: {merged_post_count} posts (was {existing_topic.get('post_count', 0)}), score: {merged_trending_score:.2f}")
    
    return merged_topic
