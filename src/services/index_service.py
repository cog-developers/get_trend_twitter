"""OpenSearch index management and data persistence."""

import time
from datetime import datetime, timezone
from typing import List, Dict
from opensearchpy import OpenSearch, helpers
from src.config.settings import SOURCE_INDEX, TRENDING_INDEX, EMBEDDING_DIM
from src.logging.logger import get_logger

logger = get_logger(__name__)


def ensure_embedding_mapping(client: OpenSearch):
    """Ensure the embedding field exists in the source index mapping."""
    try:
        mapping = client.indices.get_mapping(index=SOURCE_INDEX)
        properties = mapping.get(SOURCE_INDEX, {}).get("mappings", {}).get("properties", {})

        if "embedding" not in properties:
            logger.info(f"📝 Adding 'embedding' field to index {SOURCE_INDEX}...")
            update_mapping = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": False  # We don't need KNN search, just storage
                    },
                    "embedding_updated_at": {
                        "type": "date"
                    }
                }
            }
            client.indices.put_mapping(index=SOURCE_INDEX, body=update_mapping)
            logger.info("✅ Added embedding field to index mapping")
        else:
            logger.info("ℹ️ Embedding field already exists in index mapping")
    except Exception as e:
        logger.warning(f"⚠️ Could not update mapping for embeddings: {e}")


def save_embeddings_to_opensearch(client: OpenSearch, docs_with_embeddings: List[Dict]):
    """Save computed embeddings back to OpenSearch for caching with progress tracking."""
    if not docs_with_embeddings:
        return

    # Count how many actually need saving
    actions = []
    for doc in docs_with_embeddings:
        if "new_embedding" in doc and doc["new_embedding"] is not None:
            actions.append({
                "_op_type": "update",
                "_index": SOURCE_INDEX,
                "_id": doc["id"],
                "doc": {
                    "embedding": doc["new_embedding"].tolist() if hasattr(doc["new_embedding"], 'tolist') else doc["new_embedding"],
                    "embedding_updated_at": datetime.now(timezone.utc).isoformat()
                }
            })

    if not actions:
        logger.info("ℹ️ No new embeddings to save")
        return

    total_to_save = len(actions)
    logger.info(f"💾 Saving {total_to_save} embeddings to OpenSearch...")
    
    # Optimized: Use larger chunk size for better throughput
    # Process in visible batches for progress tracking
    chunk_size = 500  # Batch size for OpenSearch bulk API
    progress_batch_size = 5000  # Log progress every N posts
    start_time = time.time()
    total_saved = 0
    total_errors = 0
    
    try:
        # Process in progress-tracked batches
        for i in range(0, total_to_save, progress_batch_size):
            batch = actions[i:i + progress_batch_size]
            batch_start = time.time()
            
            success, errors = helpers.bulk(
                client,
                batch,
                chunk_size=chunk_size,
                request_timeout=120,
                raise_on_error=False
            )
            
            batch_elapsed = time.time() - batch_start
            total_saved += success
            total_errors += len(errors) if errors else 0
            
            # Log progress
            progress_pct = (total_saved / total_to_save) * 100
            elapsed_total = time.time() - start_time
            avg_time_per_post = elapsed_total / total_saved if total_saved > 0 else 0
            remaining = total_to_save - total_saved
            estimated_remaining = avg_time_per_post * remaining if avg_time_per_post > 0 else 0
            
            logger.info(
                f"⏳ Progress: {total_saved}/{total_to_save} ({progress_pct:.1f}%) | "
                f"Batch: {batch_elapsed:.2f}s | "
                f"Avg: {avg_time_per_post*1000:.1f}ms/post | "
                f"ETA: {estimated_remaining:.0f}s"
            )
        
        elapsed = time.time() - start_time
        time_per_post = elapsed / total_to_save if total_to_save > 0 else 0
        
        logger.info(f"✅ Saved {total_saved}/{total_to_save} embeddings in {elapsed:.2f}s")
        logger.info(f"📊 Performance: {time_per_post*1000:.2f}ms per post | {total_to_save/elapsed:.1f} posts/second")
        
        if total_errors > 0:
            logger.warning(f"⚠️ {total_errors} errors while saving embeddings")
    except Exception as e:
        logger.error(f"❌ Error saving embeddings: {e}")
        raise


def save_trending_topics(client: OpenSearch, topics: List[Dict]):
    """Save trending topics to OpenSearch."""
    logger.info(f"💾 Saving {len(topics)} trending topics to {TRENDING_INDEX}...")
    
    # Create index if it doesn't exist, or update mapping if it exists
    if not client.indices.exists(index=TRENDING_INDEX):
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "topic": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "cluster_id": {"type": "keyword"},
                    "post_count": {"type": "integer"},
                    "engagement_score": {"type": "float"},
                    "trending_score": {"type": "float"},
                    "keywords": {"type": "keyword"},
                    "representative_texts": {"type": "text"},
                    "member_ids": {"type": "keyword"},
                    "centroid_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": False
                    },
                    "generated_at": {"type": "date"},
                    "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "last_post_timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "rank": {"type": "integer"},
                    "user_input_id": {"type": "keyword"},
                    "filtered_sources": {"type": "keyword"},
                    "filter_key": {"type": "keyword"}
                }
            }
        }
        client.indices.create(index=TRENDING_INDEX, body=mapping)
        logger.info(f"✅ Created index: {TRENDING_INDEX}")
    else:
        # Index exists - check if new fields exist and add them if missing
        try:
            current_mapping = client.indices.get_mapping(index=TRENDING_INDEX)
            properties = current_mapping.get(TRENDING_INDEX, {}).get("mappings", {}).get("properties", {})
            
            missing_fields = {}
            
            if "timestamp" not in properties:
                missing_fields["timestamp"] = {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            
            if "user_input_id" not in properties:
                missing_fields["user_input_id"] = {"type": "keyword"}
            
            if "filtered_sources" not in properties:
                missing_fields["filtered_sources"] = {"type": "keyword"}

            if "filter_key" not in properties:
                missing_fields["filter_key"] = {"type": "keyword"}

            if "centroid_embedding" not in properties:
                missing_fields["centroid_embedding"] = {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": False
                }

            if "last_post_timestamp" not in properties:
                missing_fields["last_post_timestamp"] = {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            
            if missing_fields:
                logger.info(f"📝 Adding missing fields to existing index: {TRENDING_INDEX}")
                update_mapping = {"properties": missing_fields}
                client.indices.put_mapping(index=TRENDING_INDEX, body=update_mapping)
                logger.info(f"✅ Updated index mapping with fields: {list(missing_fields.keys())}")
            else:
                logger.info(f"ℹ️  Index '{TRENDING_INDEX}' already has all required fields")
        except Exception as e:
            logger.warning(f"⚠️  Could not update mapping: {e}")
    
    # Prepare actions - use update to merge with existing topics
    actions = []
    for rank, topic_data in enumerate(topics, 1):
        topic_data["rank"] = rank
        actions.append({
            "_op_type": "index",  # index will update if _id exists, create if not
            "_index": TRENDING_INDEX,
            "_id": topic_data["cluster_id"],
            "_source": topic_data
        })
    
    # Bulk insert
    try:
        success, errors = helpers.bulk(
            client,
            actions,
            raise_on_error=False
        )
        logger.info(f"✅ Saved {success} trending topics")
        if errors:
            logger.warning(f"⚠️ {len(errors)} errors during bulk insert")
    except Exception as e:
        logger.error(f"❌ Error saving topics: {e}")
        raise
