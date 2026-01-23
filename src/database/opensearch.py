"""
OpenSearch database operations.
Client creation, index management, and data persistence.
"""

from datetime import datetime, timezone
from typing import List, Dict

from opensearchpy import OpenSearch, helpers

from ..config.settings import (
    OPENSEARCH_NODE,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    SOURCE_INDEX,
    TRENDING_INDEX,
    EMBEDDING_DIM,
    get_logger
)

logger = get_logger(__name__)


def create_opensearch_client() -> OpenSearch:
    """Create OpenSearch client."""
    try:
        client = OpenSearch(
            [OPENSEARCH_NODE],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=90,
            max_retries=3,
            retry_on_timeout=True
        )
        if not client.ping():
            raise ConnectionError("Cannot ping OpenSearch")
        logger.info("Connected to OpenSearch")
        return client
    except Exception as e:
        logger.error(f"OpenSearch connection failed: {e}")
        raise


def ensure_embedding_mapping(client: OpenSearch, index: str = None):
    """Ensure the embedding field exists in the source index mapping."""
    index = index or SOURCE_INDEX
    try:
        mapping = client.indices.get_mapping(index=index)
        properties = mapping.get(index, {}).get("mappings", {}).get("properties", {})

        if "embedding" not in properties:
            logger.info(f"Adding 'embedding' field to index {index}...")
            update_mapping = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": False
                    },
                    "embedding_updated_at": {
                        "type": "date"
                    }
                }
            }
            client.indices.put_mapping(index=index, body=update_mapping)
            logger.info("Added embedding field to index mapping")
        else:
            logger.info("Embedding field already exists in index mapping")
    except Exception as e:
        logger.warning(f"Could not update mapping for embeddings: {e}")


def save_embeddings_to_opensearch(client: OpenSearch, docs_with_embeddings: List[Dict], index: str = None):
    """Save computed embeddings back to OpenSearch for caching."""
    if not docs_with_embeddings:
        return

    index = index or SOURCE_INDEX
    logger.info(f"Saving {len(docs_with_embeddings)} embeddings to OpenSearch...")

    actions = []
    for doc in docs_with_embeddings:
        if "new_embedding" in doc and doc["new_embedding"] is not None:
            actions.append({
                "_op_type": "update",
                "_index": index,
                "_id": doc["id"],
                "doc": {
                    "embedding": doc["new_embedding"].tolist() if hasattr(doc["new_embedding"], 'tolist') else doc["new_embedding"],
                    "embedding_updated_at": datetime.now(timezone.utc).isoformat()
                }
            })

    if not actions:
        logger.info("No new embeddings to save")
        return

    try:
        success, errors = helpers.bulk(
            client,
            actions,
            chunk_size=100,
            request_timeout=120,
            raise_on_error=False
        )
        logger.info(f"Saved {success} embeddings to OpenSearch")
        if errors:
            logger.warning(f"{len(errors)} errors while saving embeddings")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")


def save_trending_topics(client: OpenSearch, topics: List[Dict], index: str = None):
    """Save trending topics to OpenSearch."""
    index = index or TRENDING_INDEX
    logger.info(f"Saving {len(topics)} trending topics to {index}...")

    # Create index if it doesn't exist, or update mapping if it exists
    if not client.indices.exists(index=index):
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
        client.indices.create(index=index, body=mapping)
        logger.info(f"Created index: {index}")
    else:
        # Index exists - check if new fields exist and add them if missing
        try:
            current_mapping = client.indices.get_mapping(index=index)
            properties = current_mapping.get(index, {}).get("mappings", {}).get("properties", {})

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
                logger.info(f"Adding missing fields to existing index: {index}")
                update_mapping = {"properties": missing_fields}
                client.indices.put_mapping(index=index, body=update_mapping)
                logger.info(f"Updated index mapping with fields: {list(missing_fields.keys())}")
            else:
                logger.info(f"Index '{index}' already has all required fields")
        except Exception as e:
            logger.warning(f"Could not update mapping: {e}")

    # Prepare actions
    actions = []
    for rank, topic_data in enumerate(topics, 1):
        topic_data["rank"] = rank
        actions.append({
            "_op_type": "index",
            "_index": index,
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
        logger.info(f"Saved {success} trending topics")
        if errors:
            logger.warning(f"{len(errors)} errors during bulk insert")
    except Exception as e:
        logger.error(f"Error saving topics: {e}")
        raise
