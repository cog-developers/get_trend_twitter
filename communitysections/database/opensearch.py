"""
OpenSearch database operations for Community Sections.
Handles connection, index management, and bulk operations.
"""

import time
import random
from typing import List, Dict

from opensearchpy import OpenSearch, helpers, exceptions
from opensearchpy.connection import Urllib3HttpConnection

from ..config.settings import (
    OPENSEARCH_NODE,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    SOURCE_INDEX,
    TARGET_INDEX,
    RETRY_ATTEMPTS,
    ES_RETRY_DELAY,
    get_logger
)

logger = get_logger(__name__)

# Global client instance
_client = None


def create_opensearch_client() -> OpenSearch:
    """Create and return OpenSearch client."""
    global _client
    if _client is not None:
        return _client

    try:
        _client = OpenSearch(
            [OPENSEARCH_NODE],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
            connection_class=Urllib3HttpConnection,
            pool_maxsize=10,
        )
        if not _client.ping():
            raise ConnectionError("Cannot connect to OpenSearch (ping failed)")
        logger.info("Connected to OpenSearch")
        return _client
    except exceptions.AuthenticationException:
        logger.error("Authentication failed for OpenSearch. Check your username and password.")
        raise
    except exceptions.ConnectionError:
        logger.error("Failed to connect to OpenSearch. Check the OPENSEARCH_NODE URL.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while initializing OpenSearch: {e}")
        raise


def get_client() -> OpenSearch:
    """Get the OpenSearch client (creates if not exists)."""
    global _client
    if _client is None:
        return create_opensearch_client()
    return _client


def es_call_with_retries(fn, *args, **kwargs):
    """Call an OpenSearch function with exponential backoff retries for network errors."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > RETRY_ATTEMPTS:
                if isinstance(e, exceptions.RequestError) and e.status_code == 400:
                    raise
                logger.error(f"OpenSearch operation failed after {attempt} attempts: {e}")
                raise

            if isinstance(e, exceptions.RequestError) and e.status_code == 400:
                raise

            logger.warning(f"OpenSearch error (attempt {attempt}/{RETRY_ATTEMPTS}): {e}. Retrying...")
            sleep_time = ES_RETRY_DELAY * (2 ** (attempt - 1)) + random.random() * 0.5
            time.sleep(sleep_time)


def create_classified_index(force_recreate: bool = False):
    """Create or recreate the classified users index."""
    client = get_client()

    mapping = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "index": {"refresh_interval": "5s"},
        },
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "username": {"type": "keyword"},
                "name": {"type": "text"},
                "party": {"type": "keyword"},
                "confidence": {"type": "keyword"},
                "reasoning": {"type": "text"},
                "tweet_count": {"type": "integer"},
                "user_bio": {"type": "text"},
                "sample_tweets": {"type": "text"},
                "classified_at": {"type": "date"},
                "classification_run_id": {"type": "keyword"},
                "api_response_time_ms": {"type": "integer"},
                "created_at": {"type": "date"},
                "post_created_at": {"type": "date"},
                "timestamp": {"type": "date"},
            }
        },
    }

    try:
        if client.indices.exists(index=TARGET_INDEX):
            if force_recreate:
                logger.info(f"Deleting existing index '{TARGET_INDEX}'...")
                client.indices.delete(index=TARGET_INDEX)
            else:
                logger.info(f"Index '{TARGET_INDEX}' already exists, will append results")
                return
        client.indices.create(index=TARGET_INDEX, body=mapping)
        logger.info(f"Created index '{TARGET_INDEX}'")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def save_classifications_bulk(classified_users: List[Dict]):
    """Save classifications to OpenSearch in bulk."""
    if not classified_users:
        logger.warning("No classifications to save")
        return

    client = get_client()
    logger.info(f"Saving {len(classified_users)} classifications to '{TARGET_INDEX}'...")

    actions = [
        {"_index": TARGET_INDEX, "_id": user["user_id"], "_source": user}
        for user in classified_users
    ]

    try:
        success, failed = helpers.bulk(
            client,
            actions,
            chunk_size=100,
            request_timeout=120,
            raise_on_error=False,
        )
        logger.info(f"Bulk save reported success={success}")
        if failed:
            logger.warning("Some bulk items failed (see server logs)")
    except Exception as e:
        logger.error(f"Bulk save error: {e}")


def validate_index_mapping(index: str, fields: List[str]) -> None:
    """Validate that required fields exist in the index mapping."""
    client = get_client()

    try:
        mapping = es_call_with_retries(client.indices.get_mapping, index=index)
        properties = mapping.get(index, {}).get("mappings", {}).get("properties", {})

        for field in fields:
            if field not in properties:
                logger.warning(f"Field '{field}' is missing in the index mapping for '{index}'.")
            elif properties[field].get("type") != "date":
                logger.warning(f"Field '{field}' in index '{index}' is not mapped as a 'date' field.")
            else:
                logger.info(f"Field '{field}' is correctly mapped as 'date' in index '{index}'.")
    except Exception as e:
        logger.error(f"Failed to validate index mapping for '{index}': {e}")
