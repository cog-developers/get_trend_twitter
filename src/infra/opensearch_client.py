"""OpenSearch client factory."""

from opensearchpy import OpenSearch
from src.config.settings import (
    OPENSEARCH_NODE,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD
)
from src.logging.logger import get_logger

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
        logger.info("✅ Connected to OpenSearch")
        return client
    except Exception as e:
        logger.error(f"❌ OpenSearch connection failed: {e}")
        raise


# Alias for backward compatibility
create_client = create_opensearch_client
