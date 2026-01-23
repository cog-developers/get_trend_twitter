"""Configuration validation."""

from src.config.settings import (
    OPENSEARCH_NODE,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    DEEPSEEK_API_KEY,
    REDIS_URL
)
from src.infra.redis_client import validate_redis_connection
from src.logging.logger import get_logger

logger = get_logger(__name__)


def validate_config():
    """Validate required environment variables."""
    required_vars = {
        "OPENSEARCH_NODE": OPENSEARCH_NODE,
        "OPENSEARCH_USERNAME": OPENSEARCH_USERNAME,
        "OPENSEARCH_PASSWORD": OPENSEARCH_PASSWORD,
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
        "REDIS_URL": REDIS_URL
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info("✅ Configuration validated")
    
    # Validate Redis connection
    if REDIS_URL:
        if validate_redis_connection():
            logger.info("✅ Redis connection validated")
        else:
            logger.warning("⚠️ Redis connection validation failed, but continuing...")
