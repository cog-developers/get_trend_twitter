"""
Configuration and settings module.
All environment variables, constants, and validation functions.
"""

import os
import logging
import warnings
import urllib3
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== OPENSEARCH CONFIG ======
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "user-input-posts")
TRENDING_INDEX = os.getenv("TRENDING_INDEX", "trending-topics")

# ====== DEEPSEEK API CONFIG ======
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# ====== PROCESSING CONFIG ======
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "5"))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))
PCA_TARGET_DIM = int(os.getenv("PCA_TARGET_DIM", "100"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MAX_TOPICS = int(os.getenv("MAX_TOPICS", "20"))
EMBEDDING_DIM = 512  # distiluse-base-multilingual-cased-v2 output dimension

# ====== DATABASE CONFIG ======
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_RETRY_DELAY_SECONDS = float(os.getenv("DB_RETRY_DELAY_SECONDS", "5"))

# ====== WORKER CONFIG ======
ACTIVE_INPUT_POLL_SECONDS = int(os.getenv("ACTIVE_INPUT_POLL_SECONDS", "300"))

# ====== OPENSEARCH SCAN CONFIG ======
OPENSEARCH_SCAN_SIZE = int(os.getenv("OPENSEARCH_SCAN_SIZE", "200"))
OPENSEARCH_SCAN_RETRIES = int(os.getenv("OPENSEARCH_SCAN_RETRIES", "5"))
OPENSEARCH_RETRY_DELAY_SECONDS = float(os.getenv("OPENSEARCH_RETRY_DELAY_SECONDS", "5"))
INCLUDE_CACHED_EMBEDDINGS = os.getenv("INCLUDE_CACHED_EMBEDDINGS", "0").lower() in ("1", "true", "yes")

# ====== ARABIC STOPWORDS ======
ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "إلى", "مع", "عن", "ما", "لا",
    "هذا", "هذه", "هو", "هي", "أن", "إن", "كان", "كل", "قد",
    "لم", "لن", "ثم", "أو", "بل", "لكن", "حتى", "عند", "بعد", "قبل"
}


def validate_config():
    """Validate required environment variables for OpenSearch and DeepSeek."""
    required_vars = {
        "OPENSEARCH_NODE": OPENSEARCH_NODE,
        "OPENSEARCH_USERNAME": OPENSEARCH_USERNAME,
        "OPENSEARCH_PASSWORD": OPENSEARCH_PASSWORD,
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    logger.info("Configuration validated")


def validate_db_config():
    """Validate database configuration from environment variables."""
    required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
