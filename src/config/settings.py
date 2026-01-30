import os
from dotenv import load_dotenv

load_dotenv()

# OpenSearch
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "user-input-posts")
TRENDING_INDEX = os.getenv("TRENDING_INDEX", "trending-topics")

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Processing
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "3"))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "3"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "2"))
PCA_TARGET_DIM = int(os.getenv("PCA_TARGET_DIM", "100"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MAX_TOPICS = int(os.getenv("MAX_TOPICS", "20"))
EMBEDDING_DIM = 512 
