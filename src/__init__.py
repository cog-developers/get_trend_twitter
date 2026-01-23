"""
Public API for trending topics functionality.
This module provides a clean interface that matches the original get_trending_topics.py exports.
"""

# Re-export all the functions and classes that worker.py and other modules need
from src.config.validation import validate_config
from src.config.settings import (
    SOURCE_INDEX,
    EMBEDDING_DIM,
    MAX_TOPICS,
    MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    PCA_TARGET_DIM,
    EMBEDDING_BATCH_SIZE
)
from src.infra.opensearch_client import create_opensearch_client
from src.services.data_fetcher import fetch_posts
from src.services.embedding_service import EmbeddingProcessor
from src.services.clustering_service import cluster_documents
from src.services.trending_analysis_service import analyze_trending_topics
from src.services.index_service import (
    ensure_embedding_mapping,
    save_embeddings_to_opensearch,
    save_trending_topics
)
from src.common.text_utils import clean_text

__all__ = [
    'validate_config',
    'create_opensearch_client',
    'fetch_posts',
    'EmbeddingProcessor',
    'cluster_documents',
    'analyze_trending_topics',
    'save_trending_topics',
    'clean_text',
    'ensure_embedding_mapping',
    'save_embeddings_to_opensearch',
    'SOURCE_INDEX',
    'EMBEDDING_DIM',
    'MAX_TOPICS',
    'MIN_CLUSTER_SIZE',
    'HDBSCAN_MIN_CLUSTER_SIZE',
    'HDBSCAN_MIN_SAMPLES',
    'PCA_TARGET_DIM',
    'EMBEDDING_BATCH_SIZE',
]
