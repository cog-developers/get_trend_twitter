"""Clustering service using HDBSCAN."""

import numpy as np
import hdbscan
from src.config.settings import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES
from src.logging.logger import get_logger

logger = get_logger(__name__)


def cluster_documents(embeddings: np.ndarray) -> np.ndarray:
    """Cluster documents using HDBSCAN."""
    logger.info("🎯 Clustering documents...")
    
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set([l for l in labels if l != -1]))
        noise = np.sum(labels == -1)
        
        logger.info(f"✅ Found {n_clusters} clusters, {noise} noise points")
        return labels
    except Exception as e:
        logger.error(f"❌ Clustering failed: {e}")
        raise
