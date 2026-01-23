"""
Embedding processor module.
Handles text embeddings, dimensionality reduction, and clustering.
"""

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import hdbscan

from ..config.settings import (
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    PCA_TARGET_DIM,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    get_logger
)

logger = get_logger(__name__)


class EmbeddingProcessor:
    """Handle text embeddings and clustering with caching support."""

    def __init__(self):
        self.model = None

    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info("Loading sentence transformer model...")
            self.model = SentenceTransformer(
                'sentence-transformers/distiluse-base-multilingual-cased-v2'
            )
            logger.info("Model loaded")
        return self.model

    def create_embeddings(self, docs: List[Dict]) -> np.ndarray:
        """
        Create embeddings for documents, using cached embeddings when available.
        Only computes new embeddings for posts without cached vectors.
        Returns the full embedding matrix and marks docs with new embeddings.
        """
        total_docs = len(docs)

        # Separate docs with and without cached embeddings
        docs_needing_embedding = []
        docs_needing_indices = []

        for i, doc in enumerate(docs):
            if doc.get("cached_embedding") is None:
                docs_needing_embedding.append(doc)
                docs_needing_indices.append(i)

        cached_count = total_docs - len(docs_needing_embedding)
        logger.info(f"Processing {total_docs} posts: {cached_count} cached, {len(docs_needing_embedding)} need embedding")

        # Initialize embeddings array
        embeddings = np.zeros((total_docs, EMBEDDING_DIM), dtype=np.float32)

        # Fill in cached embeddings
        for i, doc in enumerate(docs):
            if doc.get("cached_embedding") is not None:
                embeddings[i] = doc["cached_embedding"]

        # Generate new embeddings only for docs that need them
        if docs_needing_embedding:
            logger.info(f"Generating embeddings for {len(docs_needing_embedding)} new posts...")
            model = self.load_model()
            texts = [d["cleaned"] for d in docs_needing_embedding]

            try:
                new_embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    normalize_embeddings=True
                )

                # Fill in new embeddings and mark docs for saving
                for j, idx in enumerate(docs_needing_indices):
                    embeddings[idx] = new_embeddings[j]
                    # Mark this doc as having a new embedding to save
                    docs[idx]["new_embedding"] = new_embeddings[j]

                logger.info(f"Generated {len(docs_needing_embedding)} new embeddings")

            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                raise
        else:
            logger.info("All embeddings loaded from cache - no model inference needed!")

        logger.info(f"Total embeddings ready: shape={embeddings.shape}")
        return embeddings

    def reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        logger.info("Reducing dimensionality with PCA...")

        try:
            n_comp = min(PCA_TARGET_DIM, embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(embeddings)

            variance_explained = np.sum(pca.explained_variance_ratio_)
            logger.info(
                f"Reduced to {reduced.shape[1]} dims "
                f"(variance explained: {variance_explained:.2%})"
            )
            return reduced
        except Exception as e:
            logger.error(f"Error in dimensionality reduction: {e}")
            raise

    def cluster_documents(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster documents using HDBSCAN."""
        logger.info("Clustering documents...")

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

            logger.info(f"Found {n_clusters} clusters, {noise} noise points")
            return labels
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise
