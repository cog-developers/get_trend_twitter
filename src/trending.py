"""
Main trending topics orchestration module.
Coordinates the full pipeline: fetch -> embed -> cluster -> analyze -> save.
"""

import time
from typing import List, Dict, Optional

from .config.settings import (
    validate_config,
    MIN_CLUSTER_SIZE,
    MAX_TOPICS,
    USE_REDIS_CACHE,
    get_logger
)
from .database.opensearch import (
    create_opensearch_client,
    save_trending_topics
)
from .database.redis_cache import (
    create_redis_client,
    save_embeddings_from_docs
)
from .services.posts import fetch_posts
from .processing.embeddings import EmbeddingProcessor
from .services.analysis import analyze_trending_topics
from .output.formatters import print_trending_topics, save_topics_to_json

logger = get_logger(__name__)


def run_trending_topics_pipeline(
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    save_to_index: bool = True,
    save_json: bool = True,
    print_results: bool = True,
    max_topics: int = None
) -> List[Dict]:
    """
    Run the complete trending topics pipeline.

    Args:
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by
        save_to_index: Whether to save results to OpenSearch
        save_json: Whether to save results to a JSON file
        print_results: Whether to print results to console
        max_topics: Maximum number of topics to return

    Returns:
        List of trending topic dictionaries
    """
    start_time = time.time()
    max_topics = max_topics or MAX_TOPICS

    try:
        validate_config()

        # Create OpenSearch client
        client = create_opensearch_client()

        try:
            # Initialize Redis for embedding cache
            if USE_REDIS_CACHE:
                create_redis_client()

            # Fetch posts (cached embeddings loaded from Redis)
            docs = fetch_posts(
                client,
                user_input_id=user_input_id,
                source_ids=source_ids
            )

            if len(docs) < MIN_CLUSTER_SIZE:
                logger.error(f"Not enough posts ({len(docs)} < {MIN_CLUSTER_SIZE})")
                return []

            # Create embeddings (uses cache, only generates for new posts)
            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(docs)

            # Save new embeddings to Redis cache for future runs
            if USE_REDIS_CACHE:
                save_embeddings_from_docs(docs)

            # Reduce dimensionality
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)

            # Cluster documents
            labels = embedding_processor.cluster_documents(embeddings_reduced)

            # Analyze trending topics
            trending_topics = analyze_trending_topics(
                docs,
                labels,
                embeddings_reduced=embeddings_reduced,
                embeddings_raw=embeddings,
                max_topics=max_topics
            )

            if not trending_topics:
                logger.warning("No trending topics found")
                return []

            # Add filter metadata if filters were applied
            if user_input_id or source_ids:
                for topic in trending_topics:
                    topic['user_input_id'] = user_input_id
                    topic['filtered_sources'] = source_ids or []

            # Save to OpenSearch
            if save_to_index:
                save_trending_topics(client, trending_topics)

            # Print results
            if print_results:
                print_trending_topics(trending_topics)

            # Save JSON file
            if save_json:
                save_topics_to_json(trending_topics)

            elapsed = time.time() - start_time
            logger.info(f"Complete! Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
            logger.info(f"Processed {len(docs)} posts, found {len(trending_topics)} trending topics")

            return trending_topics

        finally:
            client.close()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


def main():
    """Main entry point for standalone execution."""
    run_trending_topics_pipeline()


if __name__ == "__main__":
    main()
