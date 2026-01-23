"""
Worker service module.
Background worker that polls MySQL for active user inputs and processes them.

Usage:
    python -m src.worker

Or with systemd:
    systemctl start trending-topics-worker
"""

import time
from datetime import datetime
from typing import Optional, List, Dict

from .config.settings import (
    validate_config,
    validate_db_config,
    MIN_CLUSTER_SIZE,
    MAX_TOPICS,
    ACTIVE_INPUT_POLL_SECONDS,
    get_logger
)
from .database.opensearch import (
    create_opensearch_client,
    ensure_embedding_mapping,
    save_embeddings_to_opensearch,
    save_trending_topics
)
from .database.mysql import (
    fetch_active_inputs,
    normalize_source_ids
)
from .services.posts import fetch_posts
from .processing.embeddings import EmbeddingProcessor
from .services.analysis import analyze_trending_topics
from .jobs.tracker import (
    generate_job_id,
    get_job_status,
    create_job,
    update_job_status
)

logger = get_logger(__name__)


def process_trending_topics_job(
    job_id: str,
    user_input_id: Optional[str],
    source_ids: Optional[List[str]],
    min_cluster_size: int = None,
    save_to_index: bool = True
) -> None:
    """
    Background function to process trending topics generation.
    Updates job status as it progresses.
    Uses cached embeddings when available to avoid recomputation.
    """
    min_cluster_size = min_cluster_size or MIN_CLUSTER_SIZE

    try:
        logger.info(f"Starting job processing: {job_id}")
        update_job_status(job_id, "processing", progress=10)

        # Validate configuration
        validate_config()

        # Create OpenSearch client
        client = create_opensearch_client()

        try:
            # Ensure embedding field exists in index mapping
            ensure_embedding_mapping(client)
            update_job_status(job_id, "processing", progress=20)

            # Fetch posts with filters (includes cached embeddings)
            posts = fetch_posts(
                client,
                user_input_id=user_input_id,
                source_ids=source_ids
            )

            if not posts:
                update_job_status(
                    job_id,
                    "failed",
                    error="No posts found matching the filters"
                )
                return

            logger.info(f"Found {len(posts)} posts matching filters")
            update_job_status(job_id, "processing", progress=30)

            # Use provided min_cluster_size or default
            effective_min_cluster_size = max(min_cluster_size, MIN_CLUSTER_SIZE)

            if len(posts) < effective_min_cluster_size:
                update_job_status(
                    job_id,
                    "failed",
                    error=f"Not enough posts ({len(posts)} < {effective_min_cluster_size})"
                )
                return

            # Create embeddings (uses cache, only generates for new posts)
            update_job_status(job_id, "processing", progress=40)
            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(posts)

            # Save new embeddings back to OpenSearch for future runs
            save_embeddings_to_opensearch(client, posts)

            # Reduce dimensionality
            update_job_status(job_id, "processing", progress=60)
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)

            # Cluster documents
            update_job_status(job_id, "processing", progress=75)
            labels = embedding_processor.cluster_documents(embeddings_reduced)

            # Analyze trending topics
            update_job_status(job_id, "processing", progress=85)
            trending_topics = analyze_trending_topics(
                posts,
                labels,
                embeddings_reduced=embeddings_reduced,
                embeddings_raw=embeddings,
                max_topics=MAX_TOPICS
            )

            # Add filter metadata to each topic and make cluster_id unique
            filtered_sources = source_ids if source_ids else []
            filter_key = job_id

            for topic in trending_topics:
                topic['user_input_id'] = user_input_id
                topic['filtered_sources'] = filtered_sources
                topic['filter_key'] = filter_key
                # Make cluster_id unique by adding filter hash
                original_cluster_id = topic.get('cluster_id', '')
                topic['cluster_id'] = f"{original_cluster_id}_{filter_key}"

            # Save to OpenSearch if requested
            if save_to_index:
                update_job_status(job_id, "processing", progress=90)
                save_trending_topics(client, trending_topics)

            # Prepare result
            result = {
                "trending_topics": trending_topics,
                "total_topics": len(trending_topics),
                "total_posts_processed": len(posts),
                "filters_applied": {
                    "user_input_id": user_input_id,
                    "source_ids": source_ids
                },
                "generated_at": datetime.utcnow().isoformat()
            }

            # Mark as completed
            update_job_status(job_id, "completed", result=result, progress=100)
            logger.info(f"Job completed successfully: {job_id}")

        finally:
            client.close()

    except ValueError as e:
        logger.error(f"Validation error in job {job_id}: {e}")
        update_job_status(job_id, "failed", error=f"Validation error: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, "failed", error=f"Processing error: {str(e)}")


def process_active_inputs() -> None:
    """Fetch and process active user inputs from MySQL."""
    try:
        logger.info("Fetching active user tracking inputs...")
        active_inputs = fetch_active_inputs()
        logger.info(f"Found {len(active_inputs)} active inputs to process")

        if not active_inputs:
            logger.info("No active inputs found. Waiting for next poll.")
            return

        for input_row in active_inputs:
            user_input_id = str(input_row.get("id"))
            source_ids = normalize_source_ids(input_row.get("accounts"))
            min_cluster_size = 5
            save_to_index = True

            logger.info(f"Processing input: user_input_id={user_input_id}, source_ids={source_ids}")

            job_id = generate_job_id(user_input_id, source_ids)
            existing_job = get_job_status(job_id)
            if existing_job and existing_job["status"] in ["pending", "processing"]:
                logger.info(f"Job {job_id} already in progress, skipping")
                continue

            create_job(job_id, user_input_id, source_ids)
            process_trending_topics_job(
                job_id,
                user_input_id,
                source_ids,
                min_cluster_size,
                save_to_index
            )

    except Exception as e:
        logger.error(f"Error in process_active_inputs: {e}", exc_info=True)
        raise


def main():
    """Main entry point for the worker service."""
    logger.info("Starting Trending Topics Worker Service")
    logger.info("This service polls MySQL for active user inputs")
    logger.info("This should be run with systemd, NOT PM2")

    # Validate configuration
    try:
        validate_config()
        validate_db_config()
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        exit(1)

    poll_seconds = ACTIVE_INPUT_POLL_SECONDS
    if poll_seconds <= 0:
        process_active_inputs()
        exit(0)

    while True:
        process_active_inputs()
        logger.info(f"Sleeping {poll_seconds}s before next poll")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
