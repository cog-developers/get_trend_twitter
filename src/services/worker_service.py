"""Worker service for processing trending topics jobs with incremental optimization."""

from datetime import datetime
from typing import Optional, List
from opensearchpy import OpenSearch

from src.config.validation import validate_config
from src.config.settings import MIN_CLUSTER_SIZE, MAX_TOPICS
from src.infra.opensearch_client import create_opensearch_client
from src.services.incremental_fetcher import fetch_new_posts
from src.services.topic_manager import (
    get_existing_topics,
    get_last_processed_timestamp,
    get_existing_member_ids
)
from src.services.incremental_merger import merge_topics_incremental
from src.services.embedding_service import EmbeddingProcessor
from src.services.clustering_service import cluster_documents
from src.services.trending_analysis_service import analyze_trending_topics
from src.services.index_service import (
    ensure_embedding_mapping,
    save_embeddings_to_opensearch,
    save_trending_topics
)
from src.services.job_tracker import update_job_status
from src.logging.logger import get_logger

logger = get_logger(__name__)


def process_trending_topics_job(
    job_id: str,
    user_input_id: Optional[str],
    source_ids: Optional[List[str]],
    min_cluster_size: int,
    save_to_index: bool
):
    """
    Optimized incremental processing:
    1. Load existing topics
    2. Fetch only NEW posts since last run
    3. Process new posts
    4. Merge with existing topics intelligently
    5. Save updated topics
    
    Uses cached embeddings when available to avoid recomputation.
    """
    try:
        logger.info(f"🔄 Starting incremental job processing: {job_id}")
        update_job_status(job_id, "processing", progress=10)

        # Validate configuration
        validate_config()

        # Create OpenSearch client
        client = create_opensearch_client()

        try:
            # Ensure embedding field exists in index mapping
            ensure_embedding_mapping(client)

            update_job_status(job_id, "processing", progress=15)

            # Step 1: Load existing topics
            logger.info("📥 Loading existing topics...")
            existing_topics = []
            if user_input_id:
                existing_topics = get_existing_topics(client, user_input_id, source_ids)
            
            # Step 2: Determine last processed timestamp
            last_timestamp_ms = get_last_processed_timestamp(existing_topics)
            existing_member_ids = get_existing_member_ids(existing_topics)
            
            if last_timestamp_ms:
                logger.info(f"📅 Last processed: {datetime.fromtimestamp(last_timestamp_ms/1000)}")
            else:
                logger.info("📅 No previous run, processing all posts")

            update_job_status(job_id, "processing", progress=20)

            # Step 3: Fetch only NEW posts
            logger.info("📥 Fetching new posts...")
            new_posts = fetch_new_posts(
                client,
                user_input_id=user_input_id,
                source_ids=source_ids,
                since_timestamp_ms=last_timestamp_ms
            )
            
            # Filter out posts already in existing topics
            if existing_member_ids:
                initial_count = len(new_posts)
                new_posts = [p for p in new_posts if p['id'] not in existing_member_ids]
                logger.info(f"🔍 Filtered: {initial_count} → {len(new_posts)} truly new posts")

            if not new_posts:
                logger.info("✅ No new posts, existing topics are up to date")
                update_job_status(job_id, "completed", result={
                    "trending_topics": existing_topics[:MAX_TOPICS],
                    "total_topics": len(existing_topics),
                    "total_posts_processed": 0,
                    "incremental": True,
                    "message": "No new posts to process"
                }, progress=100)
                return

            logger.info(f"✅ Found {len(new_posts)} new posts to process")
            update_job_status(job_id, "processing", progress=30)

            # Use provided min_cluster_size or default
            effective_min_cluster_size = max(min_cluster_size, MIN_CLUSTER_SIZE)

            if len(new_posts) < effective_min_cluster_size:
                logger.warning(f"⚠️ Not enough new posts ({len(new_posts)} < {effective_min_cluster_size})")
                update_job_status(job_id, "completed", result={
                    "trending_topics": existing_topics[:MAX_TOPICS],
                    "total_topics": len(existing_topics),
                    "total_posts_processed": len(new_posts),
                    "incremental": True,
                    "message": "Not enough new posts for clustering"
                }, progress=100)
                return

            # Step 4: Process new posts
            update_job_status(job_id, "processing", progress=40)
            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(new_posts)

            # Save new embeddings back to OpenSearch for future runs
            save_embeddings_to_opensearch(client, new_posts)

            # Reduce dimensionality
            update_job_status(job_id, "processing", progress=60)
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)
            
            # Cluster documents
            update_job_status(job_id, "processing", progress=75)
            labels = cluster_documents(embeddings_reduced)
            
            # Analyze trending topics
            update_job_status(job_id, "processing", progress=85)
            new_topics = analyze_trending_topics(
                new_posts,
                labels,
                embeddings_reduced=embeddings_reduced,
                embeddings_raw=embeddings,
                max_topics=MAX_TOPICS
            )
            
            # Add filter metadata to each new topic
            filtered_sources = source_ids if source_ids else []
            filter_key = job_id
            
            for topic in new_topics:
                topic['user_input_id'] = user_input_id
                topic['filtered_sources'] = filtered_sources
                topic['filter_key'] = filter_key
                # Make cluster_id unique by adding filter hash
                original_cluster_id = topic.get('cluster_id', '')
                topic['cluster_id'] = f"{original_cluster_id}_{filter_key}"
            
            logger.info(f"✅ Generated {len(new_topics)} new topics from {len(new_posts)} new posts")
            
            # Step 5: Merge with existing topics
            update_job_status(job_id, "processing", progress=90)
            logger.info("🔄 Merging with existing topics...")
            merged_topics = merge_topics_incremental(
                existing_topics,
                new_topics,
                max_topics=MAX_TOPICS
            )
            
            logger.info(f"✅ Final result: {len(merged_topics)} topics (target: {MAX_TOPICS})")
            
            # Step 6: Save updated topics
            if save_to_index:
                update_job_status(job_id, "processing", progress=95)
                save_trending_topics(client, merged_topics)
            
            # Prepare result
            result = {
                "trending_topics": merged_topics,
                "total_topics": len(merged_topics),
                "total_posts_processed": len(new_posts),
                "existing_topics_count": len(existing_topics),
                "new_topics_count": len(new_topics),
                "incremental": True,
                "filters_applied": {
                    "user_input_id": user_input_id,
                    "source_ids": source_ids
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            # Mark as completed
            update_job_status(job_id, "completed", result=result, progress=100)
            logger.info(f"✅ Incremental job completed successfully: {job_id}")
            
        finally:
            client.close()
            
    except ValueError as e:
        logger.error(f"❌ Validation error in job {job_id}: {e}")
        update_job_status(job_id, "failed", error=f"Validation error: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ Error processing job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, "failed", error=f"Processing error: {str(e)}")
