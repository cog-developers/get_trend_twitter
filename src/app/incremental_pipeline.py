"""Optimized incremental trending topics pipeline."""

import time
import json
from datetime import datetime
from typing import Optional, List
from opensearchpy import OpenSearch

from src.config.validation import validate_config
from src.config.settings import SOURCE_INDEX, MIN_CLUSTER_SIZE, MAX_TOPICS
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
from src.logging.logger import get_logger

logger = get_logger(__name__)


def print_trending_topics(topics: List[dict]):
    """Print trending topics summary."""
    print("\n" + "=" * 80)
    print("🔥 TRENDING TOPICS")
    print("=" * 80)
    
    for i, topic_data in enumerate(topics[:MAX_TOPICS], 1):
        print(f"\n{i}. {topic_data['topic']}")
        print(f"   📊 Posts: {topic_data['post_count']} | "
              f"Engagement: {topic_data['engagement_score']:.1f} | "
              f"Score: {topic_data['trending_score']:.2f}")
        print(f"   🏷️  Keywords: {', '.join(topic_data['keywords'][:3])}")
    
    print("\n" + "=" * 80)


def process_incremental(
    user_input_id: str,
    source_ids: Optional[List[str]] = None
):
    """
    Optimized incremental processing:
    1. Load existing topics
    2. Fetch only NEW posts since last run
    3. Process new posts
    4. Merge with existing topics intelligently
    5. Save updated topics
    """
    start_time = time.time()
    
    try:
        validate_config()
        client = create_opensearch_client()
        
        try:
            # Step 1: Load existing topics
            logger.info("📥 Step 1: Loading existing topics...")
            existing_topics = get_existing_topics(client, user_input_id, source_ids)
            
            # Step 2: Determine last processed timestamp
            last_timestamp_ms = get_last_processed_timestamp(existing_topics)
            existing_member_ids = get_existing_member_ids(existing_topics)
            
            if last_timestamp_ms:
                logger.info(f"📅 Last processed timestamp: {last_timestamp_ms} ({datetime.fromtimestamp(last_timestamp_ms/1000)})")
            else:
                logger.info("📅 No previous run found, processing all posts")
            
            # Step 3: Fetch only NEW posts
            logger.info("📥 Step 2: Fetching new posts...")
            ensure_embedding_mapping(client)
            
            new_posts = fetch_new_posts(
                client,
                user_input_id=user_input_id,
                source_ids=source_ids,
                since_timestamp_ms=last_timestamp_ms
            )
            
            # Filter out posts already in existing topics
            if existing_member_ids:
                new_posts = [p for p in new_posts if p['id'] not in existing_member_ids]
                logger.info(f"🔍 Filtered to {len(new_posts)} truly new posts (excluding {len(existing_member_ids)} existing)")
            
            if not new_posts:
                logger.info("✅ No new posts to process, existing topics are up to date")
                return existing_topics[:MAX_TOPICS]
            
            if len(new_posts) < MIN_CLUSTER_SIZE:
                logger.warning(f"⚠️ Not enough new posts ({len(new_posts)} < {MIN_CLUSTER_SIZE}), keeping existing topics")
                return existing_topics[:MAX_TOPICS]
            
            # Step 4: Process new posts
            logger.info("🔢 Step 3: Processing new posts...")
            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(new_posts)
            
            # Save new embeddings
            save_embeddings_to_opensearch(client, new_posts)
            
            # Reduce dimensionality
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)
            
            # Cluster new posts
            labels = cluster_documents(embeddings_reduced)
            
            # Analyze new topics
            new_topics = analyze_trending_topics(
                new_posts,
                labels,
                embeddings_reduced=embeddings_reduced,
                embeddings_raw=embeddings,
                max_topics=MAX_TOPICS
            )
            
            # Add metadata to new topics
            filtered_sources = source_ids if source_ids else []
            for topic in new_topics:
                topic['user_input_id'] = user_input_id
                topic['filtered_sources'] = filtered_sources
            
            logger.info(f"✅ Generated {len(new_topics)} new topics from {len(new_posts)} new posts")
            
            # Step 5: Merge with existing topics
            logger.info("🔄 Step 4: Merging with existing topics...")
            merged_topics = merge_topics_incremental(
                existing_topics,
                new_topics,
                max_topics=MAX_TOPICS
            )
            
            logger.info(f"✅ Final result: {len(merged_topics)} topics (target: {MAX_TOPICS})")
            
            # Step 6: Save updated topics
            logger.info("💾 Step 5: Saving updated topics...")
            save_trending_topics(client, merged_topics)
            
            elapsed = time.time() - start_time
            logger.info(f"\n🎉 Incremental processing complete! Time: {elapsed:.2f}s")
            logger.info(f"📊 Processed {len(new_posts)} new posts, {len(existing_topics)} existing topics → {len(merged_topics)} final topics")
            
            return merged_topics
            
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        raise


def main():
    """Main execution function - supports incremental processing."""
    start_time = time.time()

    try:
        validate_config()
        client = create_opensearch_client()

        try:
            ensure_embedding_mapping(client)
            
            # For now, process all posts (can be enhanced to accept user_input_id as arg)
            # TODO: Add command-line args or config for user_input_id
            logger.info("⚠️ Running in full mode (no user_input_id specified)")
            logger.info("💡 Use process_incremental(user_input_id, source_ids) for incremental processing")
            
            # Fallback to full processing
            from src.services.data_fetcher import fetch_posts
            from src.services.embedding_service import EmbeddingProcessor
            from src.services.clustering_service import cluster_documents
            from src.services.trending_analysis_service import analyze_trending_topics
            
            docs = fetch_posts(client)
            
            if len(docs) < MIN_CLUSTER_SIZE:
                logger.error(f"❌ Not enough posts ({len(docs)} < {MIN_CLUSTER_SIZE})")
                return

            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(docs)
            save_embeddings_to_opensearch(client, docs)
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)
            labels = cluster_documents(embeddings_reduced)
            
            trending_topics = analyze_trending_topics(
                docs,
                labels,
                embeddings_reduced=embeddings_reduced,
                embeddings_raw=embeddings,
                max_topics=MAX_TOPICS
            )

            if not trending_topics:
                logger.warning("⚠️ No trending topics found")
                return

            save_trending_topics(client, trending_topics)
            print_trending_topics(trending_topics)

            output_file = f"trending_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(trending_topics, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved results to {output_file}")

            elapsed = time.time() - start_time
            logger.info(f"\n🎉 Complete! Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
            logger.info(f"📊 Processed {len(docs)} posts, found {len(trending_topics)} trending topics")

        finally:
            client.close()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
