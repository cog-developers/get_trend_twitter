"""Main trending topics pipeline."""

import time
import json
from datetime import datetime
from typing import Optional, List
from opensearchpy import OpenSearch

from src.config.validation import validate_config
from src.config.settings import SOURCE_INDEX, MIN_CLUSTER_SIZE, MAX_TOPICS
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
from src.logging.logger import get_logger

logger = get_logger(__name__)


def print_trending_topics(topics: List[dict]):
    """Print trending topics summary."""
    print("\n" + "=" * 80)
    print("🔥 TRENDING TOPICS")
    print("=" * 80)
    
    for i, topic_data in enumerate(topics[:MAX_TOPICS], 1):  # Top N
        print(f"\n{i}. {topic_data['topic']}")
        print(f"   📊 Posts: {topic_data['post_count']} | "
              f"Engagement: {topic_data['engagement_score']:.1f} | "
              f"Score: {topic_data['trending_score']:.2f}")
        print(f"   🏷️  Keywords: {', '.join(topic_data['keywords'][:3])}")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    start_time = time.time()

    try:
        validate_config()

        # Create OpenSearch client
        client = create_opensearch_client()

        # Ensure embedding field exists in index mapping
        ensure_embedding_mapping(client)

        # Fetch posts (including cached embeddings)
        docs = fetch_posts(client)

        if len(docs) < MIN_CLUSTER_SIZE:
            logger.error(f"❌ Not enough posts ({len(docs)} < {MIN_CLUSTER_SIZE})")
            return

        # Create embeddings (uses cache, only generates for new posts)
        embedding_processor = EmbeddingProcessor()
        embeddings = embedding_processor.create_embeddings(docs)

        # Save new embeddings back to OpenSearch for future runs
        save_embeddings_to_opensearch(client, docs)

        # Reduce dimensionality
        embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)

        # Cluster documents
        labels = cluster_documents(embeddings_reduced)

        # Analyze trending topics
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

        # Save to OpenSearch
        save_trending_topics(client, trending_topics)

        # Print results
        print_trending_topics(trending_topics)

        # Save JSON file
        output_file = f"trending_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trending_topics, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved results to {output_file}")

        elapsed = time.time() - start_time
        logger.info(f"\n🎉 Complete! Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        logger.info(f"📊 Processed {len(docs)} posts, found {len(trending_topics)} trending topics")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        raise
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    main()
