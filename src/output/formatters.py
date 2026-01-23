"""
Output formatting module.
Handles console output and JSON export.
"""

import json
from datetime import datetime
from typing import List, Dict

from ..config.settings import MAX_TOPICS, get_logger

logger = get_logger(__name__)


def print_trending_topics(topics: List[Dict], max_display: int = None):
    """Print trending topics summary to console."""
    max_display = max_display or MAX_TOPICS

    print("\n" + "=" * 80)
    print("TRENDING TOPICS")
    print("=" * 80)

    for i, topic_data in enumerate(topics[:max_display], 1):
        print(f"\n{i}. {topic_data['topic']}")
        print(f"   Posts: {topic_data['post_count']} | "
              f"Engagement: {topic_data['engagement_score']:.1f} | "
              f"Score: {topic_data['trending_score']:.2f}")
        print(f"   Keywords: {', '.join(topic_data['keywords'][:3])}")

    print("\n" + "=" * 80)


def save_topics_to_json(topics: List[Dict], output_file: str = None) -> str:
    """Save trending topics to a JSON file."""
    if output_file is None:
        output_file = f"trending_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved results to {output_file}")
    return output_file


def format_topics_summary(topics: List[Dict]) -> str:
    """Format topics as a summary string."""
    lines = []
    for i, topic_data in enumerate(topics, 1):
        lines.append(
            f"{i}. {topic_data['topic']} "
            f"(posts: {topic_data['post_count']}, score: {topic_data['trending_score']:.2f})"
        )
    return "\n".join(lines)
