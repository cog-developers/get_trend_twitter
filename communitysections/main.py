"""
Community Sections - Main Entry Point.
Classifies Twitter users into Lebanese political parties.

Usage:
    python -m communitysections.main
"""

import time
from datetime import datetime

from .config.settings import (
    validate_config,
    MAX_USERS,
    SOURCE_INDEX,
    TARGET_INDEX,
    get_logger
)
from .database.opensearch import (
    create_opensearch_client,
    create_classified_index,
    save_classifications_bulk,
    validate_index_mapping
)
from .services.users import fetch_all_users
from .services.classifier import classify_users_parallel
from .output.formatters import (
    print_classification_summary,
    print_classification_results
)

logger = get_logger(__name__)


def run_classification_pipeline(
    max_users: int = 0,
    force_recreate_index: bool = False,
    print_results: bool = True
) -> dict:
    """
    Run the full classification pipeline.

    Args:
        max_users: Maximum users to process (0 = all)
        force_recreate_index: Delete and recreate target index
        print_results: Print detailed results to console

    Returns:
        dict with classification results and statistics
    """
    start_time = time.time()
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Classification Run ID: {run_id}")

    # Validate configuration
    validate_config()

    # Initialize OpenSearch
    create_opensearch_client()

    # Validate source index mapping
    validate_index_mapping(SOURCE_INDEX, ["created_at", "post_created_at", "timestamp"])

    # Create target index
    create_classified_index(force_recreate=force_recreate_index)

    # Fetch users
    users = fetch_all_users(limit=max_users)
    if not users:
        logger.error("No users found to classify")
        return {"success": False, "error": "No users found"}

    # Filter out invalid entries
    total_before = len(users)
    users = [u for u in users if isinstance(u, dict) and u.get("user_id")]
    filtered_out = total_before - len(users)
    if filtered_out:
        logger.warning(f"Filtered out {filtered_out} invalid user entries before classification")

    # Classify users
    classified = classify_users_parallel(users, run_id)

    # Save to OpenSearch
    save_classifications_bulk(classified)

    # Print results
    if print_results:
        print_classification_summary(classified)
        print_classification_results(classified)

    # Calculate statistics
    elapsed = time.time() - start_time
    per_user = elapsed / len(users) if users else 0

    logger.info(f"\nTotal execution time: {elapsed:.1f}s ({per_user:.2f}s per user)")
    logger.info(f"Results saved to OpenSearch index: '{TARGET_INDEX}'")
    logger.info(f"Query example: GET {TARGET_INDEX}/_search?q=classification_run_id:{run_id}")

    return {
        "success": True,
        "run_id": run_id,
        "total_users": len(users),
        "classified_count": len(classified),
        "execution_time_seconds": elapsed,
        "per_user_seconds": per_user,
        "target_index": TARGET_INDEX
    }


def main():
    """Main entry point."""
    try:
        run_classification_pipeline(
            max_users=MAX_USERS,
            force_recreate_index=False,
            print_results=True
        )
    except KeyboardInterrupt:
        logger.warning("\nClassification interrupted by user")
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
