"""Worker application entry point."""

import os
import time
from dotenv import load_dotenv

from src.config.validation import validate_config
from src.infra.db_client import (
    validate_db_config,
    fetch_active_inputs,
    normalize_source_ids
)
from src.services.job_tracker import generate_job_id
from src.services.worker_service import process_trending_topics_job
from src.logging.logger import get_logger

# ====== SETUP ======
load_dotenv()

logger = get_logger(__name__)


def process_active_inputs():
    """Fetch and process active user inputs."""
    try:
        logger.info("📥 Fetching active user tracking inputs...")
        active_inputs = fetch_active_inputs()
        logger.info(f"✅ Found {len(active_inputs)} active inputs to process")

        if not active_inputs:
            logger.info("ℹ️ No active inputs found. Exiting.")
            return

        for input_row in active_inputs:
            user_input_id = str(input_row.get("id"))
            source_ids = normalize_source_ids(input_row.get("accounts"))
            min_cluster_size = 5
            save_to_index = True

            logger.info(f"📊 Processing input: user_input_id={user_input_id}, source_ids={source_ids}")

            # Generate job ID for logging/tracking purposes only
            job_id = generate_job_id(user_input_id, source_ids)
            
            # Process immediately without checking cache
            process_trending_topics_job(
                job_id,
                user_input_id,
                source_ids,
                min_cluster_size,
                save_to_index
            )
    except Exception as e:
        logger.error(f"❌ Error in process_active_inputs: {e}", exc_info=True)
        raise


def main():
    """Main entry point for worker service."""
    logger.info("🚀 Starting Trending Topics Worker Service")
    logger.info("📡 This service polls MySQL for active user inputs")
    logger.info("⚠️  This should be run with systemd, NOT PM2")
    
    # Validate configuration
    try:
        validate_config()
        validate_db_config()
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        exit(1)
    
    poll_seconds = int(os.getenv("ACTIVE_INPUT_POLL_SECONDS", "300"))
    if poll_seconds <= 0:
        process_active_inputs()
        exit(0)

    while True:
        process_active_inputs()
        logger.info(f"⏳ Sleeping {poll_seconds}s before next poll")
        time.sleep(poll_seconds)


if __name__ == '__main__':
    main()
