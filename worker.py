"""
Trending Topics Worker Service
Dedicated worker process for polling active user inputs from MySQL and processing them.
This runs as a separate service (systemd) and should NOT be run with PM2.

Usage:
    python worker.py

Or with systemd:
    systemctl start trending-topics-worker
    systemctl enable trending-topics-worker
"""

import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional, List, Dict
from dotenv import load_dotenv
import pymysql

# Import trending topics functions
from get_trending_topics import (
    validate_config,
    create_opensearch_client,
    fetch_posts,
    EmbeddingProcessor,
    analyze_trending_topics,
    save_trending_topics,
    clean_text,
    SOURCE_INDEX,
    ensure_embedding_mapping,
    save_embeddings_to_opensearch,
    EMBEDDING_DIM,
    MAX_TOPICS
)
from opensearchpy import helpers
import numpy as np

# ====== SETUP ======
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== JOB TRACKING SYSTEM ======
# In-memory job storage (can be replaced with Redis/database for production)
jobs = {}  # job_id -> {status, result, error, created_at, finished_at}
jobs_lock = __import__('threading').Lock()

def generate_job_id(user_input_id: Optional[str], source_ids: Optional[List[str]]) -> str:
    """Generate a unique job ID based on filters"""
    filter_str = f"{user_input_id}_{sorted(source_ids) if source_ids else 'all'}"
    return hashlib.md5(filter_str.encode()).hexdigest()

def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status by job_id"""
    with jobs_lock:
        return jobs.get(job_id)

def create_job(job_id: str, user_input_id: Optional[str], source_ids: Optional[List[str]]) -> Dict:
    """Create a new job entry"""
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "user_input_id": user_input_id,
            "source_ids": source_ids,
            "created_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "result": None,
            "error": None,
            "progress": 0,
            "estimated_time_remaining": None
        }
        return jobs[job_id]

def update_job_status(job_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None, progress: int = 0):
    """Update job status"""
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            if result:
                jobs[job_id]["result"] = result
            if error:
                jobs[job_id]["error"] = error
            jobs[job_id]["progress"] = progress
            if status in ["completed", "failed"]:
                jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()

# ====== DATABASE ACCESS ======

def validate_db_config():
    """Validate database configuration from environment variables."""
    required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")


def get_db_connection():
    """Create a MySQL database connection, retrying on failure."""
    retry_delay = float(os.getenv("DB_RETRY_DELAY_SECONDS", "5"))
    while True:
        try:
            return pymysql.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                port=int(os.getenv("DB_PORT", "3306")),
                cursorclass=pymysql.cursors.DictCursor,
            )
        except pymysql.MySQLError as exc:
            logger.warning("DB connection failed; retrying in %ss: %s", retry_delay, exc)
            time.sleep(retry_delay)


def fetch_active_inputs() -> List[Dict]:
    """Fetch active user tracking inputs ordered by creation time."""
    query = """
        SELECT id, user_id, country, keywords, accounts, active, created_at, updated_at
        FROM user_tracking_inputs
        WHERE active = 1
        ORDER BY created_at ASC
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        connection.close()


def normalize_source_ids(raw_accounts) -> Optional[List[str]]:
    """Normalize accounts field into a list of source IDs."""
    if raw_accounts is None:
        return None
    if isinstance(raw_accounts, list):
        return raw_accounts or None
    if isinstance(raw_accounts, str):
        try:
            parsed = json.loads(raw_accounts)
            return parsed or None
        except json.JSONDecodeError:
            return None
    return None

# ====== PROCESSING FUNCTIONS ======

def fetch_posts_with_filters(
    client,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None
):
    """
    Fetch posts with optional filters for user_input_id and source_ids.
    Includes cached embeddings when available.

    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by (None = all sources)

    Returns:
        List of filtered posts with cached embeddings if available
    """
    logger.info(f"📥 Fetching posts with filters: user_input_id={user_input_id}, source_ids={source_ids}")

    # Build query with filters
    must_clauses = []

    if user_input_id:
        must_clauses.append({"term": {"user_input_id": user_input_id}})

    if source_ids:
        must_clauses.append({"terms": {"source_id": source_ids}})

    query = {
        "query": {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}]
            }
        },
        "_source": [
            "post_text", "text", "content", "created_at", "timestamp",
            "author", "likes", "retweets", "replies", "user_input_id", "source_id",
            "embedding", "embedding_updated_at"  # Include cached embeddings
        ]
    }

    docs = []
    cached_count = 0
    try:
        for hit in helpers.scan(
            client,
            query=query,
            index=SOURCE_INDEX,
            size=500,
            scroll="10m",
            raise_on_error=False
        ):
            try:
                src = hit.get("_source", {})

                # Try multiple field names for post text
                text = (
                    src.get("post_text") or
                    src.get("text") or
                    src.get("content") or
                    ""
                )

                if not text or len(text.strip()) < 10:
                    continue

                cleaned = clean_text(text)
                if cleaned and len(cleaned) > 10:
                    # Get cached embedding if available
                    cached_embedding = src.get("embedding")
                    has_cached = cached_embedding is not None and len(cached_embedding) == EMBEDDING_DIM

                    if has_cached:
                        cached_count += 1

                    docs.append({
                        "id": hit["_id"],
                        "text": text.strip(),
                        "cleaned": cleaned,
                        "author": src.get("author"),
                        "created_at": src.get("created_at") or src.get("timestamp"),
                        "timestamp": src.get("timestamp"),
                        "post_created_at": src.get("post_created_at"),
                        "likes": src.get("likes", 0) or 0,
                        "retweets": src.get("retweets", 0) or 0,
                        "replies": src.get("replies", 0) or 0,
                        "user_input_id": src.get("user_input_id"),
                        "source_id": src.get("source_id"),
                        "cached_embedding": np.array(cached_embedding) if has_cached else None,
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error processing document {hit.get('_id')}: {e}")
                continue

        logger.info(f"✅ Loaded {len(docs)} valid posts ({cached_count} with cached embeddings, {len(docs) - cached_count} need embedding)")
        return docs

    except Exception as e:
        logger.error(f"❌ Error fetching posts: {e}")
        raise


def process_trending_topics_job(
    job_id: str,
    user_input_id: Optional[str],
    source_ids: Optional[List[str]],
    min_cluster_size: int,
    save_to_index: bool
):
    """
    Background function to process trending topics generation.
    Updates job status as it progresses.
    Uses cached embeddings when available to avoid recomputation.
    """
    try:
        logger.info(f"🔄 Starting job processing: {job_id}")
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
            posts = fetch_posts_with_filters(
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

            logger.info(f"✅ Found {len(posts)} posts matching filters")
            update_job_status(job_id, "processing", progress=30)

            # Import clustering components
            from get_trending_topics import (
                MIN_CLUSTER_SIZE,
                HDBSCAN_MIN_CLUSTER_SIZE,
                HDBSCAN_MIN_SAMPLES,
                PCA_TARGET_DIM,
                EMBEDDING_BATCH_SIZE
            )

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
            # Use stable filter key (job_id is md5 of filter params)
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
            logger.info(f"✅ Job completed successfully: {job_id}")
            
        finally:
            client.close()
            
    except ValueError as e:
        logger.error(f"❌ Validation error in job {job_id}: {e}")
        update_job_status(job_id, "failed", error=f"Validation error: {str(e)}")
        
    except Exception as e:
        logger.error(f"❌ Error processing job {job_id}: {e}", exc_info=True)
        update_job_status(job_id, "failed", error=f"Processing error: {str(e)}")


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

            job_id = generate_job_id(user_input_id, source_ids)
            existing_job = get_job_status(job_id)
            if existing_job and existing_job["status"] in ["pending", "processing"]:
                logger.info(f"⏳ Job {job_id} already in progress, skipping")
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
        logger.error(f"❌ Error in process_active_inputs: {e}", exc_info=True)
        raise


# ====== MAIN ======
if __name__ == '__main__':
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
