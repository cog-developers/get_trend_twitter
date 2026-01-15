
"""
API Server for Trending Topics Generation
Provides REST API endpoints to generate trending topics with filtering options.
"""

import os
import json
import time
import logging
import threading
import hashlib
from datetime import datetime
from typing import Optional, List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pika
from pika.exceptions import AMQPConnectionError

# Import trending topics functions
from get_trending_topics import (
    validate_config,
    create_opensearch_client,
    fetch_posts,
    EmbeddingProcessor,
    analyze_trending_topics,
    save_trending_topics,
    print_trending_topics
)

# ====== SETUP ======
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ====== JOB TRACKING SYSTEM ======
# In-memory job storage (can be replaced with Redis/database for production)
jobs = {}  # job_id -> {status, result, error, created_at, finished_at}
jobs_lock = threading.Lock()

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

# ====== API ENDPOINTS ======

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route('/api/trending-topics', methods=['POST'])
def generate_trending_topics():
    """
    Start async job to generate trending topics with optional filters.
    Returns immediately with job_id and estimated completion time.
    
    Request Body (JSON):
    {
        "user_input_id": "optional_user_input_id",
        "source_ids": ["source1", "source2"] or null for all sources,
        "min_cluster_size": 5 (optional),
        "save_to_index": true (optional, default: true)
    }
    
    Returns:
    {
        "success": true,
        "message": "Job started successfully. Processing will complete in approximately 2-5 minutes.",
        "data": {
            "job_id": "abc123...",
            "status": "pending",
            "estimated_completion_time": "2-5 minutes",
            "check_status_url": "/api/trending-topics/status?user_input_id=...&source_ids=...",
            "get_results_url": "/api/trending-topics/results?user_input_id=...&source_ids=..."
        }
    }
    """
    try:
        # Parse request data
        data = request.get_json() or {}
        user_input_id = data.get('user_input_id')
        source_ids = data.get('source_ids')  # Can be list or null
        min_cluster_size = data.get('min_cluster_size', 5)
        save_to_index = data.get('save_to_index', True)
        
        logger.info(f"📥 Received async job request: user_input_id={user_input_id}, source_ids={source_ids}")
        
        # Generate job ID based on filters
        job_id = generate_job_id(user_input_id, source_ids)
        
        # Check if job already exists
        existing_job = get_job_status(job_id)
        if existing_job:
            if existing_job["status"] == "processing":
                return jsonify({
                    "success": True,
                    "message": "Job already in progress",
                    "data": {
                        "job_id": job_id,
                        "status": existing_job["status"],
                        "created_at": existing_job["created_at"],
                        "check_status_url": f"/api/trending-topics/status?user_input_id={user_input_id or ''}&source_ids={','.join(source_ids) if source_ids else ''}",
                        "get_results_url": f"/api/trending-topics/results?user_input_id={user_input_id or ''}&source_ids={','.join(source_ids) if source_ids else ''}"
                    }
                }), 200
            elif existing_job["status"] == "completed":
                return jsonify({
                    "success": True,
                    "message": "Job already completed. Use GET /api/trending-topics/results to retrieve results.",
                    "data": {
                        "job_id": job_id,
                        "status": existing_job["status"],
                        "completed_at": existing_job["finished_at"],
                        "get_results_url": f"/api/trending-topics/results?user_input_id={user_input_id or ''}&source_ids={','.join(source_ids) if source_ids else ''}"
                    }
                }), 200
        
        # Create new job
        job = create_job(job_id, user_input_id, source_ids)
        
        # Start background processing
        thread = threading.Thread(
            target=process_trending_topics_job,
            args=(job_id, user_input_id, source_ids, min_cluster_size, save_to_index),
            daemon=True
        )
        thread.start()
        
        logger.info(f"🚀 Started background job: {job_id}")
        
        return jsonify({
            "success": True,
            "message": "Job started successfully. Processing will complete in approximately 2-5 minutes.",
            "data": {
                "job_id": job_id,
                "status": "pending",
                "estimated_completion_time": "2-5 minutes",
                "created_at": job["created_at"],
                "check_status_url": f"/api/trending-topics/status?user_input_id={user_input_id or ''}&source_ids={','.join(source_ids) if source_ids else ''}",
                "get_results_url": f"/api/trending-topics/results?user_input_id={user_input_id or ''}&source_ids={','.join(source_ids) if source_ids else ''}"
            }
        }), 202  # 202 Accepted
        
    except Exception as e:
        logger.error(f"❌ Error starting job: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error starting job: {str(e)}"
        }), 500


@app.route('/api/trending-topics/status', methods=['GET'])
def check_job_status():
    """
    Check the status of a trending topics generation job.
    
    Query Parameters:
    - user_input_id: User input ID used in the job
    - source_ids: Comma-separated list of source IDs (or omit for all sources)
    
    Returns:
    {
        "success": true,
        "data": {
            "job_id": "...",
            "status": "pending|processing|completed|failed",
            "progress": 0-100,
            "created_at": "...",
            "finished_at": "...",
            "estimated_time_remaining": "...",
            "error": "..." (if failed)
        }
    }
    """
    try:
        user_input_id = request.args.get('user_input_id') or None
        source_ids_param = request.args.get('source_ids', '')
        source_ids = [s.strip() for s in source_ids_param.split(',')] if source_ids_param else None
        
        # Generate job ID from filters
        job_id = generate_job_id(user_input_id, source_ids)
        
        job = get_job_status(job_id)
        
        if not job:
            return jsonify({
                "success": False,
                "message": f"Job not found for user_input_id={user_input_id}, source_ids={source_ids}"
            }), 404
        
        return jsonify({
            "success": True,
            "data": {
                "job_id": job["job_id"],
                "status": job["status"],
                "progress": job.get("progress", 0),
                "created_at": job["created_at"],
                "finished_at": job.get("finished_at"),
                "estimated_time_remaining": job.get("estimated_time_remaining"),
                "error": job.get("error")
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error checking job status: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error checking status: {str(e)}"
        }), 500


@app.route('/api/trending-topics/results', methods=['GET'])
def get_trending_topics_results():
    """
    Get trending topics results directly from OpenSearch based on user_input_id and source_ids.
    No longer uses job_id - queries OpenSearch index directly.
    
    Query Parameters:
    - user_input_id: User input ID to filter by
    - source_ids: Comma-separated list of source IDs (or omit for all sources)
    - limit: Limit number of results (default: 20)
    - min_score: Minimum trending score (optional)
    
    Returns:
    {
        "success": true,
        "data": {
            "trending_topics": [...],
            "total_topics": 10,
            "filters_applied": {...}
        }
    }
    """
    try:
        from get_trending_topics import create_opensearch_client, TRENDING_INDEX
        
        client = create_opensearch_client()
        
        try:
            # Get query parameters
            user_input_id = request.args.get('user_input_id') or None
            source_ids_param = request.args.get('source_ids', '')
            source_ids = [s.strip() for s in source_ids_param.split(',')] if source_ids_param else None
            limit = int(request.args.get('limit', 20))
            min_score = request.args.get('min_score', type=float)
            
            # Build query
            must_clauses = []
            
            if user_input_id:
                must_clauses.append({"term": {"user_input_id": user_input_id}})
            
            if source_ids:
                # source_ids can be a list, handle both single and multiple values
                if isinstance(source_ids, list) and len(source_ids) > 0:
                    must_clauses.append({"terms": {"filtered_sources": source_ids}})
                elif source_ids:
                    must_clauses.append({"term": {"filtered_sources": source_ids}})
            
            query = {
                "size": limit,
                "sort": [{"trending_score": {"order": "desc"}}],
                "query": {
                    "bool": {
                        "must": must_clauses if must_clauses else [{"match_all": {}}]
                    }
                }
            }
            
            if min_score:
                query["query"]["bool"]["must"].append({
                    "range": {"trending_score": {"gte": min_score}}
                })
            
            # Search OpenSearch
            response = client.search(index=TRENDING_INDEX, body=query)
            hits = response.get('hits', {}).get('hits', [])
            
            topics = [hit['_source'] for hit in hits]
            
            return jsonify({
                "success": True,
                "data": {
                    "trending_topics": topics,
                    "total_topics": len(topics),
                    "filters_applied": {
                        "user_input_id": user_input_id,
                        "source_ids": source_ids
                    }
                }
            }), 200
            
        finally:
            client.close()
            
    except Exception as e:
        logger.error(f"❌ Error getting results: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Error getting results: {str(e)}"
        }), 500


# GET endpoint removed - now using RabbitMQ consumer instead


# ====== BACKGROUND PROCESSING ======

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
    """
    try:
        logger.info(f"🔄 Starting job processing: {job_id}")
        update_job_status(job_id, "processing", progress=10)
        
        # Validate configuration
        validate_config()
        
        # Create OpenSearch client
        client = create_opensearch_client()
        
        try:
            update_job_status(job_id, "processing", progress=20)
            
            # Fetch posts with filters
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
            
            # Create embeddings
            update_job_status(job_id, "processing", progress=40)
            embedding_processor = EmbeddingProcessor()
            embeddings = embedding_processor.create_embeddings(posts)
            
            # Reduce dimensionality
            update_job_status(job_id, "processing", progress=60)
            embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)
            
            # Cluster documents
            update_job_status(job_id, "processing", progress=75)
            labels = embedding_processor.cluster_documents(embeddings_reduced)
            
            # Analyze trending topics
            update_job_status(job_id, "processing", progress=85)
            trending_topics = analyze_trending_topics(posts, labels, embeddings_reduced)
            
            # Add filter metadata to each topic and make cluster_id unique
            filtered_sources = source_ids if source_ids else []
            filter_hash = hash(f"{user_input_id}_{str(source_ids)}")
            
            for topic in trending_topics:
                topic['user_input_id'] = user_input_id
                topic['filtered_sources'] = filtered_sources
                # Make cluster_id unique by adding filter hash
                original_cluster_id = topic.get('cluster_id', '')
                topic['cluster_id'] = f"{original_cluster_id}_{abs(filter_hash)}"
            
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


# ====== HELPER FUNCTIONS ======

def fetch_posts_with_filters(
    client,
    user_input_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None
):
    """
    Fetch posts with optional filters for user_input_id and source_ids.
    
    Args:
        client: OpenSearch client
        user_input_id: Optional user input ID to filter by
        source_ids: Optional list of source IDs to filter by (None = all sources)
    
    Returns:
        List of filtered posts
    """
    from get_trending_topics import SOURCE_INDEX, clean_text
    
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
            "author", "likes", "retweets", "replies", "user_input_id", "source_id"
        ]
    }
    
    docs = []
    try:
        from opensearchpy import helpers
        
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
                    docs.append({
                        "id": hit["_id"],
                        "text": text.strip(),
                        "cleaned": cleaned,
                        "author": src.get("author"),
                        "created_at": src.get("created_at") or src.get("timestamp"),
                        "likes": src.get("likes", 0) or 0,
                        "retweets": src.get("retweets", 0) or 0,
                        "replies": src.get("replies", 0) or 0,
                        "user_input_id": src.get("user_input_id"),
                        "source_id": src.get("source_id"),
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error processing document {hit.get('_id')}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(docs)} valid posts with filters")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error fetching posts: {e}")
        raise


# ====== RABBITMQ CONSUMER ======

def setup_rabbitmq_consumer():
    """
    Setup RabbitMQ consumer to listen to trending-topics queue.
    Processes messages containing user-input-id and source-ids.
    """
    rabbitmq_url = os.getenv('RABBITMQ_URL', 'amqp://admin:admin@185.217.126.143:5672')
    queue_name = 'trending-topics'
    
    def on_message(channel, method, properties, body):
        """Process incoming message from RabbitMQ queue"""
        try:
            logger.info(f"📨 Received message from RabbitMQ queue: {queue_name}")
            
            # Parse message
            message = json.loads(body)
            user_input_id = message.get('user-input-id') or message.get('user_input_id')
            source_ids = message.get('source-ids') or message.get('source_ids')
            min_cluster_size = message.get('min_cluster_size', 5)
            save_to_index = message.get('save_to_index', True)
            
            logger.info(f"📥 Processing request: user_input_id={user_input_id}, source_ids={source_ids}")
            
            # Generate job ID
            job_id = generate_job_id(user_input_id, source_ids)
            
            # Check if job already exists
            existing_job = get_job_status(job_id)
            if existing_job and existing_job["status"] in ["pending", "processing"]:
                logger.info(f"⏳ Job {job_id} already in progress, skipping")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            
            # Create new job
            job = create_job(job_id, user_input_id, source_ids)
            
            # Start background processing
            thread = threading.Thread(
                target=process_trending_topics_job,
                args=(job_id, user_input_id, source_ids, min_cluster_size, save_to_index),
                daemon=True
            )
            thread.start()
            
            logger.info(f"🚀 Started background job from RabbitMQ: {job_id}")
            
            # Acknowledge message
            channel.basic_ack(delivery_tag=method.delivery_tag)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in message: {e}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"❌ Error processing RabbitMQ message: {e}", exc_info=True)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
    
    def connect_and_consume():
        """Connect to RabbitMQ and start consuming messages"""
        while True:
            try:
                logger.info(f"🔌 Connecting to RabbitMQ: {rabbitmq_url.split('@')[1] if '@' in rabbitmq_url else rabbitmq_url}")
                
                # Parse connection URL
                connection_params = pika.URLParameters(rabbitmq_url)
                connection = pika.BlockingConnection(connection_params)
                channel = connection.channel()
                
                # Declare queue (create if doesn't exist)
                channel.queue_declare(queue=queue_name, durable=True)
                
                # Set QoS to process one message at a time
                channel.basic_qos(prefetch_count=1)
                
                # Set up consumer
                channel.basic_consume(
                    queue=queue_name,
                    on_message_callback=on_message
                )
                
                logger.info(f"✅ RabbitMQ consumer started. Listening to queue: {queue_name}")
                logger.info(f"⏳ Waiting for messages. To exit press CTRL+C")
                
                # Start consuming
                channel.start_consuming()
                
            except AMQPConnectionError as e:
                logger.error(f"❌ RabbitMQ connection error: {e}. Retrying in 10 seconds...")
                time.sleep(10)
            except KeyboardInterrupt:
                logger.info("🛑 RabbitMQ consumer stopped by user")
                if 'connection' in locals() and not connection.is_closed:
                    connection.close()
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in RabbitMQ consumer: {e}", exc_info=True)
                time.sleep(10)
    
    # Start consumer in background thread
    consumer_thread = threading.Thread(target=connect_and_consume, daemon=True)
    consumer_thread.start()
    logger.info("🐰 RabbitMQ consumer thread started")


# ====== MAIN ======
if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5001))
    host = os.getenv('API_HOST', '0.0.0.0')
    
    logger.info(f"🚀 Starting API server on {host}:{port}")
    logger.info(f"📡 Endpoints:")
    logger.info(f"   POST /api/trending-topics - Generate trending topics with filters")
    logger.info(f"   GET  /api/trending-topics/results - Get trending topics from index")
    logger.info(f"   GET  /api/trending-topics/status - Check job status")
    logger.info(f"   GET  /health - Health check")
    
    # Start RabbitMQ consumer
    setup_rabbitmq_consumer()
    
    app.run(host=host, port=port, debug=False)
