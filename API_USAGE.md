# API Server Usage Guide

## Overview

The API server provides REST endpoints to generate and retrieve trending topics with filtering capabilities based on `user_input_id` and `source_id`.

## Starting the Server

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the API server
python api_server.py
```

The server will start on `http://0.0.0.0:5000` by default. You can configure the host and port using environment variables:

```bash
export API_HOST=0.0.0.0
export API_PORT=5000
python api_server.py
```

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API server is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Example:**
```bash
curl http://localhost:5000/health
```

---

### 2. Generate Trending Topics (Async)

**POST** `/api/trending-topics`

Start an asynchronous job to generate trending topics with optional filters. Returns immediately with a job ID and estimated completion time (2-5 minutes).

**Request Body:**
```json
{
  "user_input_id": "optional_user_input_id",
  "source_ids": ["source1", "source2"],
  "min_cluster_size": 5,
  "save_to_index": true
}
```

**Parameters:**
- `user_input_id` (optional): Filter posts by specific user input ID
- `source_ids` (optional): Array of source IDs to filter by. If `null` or not provided, all sources are included
- `min_cluster_size` (optional, default: 5): Minimum cluster size for trending topics
- `save_to_index` (optional, default: true): Whether to save results to OpenSearch index

**Response (202 Accepted):**
```json
{
  "success": true,
  "message": "Job started successfully. Processing will complete in approximately 2-5 minutes.",
  "data": {
    "job_id": "abc123def456...",
    "status": "pending",
    "estimated_completion_time": "2-5 minutes",
    "created_at": "2024-01-01T12:00:00Z",
    "check_status_url": "/api/trending-topics/status?user_input_id=user_123&source_ids=source1,source2",
    "get_results_url": "/api/trending-topics/results?user_input_id=user_123&source_ids=source1,source2"
  }
}
```

**Note:** The job runs in the background. Use the status and results endpoints to check progress and retrieve results.

**Examples:**

1. **Start job for all posts:**
```bash
curl -X POST http://localhost:5001/api/trending-topics \
  -H "Content-Type: application/json" \
  -d '{}'
```

2. **Start job for specific user_input_id:**
```bash
curl -X POST http://localhost:5001/api/trending-topics \
  -H "Content-Type: application/json" \
  -d '{
    "user_input_id": "user_123"
  }'
```

3. **Start job for specific sources:**
```bash
curl -X POST http://localhost:5001/api/trending-topics \
  -H "Content-Type: application/json" \
  -d '{
    "source_ids": ["source1", "source2"]
  }'
```

4. **Start job with both filters:**
```bash
curl -X POST http://localhost:5001/api/trending-topics \
  -H "Content-Type: application/json" \
  -d '{
    "user_input_id": "user_123",
    "source_ids": ["source1", "source2"],
    "min_cluster_size": 10
  }'
```

---

### 3. Check Job Status

**GET** `/api/trending-topics/status`

Check the status of a trending topics generation job.

**Query Parameters:**
- `user_input_id` (optional): User input ID used in the job
- `source_ids` (optional): Comma-separated list of source IDs

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": "abc123def456...",
    "status": "pending|processing|completed|failed",
    "progress": 75,
    "created_at": "2024-01-01T12:00:00Z",
    "finished_at": null,
    "estimated_time_remaining": null,
    "error": null
  }
}
```

**Status Values:**
- `pending`: Job created but not started yet
- `processing`: Job is currently running
- `completed`: Job finished successfully
- `failed`: Job failed with an error

**Examples:**

1. **Check status for specific user_input_id:**
```bash
curl "http://localhost:5001/api/trending-topics/status?user_input_id=user_123"
```

2. **Check status with source_ids:**
```bash
curl "http://localhost:5001/api/trending-topics/status?user_input_id=user_123&source_ids=source1,source2"
```

---

### 4. Get Job Results

**GET** `/api/trending-topics/results`

Get the results of a completed trending topics generation job.

**Query Parameters:**
- `user_input_id` (optional): User input ID used in the job
- `source_ids` (optional): Comma-separated list of source IDs

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "trending_topics": [
      {
        "topic": "المناقشات السياسية حول الانتخابات",
        "cluster_id": "cluster_0_123456",
        "post_count": 120,
        "engagement_score": 5000.0,
        "trending_score": 48.0,
        "keywords": ["انتخابات", "سياسة", "حكومة"],
        "representative_texts": ["...", "...", "..."],
        "member_ids": ["post_1", "post_2", ...],
        "generated_at": "2024-01-01T12:00:00Z",
        "timestamp": 1704110400000,
        "user_input_id": "user_123",
        "filtered_sources": ["source1", "source2"]
      }
    ],
    "total_topics": 10,
    "total_posts_processed": 1000,
    "filters_applied": {
      "user_input_id": "user_123",
      "source_ids": ["source1", "source2"]
    },
    "generated_at": "2024-01-01T12:00:00Z"
  }
}
```

**Response (202 Accepted - Still Processing):**
```json
{
  "success": false,
  "message": "Job is still processing. Please check status first.",
  "data": {
    "status": "processing",
    "progress": 75,
    "check_status_url": "/api/trending-topics/status?user_input_id=user_123&source_ids=source1,source2"
  }
}
```

**Examples:**

1. **Get results for specific user_input_id:**
```bash
curl "http://localhost:5001/api/trending-topics/results?user_input_id=user_123"
```

2. **Get results with source_ids:**
```bash
curl "http://localhost:5001/api/trending-topics/results?user_input_id=user_123&source_ids=source1,source2"
```

---

### 5. Get Trending Topics (From Index)

**GET** `/api/trending-topics`

Retrieve trending topics directly from the OpenSearch index with optional filters (bypasses job system).

**Query Parameters:**
- `user_input_id` (optional): Filter by user_input_id
- `source_id` (optional, can be multiple): Filter by source_id. Can be specified multiple times
- `limit` (optional, default: 20): Limit number of results
- `min_score` (optional): Minimum trending score

**Response:**
```json
{
  "success": true,
  "data": {
    "topics": [
      {
        "topic": "المناقشات السياسية حول الانتخابات",
        "cluster_id": "cluster_0_123456",
        "post_count": 120,
        "engagement_score": 5000.0,
        "trending_score": 48.0,
        "keywords": ["انتخابات", "سياسة", "حكومة"],
        "user_input_id": "user_123",
        "filtered_sources": ["source1", "source2"],
        ...
      }
    ],
    "total": 10
  }
}
```

**Examples:**

1. **Get all trending topics (top 20):**
```bash
curl http://localhost:5001/api/trending-topics
```

2. **Get trending topics for specific user_input_id:**
```bash
curl "http://localhost:5001/api/trending-topics?user_input_id=user_123"
```

3. **Get trending topics for specific source:**
```bash
curl "http://localhost:5001/api/trending-topics?source_id=source1"
```

4. **Get trending topics with multiple filters:**
```bash
curl "http://localhost:5001/api/trending-topics?user_input_id=user_123&source_id=source1&source_id=source2&limit=50"
```

5. **Get trending topics with minimum score:**
```bash
curl "http://localhost:5001/api/trending-topics?min_score=40.0&limit=10"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "success": false,
  "message": "Not enough posts (50 < 100)"
}
```

### 404 Not Found
```json
{
  "success": false,
  "message": "No posts found matching the filters",
  "data": {
    "filters_applied": {
      "user_input_id": "user_123",
      "source_ids": ["source1"]
    }
  }
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "message": "Internal server error: ..."
}
```

---

## Data Structure

### Trending Topic Document

Each trending topic document in OpenSearch contains:

- `topic`: Generated Arabic topic for the cluster
- `cluster_id`: Unique cluster identifier (includes filter hash for uniqueness)
- `post_count`: Number of posts in the cluster
- `engagement_score`: Total engagement (likes + retweets*2 + replies*1.5)
- `trending_score`: Calculated trending score
- `keywords`: Top 5 keywords from the cluster
- `representative_texts`: Sample texts from the cluster
- `member_ids`: IDs of all posts in the cluster
- `generated_at`: Generation timestamp (ISO format)
- `timestamp`: Generation timestamp (Unix milliseconds)
- `rank`: Ranking position (1 = most trending)
- **`user_input_id`**: User input ID used for filtering (new field)
- **`filtered_sources`**: Array of source IDs used for filtering (new field)

---

## OpenSearch Index Requirements

### Source Index (`user-input-posts`)

The source index should contain posts with the following fields:
- `post_text`, `text`, or `content`: Post content (required)
- `user_input_id`: User input ID (optional, for filtering)
- `source_id`: Source ID (optional, for filtering)
- `likes`, `retweets`, `replies`: Engagement metrics (optional)
- `author`, `created_at`, `timestamp`: Metadata (optional)

### Trending Topics Index (`trending-topics-index`)

The index is automatically created/updated with the following mapping:
- All standard trending topic fields
- `user_input_id` (keyword): User input ID filter applied
- `filtered_sources` (keyword): Array of source IDs filter applied

---

## Python Client Example

```python
import requests
import time

# Start async job to generate trending topics
response = requests.post(
    'http://localhost:5001/api/trending-topics',
    json={
        'user_input_id': 'user_123',
        'source_ids': ['source1', 'source2'],
        'min_cluster_size': 10
    }
)

if response.status_code == 202:
    job_data = response.json()['data']
    job_id = job_data['job_id']
    print(f"Job started: {job_id}")
    print(f"Estimated time: {job_data['estimated_completion_time']}")
    
    # Poll for status
    user_input_id = 'user_123'
    source_ids = 'source1,source2'
    
    while True:
        status_response = requests.get(
            'http://localhost:5001/api/trending-topics/status',
            params={
                'user_input_id': user_input_id,
                'source_ids': source_ids
            }
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()['data']
            status = status_data['status']
            progress = status_data.get('progress', 0)
            
            print(f"Status: {status} ({progress}%)")
            
            if status == 'completed':
                # Get results
                results_response = requests.get(
                    'http://localhost:5001/api/trending-topics/results',
                    params={
                        'user_input_id': user_input_id,
                        'source_ids': source_ids
                    }
                )
                
                if results_response.status_code == 200:
                    results = results_response.json()['data']
                    print(f"\nGenerated {results['total_topics']} trending topics")
                    for topic in results['trending_topics'][:5]:
                        print(f"- {topic['topic']} (Score: {topic['trending_score']})")
                break
            elif status == 'failed':
                print(f"Job failed: {status_data.get('error')}")
                break
        
        time.sleep(5)  # Wait 5 seconds before checking again
else:
    print(f"Error starting job: {response.json()}")

# Get trending topics from index (alternative method)
response = requests.get(
    'http://localhost:5001/api/trending-topics',
    params={
        'user_input_id': 'user_123',
        'limit': 10
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Found {data['data']['total']} topics in index")
```

---

## Notes

1. **Async Processing**: The POST endpoint starts jobs asynchronously and returns immediately. Use the status and results endpoints to check progress and retrieve results.

2. **Job ID**: Job IDs are generated from the filter parameters (`user_input_id` and `source_ids`). The same filters will always produce the same job ID, allowing you to check status or retrieve results for existing jobs.

3. **Processing Time**: Jobs typically take 2-5 minutes to complete, depending on the number of posts. Progress is updated at key stages (10%, 20%, 30%, 40%, 60%, 75%, 85%, 90%, 100%).

4. **Filtering**: When both `user_input_id` and `source_ids` are provided, posts must match BOTH filters (AND logic).

5. **Cluster ID Uniqueness**: Cluster IDs are made unique by appending a hash of the filters, preventing collisions when different filters produce the same cluster numbers.

6. **Job Storage**: Jobs are stored in memory. For production use, consider using Redis or a database for job persistence.

7. **Index Updates**: The trending topics index mapping is automatically updated when new fields are added. Existing documents will have `null` values for new fields until they are regenerated.

8. **CORS**: CORS is enabled by default, allowing requests from any origin. Adjust in `api_server.py` if needed.
