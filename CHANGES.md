# Changes Summary

## Overview

Added API server functionality to generate trending topics with filtering based on `user_input_id` and `source_id`, and added new fields to the trending topics index to track filter correlations.

## New Files

### 1. `api_server.py`
- Flask-based REST API server
- Endpoints for generating and retrieving trending topics
- Supports filtering by `user_input_id` and `source_ids`
- Handles requests asynchronously

### 2. `API_USAGE.md`
- Complete API documentation
- Examples for all endpoints
- Error handling guide
- Python client examples

## Modified Files

### 1. `get_trending_topics.py`
- **Updated index mapping** to include:
  - `user_input_id` (keyword field)
  - `filtered_sources` (keyword field)
- **Enhanced mapping update logic** to automatically add missing fields to existing indices

### 2. `requirements.txt`
- Added `flask>=2.3.0`
- Added `flask-cors>=4.0.0`

## New Features

### 1. Async Job System

The API now uses an asynchronous job system to handle long-running trending topics generation:

- **Jobs run in background threads** - No timeout issues
- **Progress tracking** - Check job status at any time
- **Job persistence** - Same filters = same job ID (can check status/results later)
- **Estimated completion time** - 2-5 minutes typically

### 2. API Endpoints

#### POST `/api/trending-topics` (Async)
Start an async job to generate trending topics:
- Returns immediately with job ID (202 Accepted)
- Filter by `user_input_id`
- Filter by `source_ids` (array) or all sources
- Configurable `min_cluster_size`
- Option to save/not save to index
- Estimated completion: 2-5 minutes

#### GET `/api/trending-topics/status`
Check the status of a running job:
- Query by `user_input_id` and `source_ids`
- Returns: `pending`, `processing`, `completed`, or `failed`
- Includes progress percentage (0-100)
- Shows error message if failed

#### GET `/api/trending-topics/results`
Get results of a completed job:
- Query by `user_input_id` and `source_ids`
- Returns full trending topics data when completed
- Returns 202 if still processing

#### GET `/api/trending-topics`
Retrieve trending topics directly from OpenSearch index:
- Filter by `user_input_id`
- Filter by `source_id` (can specify multiple)
- Limit results
- Filter by minimum trending score

#### GET `/health`
Health check endpoint

### 2. Filtering Capabilities

The API now supports:
- **`user_input_id`**: Filter posts by specific user input ID
- **`source_ids`**: Filter posts by one or more source IDs
- **Combined filters**: Use both filters together (AND logic)
- **All sources**: Omit `source_ids` or pass `null` to include all sources

### 3. Index Enhancements

The trending topics index now tracks:
- **`user_input_id`**: Which user input was used to generate the topics
- **`filtered_sources`**: Which sources were included in the analysis

This allows you to:
- Track trending topics by user input
- Track trending topics by source
- Understand correlations between different filters
- Query historical trending topics with specific filters

## Usage

### Start the API Server

```bash
python api_server.py
```

### Start Trending Topics Job (Async)

```bash
# Start the job (returns immediately)
curl -X POST http://localhost:5001/api/trending-topics \
  -H "Content-Type: application/json" \
  -d '{
    "user_input_id": "user_123",
    "source_ids": ["source1", "source2"]
  }'

# Check job status
curl "http://localhost:5001/api/trending-topics/status?user_input_id=user_123&source_ids=source1,source2"

# Get results when completed
curl "http://localhost:5001/api/trending-topics/results?user_input_id=user_123&source_ids=source1,source2"
```

### Get Trending Topics from Index

```bash
curl "http://localhost:5001/api/trending-topics?user_input_id=user_123&limit=10"
```

## Data Structure Changes

### Trending Topic Document (New Fields)

```json
{
  "topic": "...",
  "cluster_id": "...",
  "user_input_id": "user_123",        // NEW
  "filtered_sources": ["source1"],     // NEW
  ...
}
```

### Source Index Requirements

Your `user-input-posts` index should have:
- `user_input_id` (keyword): User input identifier
- `source_id` (keyword): Source identifier

## Migration Notes

1. **Existing Indices**: The mapping update logic will automatically add the new fields to existing indices. Existing documents will have `null` values until regenerated.

2. **Cluster ID Uniqueness**: Cluster IDs now include a hash of the filters to ensure uniqueness across different filter combinations.

3. **Backward Compatibility**: The original `get_trending_topics.py` script still works without filters. The API is an additional layer on top.

## Configuration

Add to `.env` (optional):
```env
API_HOST=0.0.0.0
API_PORT=5000
```

## Testing

1. **Health Check**:
   ```bash
   curl http://localhost:5001/health
   ```

2. **Start Job with Filters**:
   ```bash
   curl -X POST http://localhost:5001/api/trending-topics \
     -H "Content-Type: application/json" \
     -d '{"user_input_id": "test_user", "source_ids": ["test_source"]}'
   ```

3. **Check Job Status**:
   ```bash
   curl "http://localhost:5001/api/trending-topics/status?user_input_id=test_user&source_ids=test_source"
   ```

4. **Get Results**:
   ```bash
   curl "http://localhost:5001/api/trending-topics/results?user_input_id=test_user&source_ids=test_source"
   ```

5. **Retrieve from Index**:
   ```bash
   curl "http://localhost:5001/api/trending-topics?user_input_id=test_user"
   ```

## Next Steps

1. Ensure your `user-input-posts` index has `user_input_id` and `source_id` fields
2. Start the API server: `python api_server.py`
3. Test the endpoints using the examples in `API_USAGE.md`
4. Integrate the API into your application
