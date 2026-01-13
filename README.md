# Twitter Trend Analysis - Trending Topics Generator

A comprehensive system for analyzing posts from your existing Elasticsearch/OpenSearch index, generating topics, and identifying trending topics based on clustering and engagement metrics.

## 📋 Overview

This project processes posts from your existing `user-input-posts` index and:
1. **Topic Generation**: Generates meaningful Arabic topics for individual posts
2. **Content Clustering**: Groups similar content together using advanced ML clustering
3. **Trending Topics**: Identifies trending topics based on cluster size, engagement, and recency

## ✨ Features

- **Trending Topics Detection**: Automatically identifies trending topics from your posts
- **Topic Extraction**: Generates concise Arabic topics for individual posts and clusters
- **Advanced Clustering**: Uses HDBSCAN algorithm for intelligent content clustering
- **Engagement Scoring**: Calculates trending scores based on post count, engagement, and recency
- **OpenSearch Integration**: Full integration with OpenSearch/Elasticsearch for data storage and retrieval
- **Arabic Text Processing**: Specialized text cleaning and normalization for Arabic content
- **No Scraping Required**: Works directly with your existing `user-input-posts` index

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenSearch instance (local or remote)
- DeepSeek API key

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd get_trend_twitter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
# OpenSearch/Elasticsearch Configuration
OPENSEARCH_NODE=http://localhost:9200
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin

# Source Index (your existing index with posts)
OPENSEARCH_INDEX=user-input-posts

# Target Index for Trending Topics
TRENDING_INDEX=trending-topics-index

# DeepSeek API Configuration
DEEPSEEK_API_KEY=your-api-key-here
DEEPSEEK_MODEL=deepseek-chat

# Processing Configuration
BATCH_SIZE=100
MAX_WORKERS=5
MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPLES=3
PCA_TARGET_DIM=100
EMBEDDING_BATCH_SIZE=32
```

## 📁 Project Structure

```
get_trend_twitter/
├── get_trending_topics.py        # Main script: Generate trending topics (RECOMMENDED)
├── get_topic.py                  # Generate topics for individual posts
├── vectoriza0tion.py             # Advanced content clustering
├── get_community_sections.py     # User political classification (optional)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .env                          # Environment variables (create this)
```

## 🔧 Usage

### 🚀 Quick Start: Generate Trending Topics (Recommended)

The main script that does everything - reads from your `user-input-posts` index and generates trending topics.

```bash
python get_trending_topics.py
```

**What it does:**
1. Fetches all posts from `user-input-posts` index
2. Creates embeddings for posts using multilingual sentence transformers
3. Clusters similar posts together using HDBSCAN
4. Generates topics for each cluster using DeepSeek API
5. Calculates trending scores based on:
   - Cluster size (number of posts)
   - Engagement metrics (likes, retweets, replies)
   - Recency
6. Saves trending topics to `trending-topics-index`
7. Exports results to JSON file

**Output:**
- Creates/updates `trending-topics-index` in OpenSearch
- Generates JSON file: `trending_topics_YYYYMMDD_HHMMSS.json`
- Prints top 20 trending topics to console

**Expected Post Fields:**
The script looks for posts with these fields (tries multiple field names):
- `post_text`, `text`, or `content` - The post content
- `likes`, `retweets`, `replies` - Engagement metrics (optional)
- `author`, `created_at`, `timestamp` - Metadata (optional)

### 2. Generate Topics for Individual Posts

Generates concise Arabic topics for individual posts and updates them in the index.

```bash
python get_topic.py
```

**What it does:**
- Fetches posts from `user-input-posts` index
- Cleans and normalizes Arabic text
- Generates topics using DeepSeek API
- Updates posts with generated topics

**Configuration:**
- `BATCH_SIZE`: Number of posts to process in each batch
- `MAX_WORKERS`: Number of parallel workers
- `MIN_MEANINGFUL_WORDS`: Minimum words required for a post to be processed

### 3. Advanced Content Clustering

Groups similar content together using advanced ML clustering techniques with detailed analysis.

```bash
python vectoriza0tion.py
```

**What it does:**
- Fetches all documents from OpenSearch
- Creates embeddings using multilingual sentence transformers
- Reduces dimensionality with PCA
- Performs HDBSCAN/DBSCAN clustering
- Generates topics for each cluster
- Saves clusters to OpenSearch

**Clustering Pipeline:**
1. Text embedding using `distiluse-base-multilingual-cased-v2`
2. Dimensionality reduction with PCA
3. Initial clustering with HDBSCAN
4. Quality filtering based on coherence scores
5. Noise point reclustering
6. Similar cluster merging
7. Topic generation for each cluster

**Output:**
- Creates/updates `clustered-tweets-index` in OpenSearch
- Each cluster document contains:
  - `group_topic`: Generated topic for the cluster
  - `cluster_id`: Unique cluster identifier
  - `cluster_size`: Number of documents in cluster
  - `cluster_keywords`: Top keywords
  - `representative_texts`: Sample texts from cluster
  - `member_ids`: IDs of all documents in cluster
  - `authors`: Unique authors in cluster

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENSEARCH_NODE` | OpenSearch/Elasticsearch server URL | `http://localhost:9200` |
| `OPENSEARCH_USERNAME` | OpenSearch username | `admin` |
| `OPENSEARCH_PASSWORD` | OpenSearch password | `admin` |
| `OPENSEARCH_INDEX` | Source index with posts | `user-input-posts` |
| `TRENDING_INDEX` | Target index for trending topics | `trending-topics-index` |
| `DEEPSEEK_API_KEY` | DeepSeek API key | **Required** |
| `DEEPSEEK_MODEL` | DeepSeek model name | `deepseek-chat` |
| `BATCH_SIZE` | Batch size for processing | `100` |
| `MAX_WORKERS` | Number of parallel workers | `5` |
| `MIN_CLUSTER_SIZE` | Minimum cluster size for trending | `5` |
| `HDBSCAN_MIN_CLUSTER_SIZE` | HDBSCAN min cluster size | `5` |
| `HDBSCAN_MIN_SAMPLES` | HDBSCAN min samples | `3` |
| `PCA_TARGET_DIM` | PCA target dimensions | `100` |
| `EMBEDDING_BATCH_SIZE` | Batch size for embeddings | `32` |

## 📦 Dependencies

- **requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **opensearch-py**: OpenSearch client library
- **tqdm**: Progress bar visualization
- **urllib3**: HTTP client utilities
- **tenacity**: Retry logic for API calls
- **numpy**: Numerical computing
- **sentence-transformers**: Text embeddings
- **scikit-learn**: Machine learning utilities
- **hdbscan**: Hierarchical density-based clustering

See `requirements.txt` for specific versions.

## 🔍 OpenSearch Index Structure

### Source Index (`user-input-posts`)

**Required fields:**
- `post_text`, `text`, or `content`: Post content (at least one must exist)

**Optional fields (for better trending analysis):**
- `likes`: Number of likes
- `retweets`: Number of retweets
- `replies`: Number of replies
- `author`: Author username
- `created_at` or `timestamp`: Creation timestamp

**Example document structure:**
```json
{
  "post_text": "Your post content here...",
  "likes": 10,
  "retweets": 5,
  "replies": 2,
  "author": "username",
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Trending Topics Index (`trending-topics-index`)

Fields:
- `topic`: Generated topic for the cluster (Arabic)
- `cluster_id`: Unique cluster identifier
- `post_count`: Number of posts in cluster
- `engagement_score`: Total engagement (likes + retweets*2 + replies*1.5)
- `trending_score`: Calculated trending score
- `keywords`: Top 5 keywords from cluster
- `representative_texts`: Sample texts from cluster
- `member_ids`: IDs of all posts in cluster
- `generated_at`: Generation timestamp
- `rank`: Ranking position (1 = most trending)

### Clustered Tweets Index (`clustered-tweets-index`) - Optional

Fields:
- `group_topic`: Generated topic for the cluster
- `cluster_id`: Unique cluster identifier
- `cluster_size`: Number of documents in cluster
- `cluster_keywords`: Top keywords in cluster
- `representative_texts`: Sample texts from cluster
- `member_ids`: IDs of all documents in cluster
- `authors`: Unique authors in cluster
- `fetched_at`: Data fetch timestamp
- `updated_at`: Last update timestamp

## 🛠️ Development

### Running Scripts

**Main workflow (recommended):**
```bash
# Generate trending topics from your existing posts
python get_trending_topics.py
```

**Individual scripts:**
```bash
# Generate topics for individual posts
python get_topic.py

# Advanced clustering with detailed analysis
python vectoriza0tion.py

# Classify users (optional, requires user data)
python get_community_sections.py
```

### Workflow

1. **Ensure your posts are in `user-input-posts` index**
   - The script will automatically read from this index
   - No scraping or data collection needed

2. **Run the trending topics script**
   ```bash
   python get_trending_topics.py
   ```

3. **Check results**
   - Console output shows top 20 trending topics
   - JSON file saved with all trending topics
   - Results saved to `trending-topics-index` in OpenSearch

### Error Handling

All scripts include:
- Retry logic for API calls
- Error handling for OpenSearch operations
- Graceful degradation on failures
- Comprehensive logging

### Logging

All scripts use Python's `logging` module with INFO level by default. Logs include:
- Progress indicators
- Error messages
- Statistics and summaries
- API response times

## 📊 Performance Considerations

- **Parallel Processing**: All scripts use `ThreadPoolExecutor` for parallel processing
- **Batch Operations**: Bulk operations for OpenSearch updates
- **Rate Limiting**: Built-in rate limiting for API calls
- **Memory Management**: Efficient handling of large datasets using scroll API

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep API keys secure
- Use environment variables for sensitive data
- Verify OpenSearch connection security settings

## 📝 Notes

- The system is optimized for Arabic text processing
- DeepSeek API is used for AI-powered classification and topic generation
- Clustering uses advanced ML algorithms (HDBSCAN, DBSCAN)
- All timestamps are stored in UTC

## 🤝 Contributing

1. Ensure all scripts run without errors
2. Maintain code documentation
3. Follow existing code style
4. Test with sample data before production use

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- DeepSeek for AI API services
- OpenSearch for search and analytics
- Sentence Transformers for multilingual embeddings

