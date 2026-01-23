# Trending Topics - Code Flow Documentation

## Overview

This document explains how all modules in `src/` work together to process social media posts and identify trending topics.

---

## Directory Structure

```
src/
├── config/
│   └── settings.py        # Configuration & environment variables
├── database/
│   ├── opensearch.py      # OpenSearch database operations
│   └── mysql.py           # MySQL database operations
├── processing/
│   ├── text.py            # Text cleaning & normalization
│   └── embeddings.py      # ML embeddings & clustering
├── services/
│   ├── posts.py           # Fetch posts from OpenSearch
│   ├── topics.py          # Generate topics via DeepSeek AI
│   └── analysis.py        # Analyze clusters & score trends
├── jobs/
│   └── tracker.py         # Job state management
├── output/
│   └── formatters.py      # Console & JSON output
├── trending.py            # Main pipeline orchestration
└── worker.py              # Background worker service
```

---

## Module Descriptions

### 1. `config/settings.py`
**Role**: Central configuration hub

```
┌─────────────────────────────────────────────────────────┐
│                    settings.py                          │
├─────────────────────────────────────────────────────────┤
│  Environment Variables:                                 │
│  • OPENSEARCH_NODE, USERNAME, PASSWORD                  │
│  • DEEPSEEK_API_KEY, MODEL                              │
│  • DB_HOST, DB_USER, DB_PASSWORD, DB_NAME               │
│  • MIN_CLUSTER_SIZE, MAX_TOPICS, PCA_TARGET_DIM         │
│                                                         │
│  Constants:                                             │
│  • ARABIC_STOPWORDS (set of 30 words)                   │
│  • EMBEDDING_DIM = 512                                  │
│                                                         │
│  Functions:                                             │
│  • validate_config() → checks OpenSearch/DeepSeek vars  │
│  • validate_db_config() → checks MySQL vars             │
│  • get_logger(name) → returns configured logger         │
└─────────────────────────────────────────────────────────┘
```

---

### 2. `database/opensearch.py`
**Role**: All OpenSearch database operations

```
┌─────────────────────────────────────────────────────────┐
│                   opensearch.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  create_opensearch_client()                             │
│  ├── Connects to OpenSearch cluster                     │
│  ├── Validates connection with ping                     │
│  └── Returns: OpenSearch client object                  │
│                                                         │
│  ensure_embedding_mapping(client)                       │
│  ├── Checks if 'embedding' field exists in index        │
│  └── Adds dense_vector field (512 dims) if missing      │
│                                                         │
│  save_embeddings_to_opensearch(client, docs)            │
│  ├── Takes docs with 'new_embedding' field              │
│  ├── Bulk updates documents with embeddings             │
│  └── Enables caching for future runs                    │
│                                                         │
│  save_trending_topics(client, topics)                   │
│  ├── Creates 'trending-topics' index if not exists      │
│  ├── Updates index mapping if fields missing            │
│  └── Bulk inserts topic documents                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3. `database/mysql.py`
**Role**: MySQL operations for worker service

```
┌─────────────────────────────────────────────────────────┐
│                     mysql.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  get_db_connection()                                    │
│  ├── Creates MySQL connection                           │
│  ├── Retries on failure (every 5 seconds)               │
│  └── Returns: pymysql connection object                 │
│                                                         │
│  fetch_active_inputs()                                  │
│  ├── Queries: SELECT * FROM user_tracking_inputs        │
│  │            WHERE active = 1                          │
│  └── Returns: List of active tracking configurations    │
│                                                         │
│  normalize_source_ids(raw_accounts)                     │
│  ├── Input: JSON string or list                         │
│  └── Output: Python list of account IDs                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 4. `processing/text.py`
**Role**: Text cleaning and timestamp parsing

```
┌─────────────────────────────────────────────────────────┐
│                      text.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  clean_text(text)                                       │
│  ├── Input: "مرحبا @user https://t.co/xxx #topic"       │
│  │                                                      │
│  ├── Step 1: Remove URLs (http://, www.)                │
│  ├── Step 2: Remove @mentions                           │
│  ├── Step 3: Remove #hashtags                           │
│  ├── Step 4: Remove emojis                              │
│  ├── Step 5: Keep only Arabic letters + numbers         │
│  ├── Step 6: Remove Arabic stopwords (و، في، من...)     │
│  ├── Step 7: Remove words < 3 characters                │
│  │                                                      │
│  └── Output: "مرحبا" (cleaned text)                     │
│                                                         │
│  parse_datetime_to_epoch_millis(value)                  │
│  ├── Handles: epoch millis, ISO strings, "Z" suffix     │
│  └── Returns: Unix timestamp in milliseconds            │
│                                                         │
│  get_post_timestamp_millis(post)                        │
│  ├── Tries fields: timestamp, created_at, post_created  │
│  └── Returns: Best available timestamp                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 5. `processing/embeddings.py`
**Role**: ML pipeline - embeddings, PCA, clustering

```
┌─────────────────────────────────────────────────────────┐
│                   embeddings.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  class EmbeddingProcessor:                              │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ load_model()                                    │    │
│  │ • Loads: distiluse-base-multilingual-cased-v2   │    │
│  │ • Output: 512-dimensional vectors               │    │
│  │ • Lazy loading (only when needed)               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ create_embeddings(docs)                         │    │
│  │ • Checks for cached embeddings first            │    │
│  │ • Only generates for posts without cache        │    │
│  │ • Marks new embeddings for saving               │    │
│  │ • Returns: numpy array (N x 512)                │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ reduce_dimensionality(embeddings)               │    │
│  │ • Algorithm: PCA                                │    │
│  │ • Reduces: 512 dims → 100 dims                  │    │
│  │ • Speeds up clustering                          │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ cluster_documents(embeddings)                   │    │
│  │ • Algorithm: HDBSCAN                            │    │
│  │ • min_cluster_size: 5                           │    │
│  │ • min_samples: 3                                │    │
│  │ • Returns: cluster labels (-1 = noise)          │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 6. `services/posts.py`
**Role**: Fetch posts from OpenSearch with filters

```
┌─────────────────────────────────────────────────────────┐
│                     posts.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  fetch_posts(client, user_input_id, source_ids)         │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Build Query                                  │    │
│  │    • Optional filter: user_input_id             │    │
│  │    • Optional filter: source_ids (accounts)     │    │
│  │    • Include cached embeddings if enabled       │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 2. Scan Index                                   │    │
│  │    • Uses scroll API for large datasets         │    │
│  │    • Handles circuit breaker errors (429)       │    │
│  │    • Retries with smaller batch size            │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 3. Process Each Document                        │    │
│  │    • Extract text (post_text/text/content)      │    │
│  │    • Skip if text < 10 chars                    │    │
│  │    • Clean text                                 │    │
│  │    • Load cached embedding if available         │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  Returns: List[Dict] with keys:                         │
│  • id, text, cleaned, author                            │
│  • likes, retweets, replies                             │
│  • cached_embedding (numpy array or None)               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 7. `services/topics.py`
**Role**: Generate topic labels using DeepSeek AI

```
┌─────────────────────────────────────────────────────────┐
│                     topics.py                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  generate_cluster_topic(posts_text)                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Build Arabic Prompt                          │    │
│  │    "أنت محلل محتوى اجتماعي محترف..."            │    │
│  │    • Rules: 10-15 words, no prefixes            │    │
│  │    • Include sample posts (max 2000 chars)      │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 2. Call DeepSeek API                            │    │
│  │    • Model: deepseek-chat                       │    │
│  │    • Temperature: 0.15 (low = consistent)       │    │
│  │    • Retry: 3 attempts with exponential backoff │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 3. Clean Response                               │    │
│  │    • Remove "الموضوع:" prefix                   │    │
│  │    • Remove numbering (1. 2. etc)               │    │
│  │    • Strip quotes                               │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  Returns: Topic string (e.g., "احتجاجات شعبية...")      │
│           or "غير قابل للتحديد" if failed               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 8. `services/analysis.py`
**Role**: Analyze clusters and identify trending topics

```
┌─────────────────────────────────────────────────────────┐
│                    analysis.py                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  analyze_trending_topics(docs, labels, embeddings...)   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Group by Cluster                             │    │
│  │    • Skip noise (label = -1)                    │    │
│  │    • Calculate total engagement per cluster     │    │
│  │    • engagement = likes + retweets*2 + replies*1.5│   │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 2. Filter Small Clusters                        │    │
│  │    • Remove clusters < MIN_CLUSTER_SIZE (5)     │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 3. Merge if Too Many Clusters                   │    │
│  │    • If clusters > MAX_TOPICS (20)              │    │
│  │    • Use KMeans on cluster centroids            │    │
│  │    • Merge similar clusters together            │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 4. For Each Cluster                             │    │
│  │    • Find 5 representative posts (near centroid)│    │
│  │    • Generate topic via DeepSeek                │    │
│  │    • Calculate trending_score                   │    │
│  │    • Extract top 5 keywords                     │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 5. Score Formula                                │    │
│  │    trending_score =                             │    │
│  │      size * 0.4 +                               │    │
│  │      min(engagement/100, 10) * 0.4 +            │    │
│  │      10 * 0.2                                   │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  Returns: List of topics sorted by trending_score       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 9. `jobs/tracker.py`
**Role**: In-memory job state management

```
┌─────────────────────────────────────────────────────────┐
│                    tracker.py                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Storage: jobs = {}  (in-memory dictionary)             │
│  Thread-safe: uses threading.Lock()                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Job Structure                                   │    │
│  │ {                                               │    │
│  │   "job_id": "abc123...",                        │    │
│  │   "status": "pending|processing|completed|failed"│   │
│  │   "user_input_id": "123",                       │    │
│  │   "source_ids": ["acc1", "acc2"],               │    │
│  │   "progress": 0-100,                            │    │
│  │   "result": {...} or None,                      │    │
│  │   "error": "..." or None,                       │    │
│  │   "created_at": "ISO timestamp",                │    │
│  │   "finished_at": "ISO timestamp" or None        │    │
│  │ }                                               │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Functions:                                             │
│  • generate_job_id(filters) → MD5 hash                  │
│  • create_job(job_id, ...) → initializes job            │
│  • get_job_status(job_id) → returns job dict            │
│  • update_job_status(job_id, status, progress, ...)     │
│  • clear_job(job_id) → removes from tracker             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 10. `output/formatters.py`
**Role**: Format and export results

```
┌─────────────────────────────────────────────────────────┐
│                   formatters.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  print_trending_topics(topics)                          │
│  ├── Prints formatted table to console                  │
│  └── Shows: rank, topic, posts, engagement, score       │
│                                                         │
│  save_topics_to_json(topics, filename)                  │
│  ├── Saves to: trending_topics_YYYYMMDD_HHMMSS.json     │
│  └── Returns: filename                                  │
│                                                         │
│  format_topics_summary(topics)                          │
│  └── Returns: formatted string for logging              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Main Entry Points

### 11. `trending.py` - Standalone Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    trending.py                          │
│              (Standalone Execution)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  run_trending_topics_pipeline(                          │
│      user_input_id=None,                                │
│      source_ids=None,                                   │
│      save_to_index=True,                                │
│      save_json=True,                                    │
│      print_results=True                                 │
│  )                                                      │
│                                                         │
│  Usage: python -m src.trending                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 12. `worker.py` - Background Service

```
┌─────────────────────────────────────────────────────────┐
│                     worker.py                           │
│              (Background Service)                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  main()                                                 │
│  ├── Validates config (OpenSearch + MySQL)              │
│  └── Loops forever:                                     │
│      ├── process_active_inputs()                        │
│      └── sleep(ACTIVE_INPUT_POLL_SECONDS)               │
│                                                         │
│  process_active_inputs()                                │
│  ├── Fetches active rows from MySQL                     │
│  └── For each row: process_trending_topics_job()        │
│                                                         │
│  process_trending_topics_job(job_id, filters...)        │
│  ├── Updates progress: 10% → 20% → ... → 100%           │
│  └── Runs full pipeline with filters                    │
│                                                         │
│  Usage: python -m src.worker                            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

```
                            ┌─────────────────┐
                            │   MySQL         │
                            │ (active inputs) │
                            └────────┬────────┘
                                     │ fetch_active_inputs()
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           WORKER SERVICE                                │
│                           (worker.py)                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  For each active input:                                         │    │
│  │  1. Generate job_id from filters                                │    │
│  │  2. Skip if job already processing                              │    │
│  │  3. Create job entry                                            │    │
│  │  4. Run pipeline ↓                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRENDING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐      │
│   │  OpenSearch  │───▶│  posts.py    │───▶│  List of Posts       │      │
│   │  (raw posts) │    │  fetch_posts │    │  with cached embeds  │      │
│   └──────────────┘    └──────────────┘    └──────────┬───────────┘      │
│                                                      │                  │
│                                                      ▼                  │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                    embeddings.py                             │      │
│   │  ┌────────────────┐  ┌─────────────┐  ┌─────────────────┐    │      │
│   │  │ create_embed() │─▶│ PCA reduce  │─▶│ HDBSCAN cluster │    │      │
│   │  │ (512-dim)      │  │ (100-dim)   │  │ (labels)        │    │      │
│   │  └────────────────┘  └─────────────┘  └────────┬────────┘    │      │
│   └────────────────────────────────────────────────│─────────────┘      │
│                                                    │                    │
│                                                    ▼                    │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                     analysis.py                              │      │
│   │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐      │      │
│   │  │ Group by    │─▶│ Merge if    │─▶│ Score & rank     │      │      │
│   │  │ cluster     │  │ >20 clusters│  │ topics           │      │      │
│   │  └─────────────┘  └─────────────┘  └────────┬─────────┘      │      │
│   └─────────────────────────────────────────────│────────────────┘      │
│                                                 │                       │
│                          ┌──────────────────────┘                       │
│                          │                                              │
│                          ▼                                              │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                     topics.py                                │      │
│   │  For each cluster:                                           │      │
│   │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │      │
│   │  │ Sample 10   │───▶│ DeepSeek AI │───▶│ Topic label     │   │      │
│   │  │ posts       │    │ API call    │    │ (Arabic)        │   │      │
│   │  └─────────────┘    └─────────────┘    └─────────────────┘   │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│   │  OpenSearch      │  │  JSON File       │  │  Console         │      │
│   │  trending-topics │  │  trending_*.json │  │  print output    │      │
│   │  index           │  │                  │  │                  │      │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
                    settings.py
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    opensearch.py    mysql.py       text.py
         │               │               │
         │               │               ▼
         │               │         embeddings.py
         │               │               │
         ▼               │               ▼
      posts.py ◀─────────┘          topics.py
         │                              │
         └──────────┬───────────────────┘
                    ▼
              analysis.py
                    │
                    ▼
              formatters.py
                    │
         ┌─────────┴─────────┐
         ▼                   ▼
    trending.py          worker.py
         │                   │
         │                   ▼
         │             tracker.py
         │
         ▼
    [Standalone]        [Service]
```

---

## Quick Reference

| Module | Purpose | Key Function |
|--------|---------|--------------|
| `settings.py` | Configuration | `validate_config()` |
| `opensearch.py` | Database ops | `create_opensearch_client()` |
| `mysql.py` | MySQL queries | `fetch_active_inputs()` |
| `text.py` | Text cleaning | `clean_text()` |
| `embeddings.py` | ML pipeline | `EmbeddingProcessor` class |
| `posts.py` | Fetch data | `fetch_posts()` |
| `topics.py` | AI generation | `generate_cluster_topic()` |
| `analysis.py` | Trending logic | `analyze_trending_topics()` |
| `tracker.py` | Job state | `update_job_status()` |
| `formatters.py` | Output | `print_trending_topics()` |
| `trending.py` | Main pipeline | `run_trending_topics_pipeline()` |
| `worker.py` | Background service | `main()` |

---

## Running the Code

```bash
# Standalone (process all posts)
python -m src.trending

# Worker service (polls MySQL)
python -m src.worker

# With systemd
systemctl start trending-topics-worker
```
