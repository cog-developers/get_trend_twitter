# Community Sections - Code Flow Documentation

## Overview

This module classifies Twitter users into Lebanese political parties using DeepSeek AI.

---

## Directory Structure

```
communitysections/
├── config/
│   └── settings.py        # Configuration, env vars, Lebanese parties prompt
├── database/
│   └── opensearch.py      # OpenSearch client, index operations
├── services/
│   ├── users.py           # Fetch users, enrich with bio/tweets
│   └── classifier.py      # DeepSeek API classification
├── output/
│   └── formatters.py      # Print summary & results
└── main.py                # Main orchestration
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
│  • DEEPSEEK_API_KEY                                     │
│  • SOURCE_INDEX (tweets), TARGET_INDEX (classified)     │
│  • MAX_WORKERS, MAX_USERS, BATCH_SIZE                   │
│                                                         │
│  Constants:                                             │
│  • LEBANESE_PARTIES_PROMPT (15 political parties)       │
│  • RETRY_ATTEMPTS, RATE_LIMIT_DELAY                     │
│                                                         │
│  Functions:                                             │
│  • validate_config() → checks DEEPSEEK_API_KEY          │
│  • get_logger(name) → returns configured logger         │
└─────────────────────────────────────────────────────────┘
```

**Lebanese Parties Covered:**
- Christian: FPM, Lebanese Forces, Kataeb, Marada
- Shia: Hezbollah, Amal Movement
- Sunni: Future Movement
- Druze: PSP
- Independent/Opposition: MMFD, LCP, October 17, Sabaa
- Other: Media/Journalist, Independent, Unknown

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
│  └── Returns: OpenSearch client object                  │
│                                                         │
│  create_classified_index(force_recreate)                │
│  ├── Creates 'classified-users-index' if not exists     │
│  ├── Defines mapping: user_id, party, confidence, etc.  │
│  └── Optional: delete and recreate                      │
│                                                         │
│  save_classifications_bulk(classified_users)            │
│  ├── Bulk inserts classification results                │
│  └── Uses user_id as document ID (upsert)               │
│                                                         │
│  es_call_with_retries(fn, *args)                        │
│  └── Exponential backoff retry for network errors       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 3. `services/users.py`
**Role**: Fetch and enrich user data

```
┌─────────────────────────────────────────────────────────┐
│                     users.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  fetch_all_users(limit)                                 │
│  ├── Primary: Uses composite aggregation                │
│  ├── Fallback: Uses scroll API                          │
│  └── Returns: List of unique users with doc_count       │
│                                                         │
│  fetch_all_users_scroll(limit)                          │
│  ├── Scrolls through all documents                      │
│  ├── Extracts user_id, username, name                   │
│  └── Counts documents per user                          │
│                                                         │
│  enrich_user_with_details(user_id)  [CACHED]            │
│  ├── Fetches user's bio from user_details               │
│  ├── Fetches 10 most recent tweets                      │
│  ├── LRU cache (maxsize=2000)                           │
│  └── Returns: (bio, tweets_text)                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 4. `services/classifier.py`
**Role**: DeepSeek AI classification

```
┌─────────────────────────────────────────────────────────┐
│                   classifier.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  classify_with_deepseek(name, username, bio, tweets)    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 1. Build Classification Prompt                  │    │
│  │    • User profile (name, username, bio)         │    │
│  │    • Recent tweets (sample)                     │    │
│  │    • Lebanese parties reference list            │    │
│  │    • Instructions for analysis                  │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 2. Call DeepSeek API                            │    │
│  │    • Model: deepseek-chat                       │    │
│  │    • Temperature: 0.2 (consistent)              │    │
│  │    • Retry on 429 (rate limit)                  │    │
│  │    • Timeout: 60s                               │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 3. Parse JSON Response                          │    │
│  │    • Extract: party, confidence, reasoning      │    │
│  │    • Handle code fences, trailing commas        │    │
│  └─────────────────────────────────────────────────┘    │
│                       ↓                                 │
│  Returns: (party, confidence, reasoning, response_ms)   │
│                                                         │
│  classify_user(user, run_id)                            │
│  ├── Enriches user with bio/tweets                      │
│  ├── Calls classify_with_deepseek                       │
│  └── Returns: document ready for indexing               │
│                                                         │
│  classify_users_parallel(users, run_id)                 │
│  ├── ThreadPoolExecutor (MAX_WORKERS threads)           │
│  ├── Progress bar with tqdm                             │
│  ├── Jitter delay between requests                      │
│  └── Fallback on errors (marks as "Unknown")            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 5. `output/formatters.py`
**Role**: Format and display results

```
┌─────────────────────────────────────────────────────────┐
│                   formatters.py                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  print_classification_summary(classified_users)         │
│  ├── Party distribution with percentages & bar chart   │
│  ├── Confidence level breakdown                         │
│  ├── Average API response times                         │
│  └── Sample classifications per party                   │
│                                                         │
│  print_classification_results(classified_users)         │
│  └── Detailed per-user results                          │
│                                                         │
│  format_party_distribution(classified_users)            │
│  └── Returns formatted string for logging               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

### 6. `main.py`
**Role**: Main orchestration

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  run_classification_pipeline(                           │
│      max_users=0,                                       │
│      force_recreate_index=False,                        │
│      print_results=True                                 │
│  )                                                      │
│                                                         │
│  Pipeline Steps:                                        │
│  1. Generate run_id (timestamp-based)                   │
│  2. Validate configuration                              │
│  3. Initialize OpenSearch client                        │
│  4. Validate source index mapping                       │
│  5. Create target index (if needed)                     │
│  6. Fetch users from source index                       │
│  7. Filter invalid entries                              │
│  8. Classify users in parallel                          │
│  9. Save results to OpenSearch                          │
│  10. Print summary & results                            │
│  11. Return statistics                                  │
│                                                         │
│  Usage: python -m communitysections.main                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLASSIFICATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────────┐                                                  │
│   │  OpenSearch      │                                                  │
│   │  SOURCE_INDEX    │                                                  │
│   │  (tweets)        │                                                  │
│   └────────┬─────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                     users.py                                 │      │
│   │  ┌────────────────┐  ┌─────────────────────────────────┐     │      │
│   │  │ fetch_all_users│─▶│ List of unique users            │     │      │
│   │  │ (aggregation)  │  │ {user_id, username, name, count}│     │      │
│   │  └────────────────┘  └─────────────────────────────────┘     │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                            │                                            │
│                            ▼                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                   classifier.py                              │      │
│   │                                                              │      │
│   │   For each user (parallel, MAX_WORKERS threads):             │      │
│   │   ┌─────────────────┐                                        │      │
│   │   │ enrich_user     │─▶ bio + recent tweets                  │      │
│   │   └────────┬────────┘                                        │      │
│   │            │                                                 │      │
│   │            ▼                                                 │      │
│   │   ┌─────────────────┐    ┌─────────────────┐                 │      │
│   │   │ Build prompt    │───▶│ DeepSeek API    │                 │      │
│   │   │ with parties    │    │ (deepseek-chat) │                 │      │
│   │   └─────────────────┘    └────────┬────────┘                 │      │
│   │                                   │                          │      │
│   │                                   ▼                          │      │
│   │   ┌─────────────────────────────────────────────────────┐    │      │
│   │   │ JSON Response:                                      │    │      │
│   │   │ {                                                   │    │      │
│   │   │   "party": "Hezbollah",                             │    │      │
│   │   │   "confidence": "high",                             │    │      │
│   │   │   "reasoning": "User frequently supports..."        │    │      │
│   │   │ }                                                   │    │      │
│   │   └─────────────────────────────────────────────────────┘    │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                            │                                            │
│                            ▼                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                   opensearch.py                              │      │
│   │  ┌─────────────────────┐                                     │      │
│   │  │ save_classifications│─▶ TARGET_INDEX                      │      │
│   │  │ _bulk()             │   (classified-users-index)          │      │
│   │  └─────────────────────┘                                     │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                            │                                            │
│                            ▼                                            │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │                   formatters.py                              │      │
│   │  ┌─────────────────────┐  ┌─────────────────────┐            │      │
│   │  │ print_summary()     │  │ print_results()     │            │      │
│   │  │ - Party breakdown   │  │ - Per-user details  │            │      │
│   │  │ - Confidence stats  │  │ - Reasoning         │            │      │
│   │  └─────────────────────┘  └─────────────────────┘            │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependencies

```
                    settings.py
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
    opensearch.py                    users.py
         │                               │
         │                               ▼
         │                         classifier.py
         │                               │
         └───────────────┬───────────────┘
                         ▼
                   formatters.py
                         │
                         ▼
                      main.py
```

---

## Output Schema (TARGET_INDEX)

```json
{
  "user_id": "123456789",
  "username": "example_user",
  "name": "Example User",
  "party": "Lebanese Forces",
  "confidence": "high",
  "reasoning": "User bio mentions LF, frequently retweets Geagea...",
  "tweet_count": 45,
  "user_bio": "Proud Lebanese | LF supporter",
  "sample_tweets": "Tweet 1 | Tweet 2 | Tweet 3...",
  "classified_at": "2024-01-15T10:30:00Z",
  "classification_run_id": "run_20240115_103000",
  "api_response_time_ms": 1250
}
```

---

## Quick Reference

| Module | Purpose | Key Function |
|--------|---------|--------------|
| `settings.py` | Configuration | `validate_config()` |
| `opensearch.py` | Database ops | `save_classifications_bulk()` |
| `users.py` | User fetching | `fetch_all_users()`, `enrich_user_with_details()` |
| `classifier.py` | AI classification | `classify_users_parallel()` |
| `formatters.py` | Output | `print_classification_summary()` |
| `main.py` | Orchestration | `run_classification_pipeline()` |

---

## Running the Code

```bash
# Run classification pipeline
python -m communitysections.main

# Environment variables needed:
# OPENSEARCH_NODE, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD
# DEEPSEEK_API_KEY
# SOURCE_INDEX (default: past-month-tweets-index)
# TARGET_INDEX (default: classified-users-index)
# MAX_USERS (default: 0 = all)
# MAX_WORKERS (default: 10)
```
