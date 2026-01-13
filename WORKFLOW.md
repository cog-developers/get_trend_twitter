# How The System Works Now

## 📊 Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Your Existing Data: user-input-posts Index                 │
│  (Already populated with posts - NO SCRAPING NEEDED)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: Fetch Posts                                        │
│  • Reads from "user-input-posts" index                      │
│  • Extracts: post_text/text/content, likes, retweets, etc.  │
│  • Filters meaningful posts (min 10 chars)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: Text Processing                                    │
│  • Cleans Arabic text (removes URLs, mentions, emojis)      │
│  • Normalizes text (removes stopwords)                      │
│  • Prepares for embedding                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: Create Embeddings                                  │
│  • Uses multilingual sentence transformer model             │
│  • Converts each post to a 512-dimensional vector           │
│  • Normalizes embeddings                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: Dimensionality Reduction (PCA)                    │
│  • Reduces from 512D to 100D (faster clustering)           │
│  • Preserves ~85-90% of variance                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: Clustering (HDBSCAN)                              │
│  • Groups similar posts together                             │
│  • Finds clusters automatically (no need to specify count)  │
│  • Filters clusters with < 5 posts                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: Generate Topics for Each Cluster                  │
│  • Selects representative posts from each cluster           │
│  • Sends to DeepSeek API to generate Arabic topic          │
│  • Cleans and validates topic text                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: Calculate Trending Scores                         │
│  • Cluster size (40% weight)                                │
│  • Engagement score (40% weight): likes + retweets*2 + ...  │
│  • Base score (20% weight)                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: Save & Export Results                             │
│  • Saves to "trending-topics-index" in OpenSearch          │
│  • Exports JSON file: trending_topics_YYYYMMDD_HHMMSS.json │
│  • Prints top 20 trending topics to console                │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Step-by-Step Execution

### When You Run: `python get_trending_topics.py`

#### **Phase 1: Data Collection** (30 seconds - 2 minutes)
```
1. Connects to OpenSearch/Elasticsearch
2. Scans "user-input-posts" index
3. Fetches all documents with:
   - post_text/text/content (required)
   - likes, retweets, replies (optional but helpful)
   - author, created_at (optional)
4. Filters out empty/invalid posts
5. Result: List of valid posts ready for processing
```

**Example:**
- Input: 10,000 posts in your index
- Output: ~8,500 valid posts (after filtering)

---

#### **Phase 2: Text Processing** (1-3 minutes)
```
1. For each post:
   - Removes URLs, mentions (@username), hashtags
   - Removes emojis
   - Normalizes Arabic text
   - Removes stopwords (و، في، من، etc.)
   - Keeps only meaningful words (>2 chars)
2. Result: Cleaned text ready for embedding
```

**Example:**
- Input: "مرحبا @user هذا منشور رائع! 😊 https://example.com"
- Output: "مرحبا منشور رائع"

---

#### **Phase 3: Embedding Creation** (5-15 minutes)
```
1. Loads multilingual sentence transformer model
   (distiluse-base-multilingual-cased-v2)
2. Converts each cleaned post to a 512-dimensional vector
3. Normalizes vectors (unit length)
4. Result: Array of embeddings [num_posts × 512]
```

**Example:**
- Input: 8,500 cleaned posts
- Output: NumPy array shape (8500, 512)
- Each post is now a point in 512-dimensional space

---

#### **Phase 4: Dimensionality Reduction** (1-2 minutes)
```
1. Applies PCA (Principal Component Analysis)
2. Reduces from 512D → 100D
3. Preserves ~85-90% of information
4. Makes clustering faster and more efficient
5. Result: Reduced embeddings [num_posts × 100]
```

**Example:**
- Input: (8500, 512)
- Output: (8500, 100)
- Speed improvement: ~5x faster clustering

---

#### **Phase 5: Clustering** (2-5 minutes)
```
1. Runs HDBSCAN clustering algorithm
2. Automatically finds clusters (no need to specify count)
3. Groups similar posts together
4. Marks outliers as "noise" (cluster -1)
5. Filters clusters with < 5 posts
6. Result: Each post assigned to a cluster ID
```

**Example:**
- Input: 8,500 posts
- Output: 
  - 50 clusters (with 5+ posts each)
  - 500 noise points (outliers)
  - Cluster 0: 120 posts about "politics"
  - Cluster 1: 85 posts about "sports"
  - etc.

---

#### **Phase 6: Topic Generation** (10-30 minutes)
```
For each cluster:
1. Selects top 5-10 most representative posts
2. Sends to DeepSeek API with prompt:
   "Generate one clear Arabic topic for these posts"
3. Cleans and validates the generated topic
4. Extracts top 5 keywords from cluster
5. Result: Topic label for each cluster
```

**Example:**
- Cluster 0 (120 posts) → Topic: "المناقشات السياسية حول الانتخابات"
- Cluster 1 (85 posts) → Topic: "أخبار كرة القدم والمباريات"
- etc.

---

#### **Phase 7: Trending Score Calculation** (1 minute)
```
For each cluster:
1. Calculates engagement score:
   engagement = likes + (retweets × 2) + (replies × 1.5)
2. Calculates trending score:
   score = (cluster_size × 0.4) + (engagement/100 × 0.4) + (10 × 0.2)
3. Sorts clusters by trending score (highest first)
4. Result: Ranked list of trending topics
```

**Example:**
- Cluster 0: 120 posts, 5000 engagement → Score: 48.0
- Cluster 1: 85 posts, 3000 engagement → Score: 34.0
- Cluster 2: 200 posts, 2000 engagement → Score: 88.0 (trending!)

---

#### **Phase 8: Save Results** (1-2 minutes)
```
1. Creates/updates "trending-topics-index" in OpenSearch
2. Saves each trending topic with:
   - topic (Arabic text)
   - rank (1 = most trending)
   - post_count
   - engagement_score
   - trending_score
   - keywords
   - representative_texts
   - member_ids (post IDs in cluster)
3. Exports JSON file locally
4. Prints top 20 to console
```

**Example Output:**
```json
[
  {
    "topic": "المناقشات السياسية حول الانتخابات",
    "rank": 1,
    "post_count": 200,
    "engagement_score": 5000,
    "trending_score": 88.0,
    "keywords": ["انتخابات", "سياسة", "حكومة"],
    "representative_texts": ["...", "...", "..."],
    "member_ids": ["post_1", "post_2", ...]
  },
  ...
]
```

---

## 📈 Real-World Example

### Scenario: You have 5,000 posts in `user-input-posts`

**Timeline:**
```
00:00 - Start script
00:30 - Fetched 4,800 valid posts
02:00 - Created embeddings (4,800 × 512)
03:00 - Reduced to 100 dimensions
05:00 - Found 35 clusters
15:00 - Generated topics for all clusters
16:00 - Calculated trending scores
17:00 - Saved results
17:00 - ✅ Complete!
```

**Results:**
- **Top Trending Topic**: "المناقشات حول الأزمة الاقتصادية"
  - 180 posts
  - Engagement: 8,500
  - Score: 74.0

- **Second Trending**: "أخبار الرياضة والمباريات"
  - 120 posts
  - Engagement: 6,200
  - Score: 64.8

- ... and 33 more topics

---

## 🎯 Key Features

### ✅ **No Scraping Required**
- Works directly with your existing `user-input-posts` index
- Just run the script and it processes what's already there

### ✅ **Automatic Clustering**
- No need to specify number of topics
- HDBSCAN finds clusters automatically
- Handles any number of topics

### ✅ **Smart Trending Detection**
- Not just based on count
- Considers engagement (likes, retweets, replies)
- Ranks by actual trending score

### ✅ **Arabic-Optimized**
- Specialized Arabic text cleaning
- Multilingual embeddings
- Arabic topic generation

### ✅ **Flexible Field Names**
- Works with `post_text`, `text`, or `content`
- Handles missing engagement metrics gracefully
- Adapts to your data structure

---

## 🔧 Configuration Options

You can adjust these in `.env`:

```env
# Minimum cluster size (smaller = more topics)
MIN_CLUSTER_SIZE=5

# Clustering sensitivity (lower = more clusters)
HDBSCAN_MIN_CLUSTER_SIZE=5
HDBSCAN_MIN_SAMPLES=3

# Processing speed
EMBEDDING_BATCH_SIZE=32  # Higher = faster but more memory
MAX_WORKERS=5            # Parallel processing
```

---

## 📊 Output Locations

1. **Console**: Top 20 trending topics printed
2. **OpenSearch**: All topics saved to `trending-topics-index`
3. **JSON File**: `trending_topics_YYYYMMDD_HHMMSS.json` in project folder

---

## 🚀 Quick Start

```bash
# 1. Make sure your .env is configured
# 2. Ensure posts are in "user-input-posts" index
# 3. Run:
python get_trending_topics.py

# 4. Check results:
#    - Console output
#    - trending_topics_*.json file
#    - OpenSearch: trending-topics-index
```

That's it! The system handles everything automatically. 🎉

