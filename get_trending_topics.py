"""
Trending Topics Generator
Reads posts from 'user-input-posts' index, generates topics, clusters them,
and identifies trending topics based on frequency and engagement.
"""

import os
import re
import time
import json
from datetime import datetime, timezone
from collections import Counter
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import hdbscan
import urllib3
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


# ====== CONFIG ======
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "user-input-posts")
TRENDING_INDEX = os.getenv("TRENDING_INDEX", "trending-topics")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Processing config
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "5"))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "5"))
HDBSCAN_MIN_SAMPLES = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))
PCA_TARGET_DIM = int(os.getenv("PCA_TARGET_DIM", "100"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MAX_TOPICS = int(os.getenv("MAX_TOPICS", "20"))
EMBEDDING_DIM = 512  # distiluse-base-multilingual-cased-v2 output dimension

ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "إلى", "مع", "عن", "ما", "لا",
    "هذا", "هذه", "هو", "هي", "أن", "إن", "كان", "كل", "قد",
    "لم", "لن", "ثم", "أو", "بل", "لكن", "حتى", "عند", "بعد", "قبل"
}

# ====== VALIDATION ======
def validate_config():
    """Validate required environment variables."""
    required_vars = {
        "OPENSEARCH_NODE": OPENSEARCH_NODE,
        "OPENSEARCH_USERNAME": OPENSEARCH_USERNAME,
        "OPENSEARCH_PASSWORD": OPENSEARCH_PASSWORD,
        "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    logger.info("✅ Configuration validated")

# ====== OPENSEARCH CLIENT ======
def create_opensearch_client() -> OpenSearch:
    """Create OpenSearch client."""
    try:
        client = OpenSearch(
            [OPENSEARCH_NODE],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=90,
            max_retries=3,
            retry_on_timeout=True
        )
        if not client.ping():
            raise ConnectionError("Cannot ping OpenSearch")
        logger.info("✅ Connected to OpenSearch")
        return client
    except Exception as e:
        logger.error(f"❌ OpenSearch connection failed: {e}")
        raise


def ensure_embedding_mapping(client: OpenSearch):
    """Ensure the embedding field exists in the source index mapping."""
    try:
        mapping = client.indices.get_mapping(index=SOURCE_INDEX)
        properties = mapping.get(SOURCE_INDEX, {}).get("mappings", {}).get("properties", {})

        if "embedding" not in properties:
            logger.info(f"📝 Adding 'embedding' field to index {SOURCE_INDEX}...")
            update_mapping = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": False  # We don't need KNN search, just storage
                    },
                    "embedding_updated_at": {
                        "type": "date"
                    }
                }
            }
            client.indices.put_mapping(index=SOURCE_INDEX, body=update_mapping)
            logger.info("✅ Added embedding field to index mapping")
        else:
            logger.info("ℹ️ Embedding field already exists in index mapping")
    except Exception as e:
        logger.warning(f"⚠️ Could not update mapping for embeddings: {e}")


def save_embeddings_to_opensearch(client: OpenSearch, docs_with_embeddings: List[Dict]):
    """Save computed embeddings back to OpenSearch for caching."""
    if not docs_with_embeddings:
        return

    logger.info(f"💾 Saving {len(docs_with_embeddings)} embeddings to OpenSearch...")

    actions = []
    for doc in docs_with_embeddings:
        if "new_embedding" in doc and doc["new_embedding"] is not None:
            actions.append({
                "_op_type": "update",
                "_index": SOURCE_INDEX,
                "_id": doc["id"],
                "doc": {
                    "embedding": doc["new_embedding"].tolist() if hasattr(doc["new_embedding"], 'tolist') else doc["new_embedding"],
                    "embedding_updated_at": datetime.now(timezone.utc).isoformat()
                }
            })

    if not actions:
        logger.info("ℹ️ No new embeddings to save")
        return

    try:
        success, errors = helpers.bulk(
            client,
            actions,
            chunk_size=100,
            request_timeout=120,
            raise_on_error=False
        )
        logger.info(f"✅ Saved {success} embeddings to OpenSearch")
        if errors:
            logger.warning(f"⚠️ {len(errors)} errors while saving embeddings")
    except Exception as e:
        logger.error(f"❌ Error saving embeddings: {e}")

# ====== TEXT PROCESSING ======
def clean_text(text: str) -> str:
    """Clean and normalize Arabic text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\u2600-\u26FF]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)
    
    # Keep only Arabic letters, numbers, and spaces
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords and short words
    words = [w for w in text.split() if len(w) > 2 and w not in ARABIC_STOPWORDS]
    
    return " ".join(words)

def _parse_any_datetime_to_epoch_millis(value) -> Optional[int]:
    """
    Best-effort parse for OpenSearch date-ish values.

    Supports:
    - epoch millis (int/float)
    - ISO strings with/without 'Z'
    - datetime-like strings that datetime.fromisoformat can parse
    Returns epoch millis (UTC) or None.
    """
    if value is None:
        return None

    # epoch millis
    if isinstance(value, (int, float)):
        # If it looks like seconds, convert to millis (heuristic)
        if value < 10_000_000_000:  # < ~2286-11-20 in seconds
            return int(value * 1000)
        return int(value)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Handle "Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    return None

def _get_post_timestamp_millis(post: Dict) -> Optional[int]:
    """Extract the best available post timestamp as epoch millis."""
    for field in ("timestamp", "created_at", "post_created_at"):
        ts = _parse_any_datetime_to_epoch_millis(post.get(field))
        if ts is not None:
            return ts
    return None

# ====== FETCH POSTS ======
def fetch_posts(client: OpenSearch) -> List[Dict]:
    """Fetch posts from user-input-posts index, including cached embeddings."""
    logger.info(f"📥 Fetching posts from index: {SOURCE_INDEX}")

    docs = []
    cached_count = 0
    try:
        query = {
            "query": {"match_all": {}},
            "_source": [
                "post_text", "text", "content", "created_at", "timestamp",
                "author", "likes", "retweets", "replies",
                "embedding", "embedding_updated_at"  # Include cached embeddings
            ]
        }

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

# ====== EMBEDDING & CLUSTERING ======
class EmbeddingProcessor:
    """Handle text embeddings and clustering with caching support."""

    def __init__(self):
        self.model = None

    def load_model(self):
        """Load sentence transformer model."""
        if self.model is None:
            logger.info("⚙️ Loading sentence transformer model...")
            self.model = SentenceTransformer(
                'sentence-transformers/distiluse-base-multilingual-cased-v2'
            )
            logger.info("✅ Model loaded")
        return self.model

    def create_embeddings(self, docs: List[Dict]) -> np.ndarray:
        """
        Create embeddings for documents, using cached embeddings when available.
        Only computes new embeddings for posts without cached vectors.
        Returns the full embedding matrix and marks docs with new embeddings.
        """
        total_docs = len(docs)

        # Separate docs with and without cached embeddings
        docs_needing_embedding = []
        docs_needing_indices = []

        for i, doc in enumerate(docs):
            if doc.get("cached_embedding") is None:
                docs_needing_embedding.append(doc)
                docs_needing_indices.append(i)

        cached_count = total_docs - len(docs_needing_embedding)
        logger.info(f"🔢 Processing {total_docs} posts: {cached_count} cached, {len(docs_needing_embedding)} need embedding")

        # Initialize embeddings array
        embeddings = np.zeros((total_docs, EMBEDDING_DIM), dtype=np.float32)

        # Fill in cached embeddings
        for i, doc in enumerate(docs):
            if doc.get("cached_embedding") is not None:
                embeddings[i] = doc["cached_embedding"]

        # Generate new embeddings only for docs that need them
        if docs_needing_embedding:
            logger.info(f"🔢 Generating embeddings for {len(docs_needing_embedding)} new posts...")
            model = self.load_model()
            texts = [d["cleaned"] for d in docs_needing_embedding]

            try:
                new_embeddings = model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    normalize_embeddings=True
                )

                # Fill in new embeddings and mark docs for saving
                for j, idx in enumerate(docs_needing_indices):
                    embeddings[idx] = new_embeddings[j]
                    # Mark this doc as having a new embedding to save
                    docs[idx]["new_embedding"] = new_embeddings[j]

                logger.info(f"✅ Generated {len(docs_needing_embedding)} new embeddings")

            except Exception as e:
                logger.error(f"❌ Error creating embeddings: {e}")
                raise
        else:
            logger.info("✅ All embeddings loaded from cache - no model inference needed!")

        logger.info(f"✅ Total embeddings ready: shape={embeddings.shape}")
        return embeddings
    
    def reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        logger.info("📉 Reducing dimensionality with PCA...")
        
        try:
            from sklearn.decomposition import PCA
            n_comp = min(PCA_TARGET_DIM, embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(embeddings)
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            logger.info(
                f"✅ Reduced to {reduced.shape[1]} dims "
                f"(variance explained: {variance_explained:.2%})"
            )
            return reduced
        except Exception as e:
            logger.error(f"❌ Error in dimensionality reduction: {e}")
            raise
    
    def cluster_documents(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster documents using HDBSCAN."""
        logger.info("🎯 Clustering documents...")
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                min_samples=HDBSCAN_MIN_SAMPLES,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set([l for l in labels if l != -1]))
            noise = np.sum(labels == -1)
            
            logger.info(f"✅ Found {n_clusters} clusters, {noise} noise points")
            return labels
        except Exception as e:
            logger.error(f"❌ Clustering failed: {e}")
            raise

# ====== TOPIC GENERATION ======
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def generate_cluster_topic(posts_text: str) -> str:
    """Generate topic for a cluster of posts."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""أنت محلل محتوى اجتماعي محترف.
مهمتك: استخرج موضوعاً واحداً واضحاً وموجزاً يصف بدقة محتوى المنشورات التالية.

⚠️ القواعد:
- اكتب الموضوع مباشرة دون مقدمات
- جملة واحدة كاملة فقط (10-15 كلمة)
- لا تبدأ بـ: "الموضوع"، "المنشورات"، "يتناول"
- لا تستخدم نقاط التعليق (...)
- إذا كان المحتوى غير ذي معنى، أجب بـ: "غير قابل للتحديد"

المنشورات:
{posts_text[:2000]}

الموضوع:"""
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        topic = response.json()["choices"][0]["message"]["content"].strip()
        
        # Clean topic
        topic = re.sub(r'^الموضوعات?\s*(المستخرجة|هو|هي)?\s*:?\s*', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^[\d\.\-\s:•]+', '', topic)
        topic = topic.strip('"').strip("'").strip()
        topic = re.sub(r'\.{2,}$', '', topic)
        
        return topic if len(topic) > 10 else "غير قابل للتحديد"
    except Exception as e:
        logger.error(f"❌ Error generating topic: {e}")
        return "غير قابل للتحديد"

# ====== TRENDING TOPICS ANALYSIS ======
def analyze_trending_topics(
    docs: List[Dict],
    labels: np.ndarray,
    embeddings_reduced: np.ndarray,
    embeddings_raw: np.ndarray,
    max_topics: int = MAX_TOPICS
) -> List[Dict]:
    """
    Analyze clusters and identify trending topics.

    If more than `max_topics` clusters are found, clusters are merged down to
    `max_topics` using KMeans on cluster centroids (in reduced embedding space),
    so that each topic contains more posts.
    """
    logger.info("📊 Analyzing trending topics...")
    
    clusters = {}
    
    # Group documents by cluster
    for idx, (doc, label) in enumerate(zip(docs, labels)):
        if label == -1:  # Skip noise
            continue
        
        if label not in clusters:
            clusters[label] = {
                "posts": [],
                "indices": [],
                "total_engagement": 0
            }
        
        clusters[label]["posts"].append(doc)
        clusters[label]["indices"].append(idx)
        clusters[label]["total_engagement"] += (
            doc.get("likes", 0) + 
            doc.get("retweets", 0) * 2 + 
            doc.get("replies", 0) * 1.5
        )
    
    # Filter by minimum cluster size
    valid_clusters = {
        k: v for k, v in clusters.items() 
        if len(v["posts"]) >= MIN_CLUSTER_SIZE
    }
    
    # Merge clusters down to max_topics (if needed)
    merged_clusters = valid_clusters
    if max_topics and max_topics > 0 and len(valid_clusters) > max_topics:
        logger.info(f"🧩 Merging {len(valid_clusters)} clusters → {max_topics} topics (MAX_TOPICS)")
        cluster_ids = list(valid_clusters.keys())
        centroids = []
        for cid in cluster_ids:
            idxs = valid_clusters[cid]["indices"]
            centroids.append(embeddings_reduced[idxs].mean(axis=0))
        centroids = np.vstack(centroids)

        try:
            km = KMeans(n_clusters=max_topics, random_state=42, n_init=10)
            assignments = km.fit_predict(centroids)
        except Exception as e:
            logger.warning(f"⚠️ KMeans merge failed, falling back to top-{max_topics} clusters: {e}")
            # Fallback: keep largest clusters only
            top = sorted(
                valid_clusters.items(),
                key=lambda kv: (len(kv[1]["posts"]), kv[1]["total_engagement"]),
                reverse=True
            )[:max_topics]
            merged_clusters = {i: v for i, (_, v) in enumerate(top)}
        else:
            grouped = {}
            for cid, g in zip(cluster_ids, assignments):
                grouped.setdefault(int(g), {"posts": [], "indices": [], "total_engagement": 0})
                grouped[int(g)]["posts"].extend(valid_clusters[cid]["posts"])
                grouped[int(g)]["indices"].extend(valid_clusters[cid]["indices"])
                grouped[int(g)]["total_engagement"] += valid_clusters[cid]["total_engagement"]
            merged_clusters = grouped

    logger.info(f"📈 Analyzing {len(merged_clusters)} clusters after merge...")
    
    trending_topics = []
    
    for cluster_id, cluster_data in merged_clusters.items():
        posts = cluster_data["posts"]
        size = len(posts)
        engagement = cluster_data["total_engagement"]
        
        # Get representative texts (top 5) using reduced space for speed
        cluster_embeds_reduced = embeddings_reduced[cluster_data["indices"]]
        centroid_reduced = cluster_embeds_reduced.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeds_reduced, centroid_reduced).flatten()
        top_indices = np.argsort(-sims)[:5]
        representative_texts = [posts[i]["text"] for i in top_indices]

        # Stable centroid for incremental assignment (raw embedding space)
        cluster_embeds_raw = embeddings_raw[cluster_data["indices"]]
        centroid_raw = cluster_embeds_raw.mean(axis=0)
        norm = float(np.linalg.norm(centroid_raw))
        if norm > 0:
            centroid_raw = centroid_raw / norm
        
        # Generate topic
        sample_text = "\n---\n".join([p["text"][:300] for p in posts[:10]])
        topic = generate_cluster_topic(sample_text)
        
        # Calculate trending score (size + engagement + recency)
        trending_score = (
            size * 0.4 +  # Cluster size weight
            min(engagement / 100, 10) * 0.4 +  # Engagement weight (capped)
            10 * 0.2  # Base score
        )
        
        # Extract keywords
        all_words = []
        for post in posts:
            words = clean_text(post["text"]).split()
            all_words.extend([w for w in words if len(w) > 2])
        
        keywords = [word for word, count in Counter(all_words).most_common(5)]
        
        current_time = datetime.utcnow()
        timestamp_ms = int(current_time.timestamp() * 1000)  # Unix timestamp in milliseconds

        # Track max post timestamp in this topic (for incremental state)
        post_ts = []
        for p in posts:
            t = _get_post_timestamp_millis(p)
            if t is not None:
                post_ts.append(t)
        last_post_timestamp_ms = max(post_ts) if post_ts else None
        
        trending_topics.append({
            "topic": topic,
            "cluster_id": f"cluster_{cluster_id}",
            "post_count": size,
            "engagement_score": engagement,
            "trending_score": trending_score,
            "keywords": keywords,
            "representative_texts": representative_texts[:3],
            "member_ids": [p["id"] for p in posts],
            "centroid_embedding": centroid_raw.tolist() if hasattr(centroid_raw, "tolist") else centroid_raw,
            "last_post_timestamp": last_post_timestamp_ms,
            "generated_at": current_time.isoformat(),
            "timestamp": timestamp_ms
        })
    
    # Sort by trending score
    trending_topics.sort(key=lambda x: x["trending_score"], reverse=True)

    # Enforce max_topics again post-scoring (safety)
    if max_topics and max_topics > 0:
        trending_topics = trending_topics[:max_topics]
    
    logger.info(f"✅ Identified {len(trending_topics)} trending topics")
    return trending_topics

# ====== SAVE TO OPENSEARCH ======
def save_trending_topics(client: OpenSearch, topics: List[Dict]):
    """Save trending topics to OpenSearch."""
    logger.info(f"💾 Saving {len(topics)} trending topics to {TRENDING_INDEX}...")
    
    # Create index if it doesn't exist, or update mapping if it exists
    if not client.indices.exists(index=TRENDING_INDEX):
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
            "mappings": {
                "properties": {
                    "topic": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "cluster_id": {"type": "keyword"},
                    "post_count": {"type": "integer"},
                    "engagement_score": {"type": "float"},
                    "trending_score": {"type": "float"},
                    "keywords": {"type": "keyword"},
                    "representative_texts": {"type": "text"},
                    "member_ids": {"type": "keyword"},
                    "centroid_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIM,
                        "index": False
                    },
                    "generated_at": {"type": "date"},
                    "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "last_post_timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "rank": {"type": "integer"},
                    "user_input_id": {"type": "keyword"},
                    "filtered_sources": {"type": "keyword"},
                    "filter_key": {"type": "keyword"}
                }
            }
        }
        client.indices.create(index=TRENDING_INDEX, body=mapping)
        logger.info(f"✅ Created index: {TRENDING_INDEX}")
    else:
        # Index exists - check if new fields exist and add them if missing
        try:
            current_mapping = client.indices.get_mapping(index=TRENDING_INDEX)
            properties = current_mapping.get(TRENDING_INDEX, {}).get("mappings", {}).get("properties", {})
            
            missing_fields = {}
            
            if "timestamp" not in properties:
                missing_fields["timestamp"] = {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            
            if "user_input_id" not in properties:
                missing_fields["user_input_id"] = {"type": "keyword"}
            
            if "filtered_sources" not in properties:
                missing_fields["filtered_sources"] = {"type": "keyword"}

            if "filter_key" not in properties:
                missing_fields["filter_key"] = {"type": "keyword"}

            if "centroid_embedding" not in properties:
                missing_fields["centroid_embedding"] = {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": False
                }

            if "last_post_timestamp" not in properties:
                missing_fields["last_post_timestamp"] = {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
            
            if missing_fields:
                logger.info(f"📝 Adding missing fields to existing index: {TRENDING_INDEX}")
                update_mapping = {"properties": missing_fields}
                client.indices.put_mapping(index=TRENDING_INDEX, body=update_mapping)
                logger.info(f"✅ Updated index mapping with fields: {list(missing_fields.keys())}")
            else:
                logger.info(f"ℹ️  Index '{TRENDING_INDEX}' already has all required fields")
        except Exception as e:
            logger.warning(f"⚠️  Could not update mapping: {e}")
    
    # Prepare actions
    actions = []
    for rank, topic_data in enumerate(topics, 1):
        topic_data["rank"] = rank
        actions.append({
            "_op_type": "index",
            "_index": TRENDING_INDEX,
            "_id": topic_data["cluster_id"],
            "_source": topic_data
        })
    
    # Bulk insert
    try:
        success, errors = helpers.bulk(
            client,
            actions,
            raise_on_error=False
        )
        logger.info(f"✅ Saved {success} trending topics")
        if errors:
            logger.warning(f"⚠️ {len(errors)} errors during bulk insert")
    except Exception as e:
        logger.error(f"❌ Error saving topics: {e}")
        raise

# ====== PRINT RESULTS ======
def print_trending_topics(topics: List[Dict]):
    """Print trending topics summary."""
    print("\n" + "=" * 80)
    print("🔥 TRENDING TOPICS")
    print("=" * 80)
    
    for i, topic_data in enumerate(topics[:MAX_TOPICS], 1):  # Top N
        print(f"\n{i}. {topic_data['topic']}")
        print(f"   📊 Posts: {topic_data['post_count']} | "
              f"Engagement: {topic_data['engagement_score']:.1f} | "
              f"Score: {topic_data['trending_score']:.2f}")
        print(f"   🏷️  Keywords: {', '.join(topic_data['keywords'][:3])}")
    
    print("\n" + "=" * 80)

# ====== MAIN ======
def main():
    """Main execution function."""
    start_time = time.time()

    try:
        validate_config()

        # Create OpenSearch client
        client = create_opensearch_client()

        # Ensure embedding field exists in index mapping
        ensure_embedding_mapping(client)

        # Fetch posts (including cached embeddings)
        docs = fetch_posts(client)

        if len(docs) < MIN_CLUSTER_SIZE:
            logger.error(f"❌ Not enough posts ({len(docs)} < {MIN_CLUSTER_SIZE})")
            return

        # Create embeddings (uses cache, only generates for new posts)
        embedding_processor = EmbeddingProcessor()
        embeddings = embedding_processor.create_embeddings(docs)

        # Save new embeddings back to OpenSearch for future runs
        save_embeddings_to_opensearch(client, docs)

        # Reduce dimensionality
        embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)

        # Cluster documents
        labels = embedding_processor.cluster_documents(embeddings_reduced)

        # Analyze trending topics
        trending_topics = analyze_trending_topics(
            docs,
            labels,
            embeddings_reduced=embeddings_reduced,
            embeddings_raw=embeddings,
            max_topics=MAX_TOPICS
        )

        if not trending_topics:
            logger.warning("⚠️ No trending topics found")
            return

        # Save to OpenSearch
        save_trending_topics(client, trending_topics)

        # Print results
        print_trending_topics(trending_topics)

        # Save JSON file
        output_file = f"trending_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(trending_topics, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved results to {output_file}")

        elapsed = time.time() - start_time
        logger.info(f"\n🎉 Complete! Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
        logger.info(f"📊 Processed {len(docs)} posts, found {len(trending_topics)} trending topics")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        raise
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    main()

