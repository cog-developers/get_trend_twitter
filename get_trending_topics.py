"""
Trending Topics Generator
Reads posts from 'user-input-posts' index, generates topics, clusters them,
and identifies trending topics based on frequency and engagement.
"""

import os
import re
import time
import json
from datetime import datetime
from collections import Counter
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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

# ====== FETCH POSTS ======
def fetch_posts(client: OpenSearch) -> List[Dict]:
    """Fetch posts from user-input-posts index."""
    logger.info(f"📥 Fetching posts from index: {SOURCE_INDEX}")
    
    docs = []
    try:
        query = {
            "query": {"match_all": {}},
            "_source": ["post_text", "text", "content", "created_at", "timestamp", "author", "likes", "retweets", "replies"]
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
                    docs.append({
                        "id": hit["_id"],
                        "text": text.strip(),
                        "cleaned": cleaned,
                        "author": src.get("author"),
                        "created_at": src.get("created_at") or src.get("timestamp"),
                        "likes": src.get("likes", 0) or 0,
                        "retweets": src.get("retweets", 0) or 0,
                        "replies": src.get("replies", 0) or 0,
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error processing document {hit.get('_id')}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(docs)} valid posts")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error fetching posts: {e}")
        raise

# ====== EMBEDDING & CLUSTERING ======
class EmbeddingProcessor:
    """Handle text embeddings and clustering."""
    
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
        """Create embeddings for documents."""
        logger.info(f"🔢 Creating embeddings for {len(docs)} posts...")
        
        model = self.load_model()
        texts = [d["cleaned"] for d in docs]
        
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True
            )
            
            logger.info(f"✅ Created embeddings: shape={embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
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
    embeddings: np.ndarray
) -> List[Dict]:
    """Analyze clusters and identify trending topics."""
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
    
    logger.info(f"📈 Analyzing {len(valid_clusters)} valid clusters...")
    
    trending_topics = []
    
    for cluster_id, cluster_data in valid_clusters.items():
        posts = cluster_data["posts"]
        size = len(posts)
        engagement = cluster_data["total_engagement"]
        
        # Get representative texts (top 5)
        cluster_embeds = embeddings[cluster_data["indices"]]
        centroid = cluster_embeds.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeds, centroid).flatten()
        top_indices = np.argsort(-sims)[:5]
        representative_texts = [posts[i]["text"] for i in top_indices]
        
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
        
        trending_topics.append({
            "topic": topic,
            "cluster_id": f"cluster_{cluster_id}",
            "post_count": size,
            "engagement_score": engagement,
            "trending_score": trending_score,
            "keywords": keywords,
            "representative_texts": representative_texts[:3],
            "member_ids": [p["id"] for p in posts],
            "generated_at": current_time.isoformat(),
            "timestamp": timestamp_ms
        })
    
    # Sort by trending score
    trending_topics.sort(key=lambda x: x["trending_score"], reverse=True)
    
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
                    "generated_at": {"type": "date"},
                    "timestamp": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "rank": {"type": "integer"},
                    "user_input_id": {"type": "keyword"},
                    "filtered_sources": {"type": "keyword"}
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
    
    for i, topic_data in enumerate(topics[:20], 1):  # Top 20
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
        
        # Fetch posts
        docs = fetch_posts(client)
        
        if len(docs) < MIN_CLUSTER_SIZE:
            logger.error(f"❌ Not enough posts ({len(docs)} < {MIN_CLUSTER_SIZE})")
            return
        
        # Create embeddings
        embedding_processor = EmbeddingProcessor()
        embeddings = embedding_processor.create_embeddings(docs)
        
        # Reduce dimensionality
        embeddings_reduced = embedding_processor.reduce_dimensionality(embeddings)
        
        # Cluster documents
        labels = embedding_processor.cluster_documents(embeddings_reduced)
        
        # Analyze trending topics
        trending_topics = analyze_trending_topics(docs, labels, embeddings_reduced)
        
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

