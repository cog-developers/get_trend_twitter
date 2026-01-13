import os
import re
import requests
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# ===== CONFIG =====
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "user-input-posts")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Processing config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
MIN_MEANINGFUL_WORDS = int(os.getenv("MIN_MEANINGFUL_WORDS", "3"))

ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "إلى", "مع", "عن", "ما", "لا", 
    "هذا", "هذه", "هو", "هي", "أن", "إن", "كان", "كل", "قد",
    "لم", "لن", "ثم", "أو", "بل", "لكن", "حتى", "عند", "بعد", "قبل"
}

# ===== VALIDATION =====
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

# ===== OpenSearch client =====
def create_opensearch_client() -> OpenSearch:
    """Create OpenSearch client with proper error handling."""
    try:
        client = OpenSearch(
            [OPENSEARCH_NODE],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=90,
            max_retries=3,
            retry_on_timeout=True
        )
        # Test connection
        client.info()
        logger.info("✅ Connected to OpenSearch")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to connect to OpenSearch: {e}")
        raise

# ===== Utilities =====
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
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords and short words
    words = [w for w in text.split() if len(w) > 2 and w not in ARABIC_STOPWORDS]
    
    return " ".join(words)

def is_meaningful(text: str, min_words: int = MIN_MEANINGFUL_WORDS) -> bool:
    """Check if a post has meaningful content."""
    if not text:
        return False
    
    words = text.split()
    
    # Check word count
    if len(words) < min_words:
        return False
    
    # Check if not just repeated characters
    unique_chars = len(set(''.join(words)))
    if unique_chars < 3:
        return False
    
    return True

def fetch_posts(client: OpenSearch, batch_size: int = 1000) -> List[Dict]:
    """Fetch posts from OpenSearch with pagination."""
    logger.info(f"🔹 Fetching posts from index: {SOURCE_INDEX}")
    
    docs = []
    try:
        # Use scroll API for efficient large result retrieval
        query = {
            "query": {"match_all": {}},
            "_source": ["post_text"],
            "size": batch_size
        }
        
        for hit in helpers.scan(
            client, 
            query=query, 
            index=SOURCE_INDEX,
            scroll='5m',
            raise_on_error=False
        ):
            try:
                text = hit.get("_source", {}).get("post_text", "")
                if not text:
                    continue
                
                cleaned = clean_text(text)
                if is_meaningful(cleaned):
                    docs.append({
                        "id": hit["_id"],
                        "post_text": text,
                        "cleaned": cleaned
                    })
            except Exception as e:
                logger.warning(f"⚠️ Error processing document {hit.get('_id')}: {e}")
                continue
        
        logger.info(f"✅ Fetched {len(docs)} meaningful posts")
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error fetching posts: {e}")
        raise

# ===== DeepSeek API with retry logic =====
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def call_deepseek_api(prompt: str) -> Optional[str]:
    """Call DeepSeek API with retry logic."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 60
    }
    
    response = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    
    return response.json()["choices"][0]["message"]["content"].strip()

def generate_topic(text: str) -> Optional[str]:
    """Generate a single-sentence topic for a post."""
    prompt = f"""أنت محلل خبير للمحتوى العربي على وسائل التواصل الاجتماعي.
مهمتك: أعطني جملة واحدة واضحة ومختصرة تصف الموضوع الرئيسي لهذا المنشور.

المنشور:
{text[:500]}

اكتب الموضوع فقط كجملة مفيدة كاملة (لا تزيد عن 15 كلمة):"""

    try:
        topic = call_deepseek_api(prompt)
        
        # Clean the topic
        topic = re.sub(r'^[\d\.\-\s:*•]+', '', topic)
        topic = topic.strip('"').strip("'").strip()
        
        # Validate topic
        if len(topic) < 5 or len(topic.split()) < 2:
            logger.warning(f"⚠️ Generated topic too short: {topic}")
            return None
        
        return topic
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error generating topic: {e}")
        return None

def process_post(post: Dict) -> Optional[Dict]:
    """Process a single post to generate its topic."""
    try:
        topic = generate_topic(post["post_text"])
        
        if not topic:
            return None
        
        return {
            "_op_type": "update",
            "_index": SOURCE_INDEX,
            "_id": post["id"],
            "doc": {"topic": topic}
        }
    except Exception as e:
        logger.error(f"❌ Error processing post {post['id']}: {e}")
        return None

def process_posts_parallel(posts: List[Dict], max_workers: int = MAX_WORKERS) -> List[Dict]:
    """Process posts in parallel with rate limiting."""
    logger.info(f"🔹 Processing {len(posts)} posts with {max_workers} workers")
    
    actions = []
    total = len(posts)
    processed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_post = {
            executor.submit(process_post, post): post 
            for post in posts
        }
        
        # Process completed tasks
        for future in as_completed(future_to_post):
            processed += 1
            
            try:
                result = future.result()
                if result:
                    actions.append(result)
                
                # Log progress every 10%
                if processed % max(1, total // 10) == 0:
                    logger.info(f"⏳ Progress: {processed}/{total} ({100*processed//total}%)")
                
            except Exception as e:
                post = future_to_post[future]
                logger.error(f"❌ Failed to process post {post['id']}: {e}")
    
    logger.info(f"✅ Successfully processed {len(actions)}/{total} posts")
    return actions

def update_opensearch_bulk(client: OpenSearch, actions: List[Dict], batch_size: int = BATCH_SIZE):
    """Update OpenSearch in batches with error handling."""
    if not actions:
        logger.warning("⚠️ No actions to update")
        return
    
    total = len(actions)
    logger.info(f"🔹 Updating {total} documents in batches of {batch_size}")
    
    success_count = 0
    error_count = 0
    
    for i in range(0, total, batch_size):
        batch = actions[i:i + batch_size]
        
        try:
            success, errors = helpers.bulk(
                client,
                batch,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            success_count += success
            
            if errors:
                error_count += len(errors)
                logger.warning(f"⚠️ Batch {i//batch_size + 1}: {len(errors)} errors")
                
            logger.info(f"⏳ Updated batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
            
        except Exception as e:
            error_count += len(batch)
            logger.error(f"❌ Failed to update batch {i//batch_size + 1}: {e}")
    
    logger.info(f"✅ Update complete: {success_count} successful, {error_count} failed")

# ===== Main =====
def main():
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Validate configuration
        validate_config()
        
        # Create OpenSearch client
        client = create_opensearch_client()
        
        # Fetch posts
        posts = fetch_posts(client)
        
        if not posts:
            logger.warning("❌ No meaningful posts to process")
            return
        
        logger.info(f"🔹 Found {len(posts)} posts to process")
        
        # Process posts in parallel
        actions = process_posts_parallel(posts, max_workers=MAX_WORKERS)
        
        if not actions:
            logger.warning("⚠️ No topics generated successfully")
            return
        
        # Update OpenSearch
        update_opensearch_bulk(client, actions, batch_size=BATCH_SIZE)
        
        elapsed = time.time() - start_time
        logger.info(f"🎉 Pipeline complete in {elapsed:.2f} seconds")
        logger.info(f"📊 Average: {elapsed/len(posts):.2f} seconds per post")
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise
    finally:
        # Cleanup
        if 'client' in locals():
            client.close()
            logger.info("🔌 OpenSearch connection closed")

if __name__ == "__main__":
    main()