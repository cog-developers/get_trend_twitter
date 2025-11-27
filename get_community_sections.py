import os
import re
import json
import time
import logging
import random
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers, exceptions
from opensearchpy.connection import Urllib3HttpConnection
from tqdm import tqdm
from urllib3.exceptions import NewConnectionError

# ====== SETUP LOGGING ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ====== LOAD ENVIRONMENT VARIABLES ======
load_dotenv()

# ====== CONFIG ======
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE", "http://localhost:9200")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
SOURCE_INDEX = os.getenv("SOURCE_INDEX", "past-month-tweets-index")
TARGET_INDEX = os.getenv("TARGET_INDEX", "classified-users-index")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Switched to the faster, more stable 'deepseek-chat' model.
DEEPSEEK_MODEL = "deepseek-chat"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))

# Increased workers to leverage the faster model and improve throughput.
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
MAX_USERS = int(os.getenv("MAX_USERS", 0))      # 0 = all users

RETRY_ATTEMPTS = 4
RATE_LIMIT_DELAY = 1.5   # seconds between API calls
ES_RETRY_DELAY = 1.0

# ====== VALIDATE ENVIRONMENT ======
if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
    logger.error("❌ DEEPSEEK_API_KEY not set in environment!")
    raise ValueError("Missing DEEPSEEK_API_KEY")

logger.info(f"🔑 Using DeepSeek API Key: {DEEPSEEK_API_KEY[:8]}...{DEEPSEEK_API_KEY[-4:]}")
logger.info(f"🌐 API Endpoint: {DEEPSEEK_API_URL}")
logger.info(f"🤖 Model: {DEEPSEEK_MODEL}")

# ====== INITIALIZE OPENSEARCH CLIENT ======
def init_opensearch():
    try:
        es = OpenSearch(
            [OPENSEARCH_NODE],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            verify_certs=False,
            timeout=60,
            max_retries=3,
            retry_on_timeout=True,
            connection_class=Urllib3HttpConnection,
            pool_maxsize=10,
        )
        if not es.ping():
            raise ConnectionError("❌ Cannot connect to OpenSearch (ping failed)")
        logger.info("✅ Connected to OpenSearch")
        return es
    except exceptions.AuthenticationException:
        logger.error("❌ Authentication failed for OpenSearch. Check your username and password.")
        raise
    except exceptions.ConnectionError:
        logger.error("❌ Failed to connect to OpenSearch. Check the OPENSEARCH_NODE URL.")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while initializing OpenSearch: {e}")
        raise


es = init_opensearch()

# ====== LEBANESE POLITICAL PARTIES ======
LEBANESE_PARTIES_PROMPT = """
Major Lebanese Political Parties and Groups:
**Christian Parties:**
1. FPM (Free Patriotic Movement/التيار الوطني الحر) - Led by Gebran Bassil, founded by Michel Aoun
2. Lebanese Forces (القوات اللبنانية/LF) - Led by Samir Geagea
3. Kataeb (حزب الكتائب اللبنانية) - Led by Samy Gemayel
4. Marada Movement (تيار المردة) - Led by Sleiman Frangieh
**Shia Parties:**
5. Hezbollah (حزب الله) - Led by Hassan Nasrallah/Naim Qassem
6. Amal Movement (حركة أمل) - Led by Nabih Berri
**Sunni Parties:**
7. Future Movement (تيار المستقبل) - Led by Saad Hariri (currently suspended)
**Druze Parties:**
8. PSP (Progressive Socialist Party/الحزب التقدمي الاشتراكي) - Led by Walid Jumblatt/Taymour Jumblatt
**Independent/Opposition:**
9. MMFD (Citizens in a State/مواطنون ومواطنات في دولة) - Led by Charbel Nahas
10. LCP (Lebanese Communist Party/الحزب الشيوعي اللبناني)
11. October 17 Revolution Groups (مجموعات ثورة 17 تشرين) - Various independent activists
12. Sabaa Party (حزب سبعة) - Political reform movement
**Other Categories:**
13. Media/Journalist (إعلامي) - Professional journalists and media figures
14. Independent (مستقل/محايد) - No clear party affiliation
15. Unknown (غير معروف) - Insufficient information
"""

# ====== CREATE TARGET INDEX ======
def create_classified_index(force_recreate: bool = False):
    """Create or recreate the classified users index"""
    mapping = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "index": {"refresh_interval": "5s"},
        },
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "username": {"type": "keyword"},
                "name": {"type": "text"},
                "party": {"type": "keyword"},
                "confidence": {"type": "keyword"},
                "reasoning": {"type": "text"},
                "tweet_count": {"type": "integer"},
                "user_bio": {"type": "text"},
                "sample_tweets": {"type": "text"},
                "classified_at": {"type": "date"},
                "classification_run_id": {"type": "keyword"},
                "api_response_time_ms": {"type": "integer"},
                "created_at": {"type": "date"},
                "post_created_at": {"type": "date"},
                "timestamp": {"type": "date"},
            }
        },
    }

    try:
        if es.indices.exists(index=TARGET_INDEX):
            if force_recreate:
                logger.info(f"🗑️  Deleting existing index '{TARGET_INDEX}'...")
                es.indices.delete(index=TARGET_INDEX)
            else:
                logger.info(f"ℹ️  Index '{TARGET_INDEX}' already exists, will append results")
                return
        es.indices.create(index=TARGET_INDEX, body=mapping)
        logger.info(f"✅ Created index '{TARGET_INDEX}'")
    except Exception as e:
        logger.error(f"❌ Error creating index: {e}")
        raise


# ====== HELPER: RETRY WRAPPER FOR ES CALLS ======
def es_call_with_retries(fn, *args, **kwargs):
    """Call an OpenSearch function with exponential backoff retries for network errors."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            err_str = str(e)
            if attempt > RETRY_ATTEMPTS:
                # IMPORTANT: If it's a 400 RequestError (like the mapping error), we re-raise immediately
                # so the calling function (enrich_user_with_details) can handle the fallback.
                if isinstance(e, exceptions.RequestError) and e.status_code == 400:
                    raise # Re-raise for specific handling

                logger.error(f"❌ OpenSearch operation failed after {attempt} attempts: {e}")
                raise
            # Retry on connection related errors or transient errors
            if isinstance(e, exceptions.RequestError) and e.status_code == 400:
                # If it's a 400, don't retry here, let the calling function handle it immediately (after one attempt)
                raise

            logger.warning(f"⏳ OpenSearch error (attempt {attempt}/{RETRY_ATTEMPTS}): {e}. Retrying...")
            sleep_time = ES_RETRY_DELAY * (2 ** (attempt - 1)) + random.random() * 0.5
            time.sleep(sleep_time)


# ====== FETCH USERS - SCROLL (robust) ======
def fetch_all_users_scroll(limit: int = 0) -> List[Dict]:
    """Fetch users using scroll API (more robust, works with any mapping)"""
    logger.info(f"📥 Fetching users from '{SOURCE_INDEX}' using scroll...")

    query = {
        "size": 100,
        "_source": ["user_details", "user_name", "user_username"],
        "query": {"match_all": {}},
    }

    users_map = {}

    try:
        response = es_call_with_retries(es.search, index=SOURCE_INDEX, body=query, scroll="2m")
        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])
        while hits:
            for hit in hits:
                source = hit.get("_source", {})
                user_details = source.get("user_details", {})
                user_id = user_details.get("user_id") or source.get("user_id")
                username = user_details.get("username") or source.get("user_username")
                name = user_details.get("name") or source.get("user_name")

                if not user_id:
                    continue

                if user_id not in users_map:
                    users_map[user_id] = {
                        "user_id": user_id,
                        "username": username or f"user_{user_id}",
                        "name": name or username or f"User {user_id}",
                        "doc_count": 1,
                    }
                else:
                    users_map[user_id]["doc_count"] += 1

                if limit > 0 and len(users_map) >= limit:
                    break

            if limit > 0 and len(users_map) >= limit:
                break

            if not scroll_id:
                break

            try:
                response = es_call_with_retries(es.scroll, scroll_id=scroll_id, scroll="2m")
                scroll_id = response.get("_scroll_id")
                hits = response.get("hits", {}).get("hits", [])
            except Exception as e:
                logger.error(f"❌ Error during scroll fetch: {e}")
                break

        # Clear scroll if we have one
        try:
            if scroll_id:
                es_call_with_retries(es.clear_scroll, scroll_id=scroll_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"❌ Error in scroll fetch: {e}")
        return []

    users = list(users_map.values())
    if limit > 0:
        users = users[:limit]

    logger.info(f"✅ Found {len(users)} unique users")
    return users


# ====== FETCH USERS (aggregation fallback) ======
def fetch_all_users(limit: int = 0) -> List[Dict]:
    """Fetch all unique users from the source index using aggregations, fallback to scroll."""
    logger.info(f"📥 Fetching users from '{SOURCE_INDEX}'...")

    try:
        mapping = es_call_with_retries(es.indices.get_mapping, index=SOURCE_INDEX)
        properties = mapping.get(SOURCE_INDEX, {}).get("mappings", {}).get("properties", {})
        user_details = properties.get("user_details", {}).get("properties", {})

        # Attempt to dynamically determine field names
        name_field = "user_details.name.keyword" if "name" in user_details else "user_name.keyword"
        username_field = "user_details.username.keyword" if "username" in user_details else "user_username.keyword"
        # User ID is often nested, but ensure it's correct for aggregation
        user_id_field = "user_details.user_id" if "user_id" in user_details else "user_id"

        # Check if the fields are actually available as keyword fields for aggregation
        if not properties.get(name_field.split('.')[0], {}).get('type') in ('keyword', 'text'):
             name_field = "user_name" # Fallback to non-keyword for aggregation fields if needed

        if not properties.get(username_field.split('.')[0], {}).get('type') in ('keyword', 'text'):
             username_field = "user_username"

        logger.info(f"📋 Using fields: user_id={user_id_field}, username={username_field}, name={name_field}")

        count_response = es_call_with_retries(es.count, index=SOURCE_INDEX)
        total_docs = count_response.get("count", 0)
        logger.info(f"📊 Total documents in index: {total_docs}")

        if total_docs == 0:
            logger.warning(f"⚠️  No documents found in '{SOURCE_INDEX}'")
            return []

    except Exception as e:
        logger.warning(f"⚠️  Could not detect field mappings: {e}, falling back to scroll method")
        return fetch_all_users_scroll(limit)

    agg_query = {
        "size": 0,
        "aggs": {
            "unique_users": {
                "composite": {
                    "size": 100,
                    "sources": [
                        {"user_id": {"terms": {"field": user_id_field}}},
                        {"username": {"terms": {"field": username_field}}},
                        {"name": {"terms": {"field": name_field}}},
                    ],
                }
            }
        },
    }

    users = []
    after_key = None
    try:
        iteration = 0
        while True:
            iteration += 1
            if after_key:
                agg_query["aggs"]["unique_users"]["composite"]["after"] = after_key
            try:
                response = es_call_with_retries(es.search, index=SOURCE_INDEX, body=agg_query)
                buckets = response.get("aggregations", {}).get("unique_users", {}).get("buckets", [])
                logger.debug(f"Iteration {iteration}: Got {len(buckets)} buckets")
                if not buckets:
                    if iteration == 1:
                        logger.warning("⚠️  No buckets returned from aggregation, trying scroll method")
                        return fetch_all_users_scroll(limit)
                    break
                for bucket in buckets:
                    key = bucket.get("key", {})
                    users.append({
                        "user_id": key.get("user_id"),
                        "username": key.get("username"),
                        "name": key.get("name"),
                        "doc_count": bucket.get("doc_count", 0),
                    })
                after_key = response.get("aggregations", {}).get("unique_users", {}).get("after_key")
                if not after_key:
                    break
                if limit > 0 and len(users) >= limit:
                    users = users[:limit]
                    break
            except Exception as e:
                logger.error(f"❌ Error fetching users with aggregation: {e}")
                if iteration == 1:
                    logger.warning("⚠️  Aggregation failed, falling back to scroll method")
                    return fetch_all_users_scroll(limit)
                break

        logger.info(f"✅ Found {len(users)} unique users via aggregation")
        return users
    except Exception as e:
        logger.warning(f"⚠️  Aggregation failed: {e}, falling back to scroll method")
        return fetch_all_users_scroll(limit)


# ====== ENRICH USER WITH DETAILS (FIXED: Robust sort field checking) ======
@lru_cache(maxsize=2000)
def enrich_user_with_details(user_id: str) -> Tuple[str, str]:
    date_sort_fields = []
    mapping = es.indices.get_mapping(index=SOURCE_INDEX)
    props = mapping[SOURCE_INDEX]['mappings']['properties']

    for f in ["created_at", "post_created_at", "timestamp"]:
        if f in props:
            date_sort_fields.append(f)

    if not date_sort_fields:
        logger.warning(f"No sortable date fields found for user {user_id}")
        date_sort_fields = []  # fallback to unsorted

    query = {
        "query": {"term": {"user_details.user_id": user_id}},
        "size": 15,
        "_source": ["user_details.user_bio", "post_text"],
    }

    if date_sort_fields:
        query["sort"] = [{field: {"order": "desc"}} for field in date_sort_fields]

    try:
        response = es_call_with_retries(es.search, index=SOURCE_INDEX, body=query)
    except Exception as e:
        logger.error(f"❌ Failed to fetch tweets for user {user_id}: {e}")
        return "", ""

    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        return "", ""

    bio = hits[0].get("_source", {}).get("user_details", {}).get("user_bio", "")
    tweets = [
        h.get("_source", {}).get("post_text", "")[:300]
        for h in hits[:10]
        if h.get("_source", {}).get("post_text")
    ]
    tweets_text = " | ".join(tweets)

    return bio, tweets_text




# ====== DEEPSEEK API CLASSIFICATION ======
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Attempt to find a JSON object inside text (handles code fences or plain JSON).
    Returns dict or None.
    """
    if not text:
        return None
    # First try direct json load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Finds the content between the first '{' and the last '}' in the string.
    # Handles markdown code fences (```json{...}```) and plain text
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)

    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            # Try to fix common mistakes: trailing commas
            candidate_fixed = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(candidate_fixed)
            except Exception:
                return None
    return None


def classify_with_deepseek(
    name: str,
    username: str,
    bio: str,
    recent_tweets: str,
    retry_count: int = 0,
) -> Tuple[str, str, str, int]:
    """
    Classify user with DeepSeek API
    Returns: (party, confidence, reasoning, response_time_ms)
    """
    prompt = f"""You are an expert analyst of Lebanese politics. Analyze this Twitter user and classify their political affiliation.

**User Profile:**
- Name: {name}
- Username: @{username}
- Bio: {bio[:400] if bio else "N/A"}
- Recent Tweets (sample): {recent_tweets[:1200] if recent_tweets else "N/A"}

{LEBANESE_PARTIES_PROMPT}

**Analysis Instructions:**
1. Identify political leanings from name, bio, and tweet content
2. Look for: party membership indicators, support/opposition rhetoric, shared ideologies, mentions of party leaders
3. Consider Arabic and English content
4. Classify into ONE category from the list above
5. If truly ambiguous, use "Independent" or "Unknown"

**Respond ONLY with valid JSON:**
{{   "party": "Exact party name from list above",   "confidence": "high|medium|low",   "reasoning": "Brief 2-3 sentence explanation" }}"""

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert political analyst specializing in Lebanese politics. Respond ONLY with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }

    start_time = time.time()

    try:
        # Increased timeout to 60s for safety, though the faster model should complete quickly.
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response_time = int((time.time() - start_time) * 1000)

        # Rate limiting handling
        if response.status_code == 429 and retry_count < RETRY_ATTEMPTS:
            wait_time = (2 ** retry_count) * RATE_LIMIT_DELAY
            logger.warning(f"⏳ Rate limited by DeepSeek, waiting {wait_time:.1f}s (retry {retry_count + 1})")
            time.sleep(wait_time + random.random() * 0.5)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)

        response.raise_for_status()
        result = response.json()

        # Validate structure
        choices = result.get("choices") or []
        if not choices:
            logger.error(f"❌ No choices returned from DeepSeek for @{username}: {result}")
            return "Unknown", "low", "No API choices", response_time

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            # Try alternatives
            content = json.dumps(result)

        # Try to extract JSON from the content robustly
        classification = extract_json_from_text(content)
        if not classification:
            logger.error(f"❌ JSON parse error for @{username}: {content[:400]}")
            return "Unknown", "low", "Invalid API response format", response_time

        return (
            classification.get("party", "Unknown"),
            classification.get("confidence", "low"),
            classification.get("reasoning", "") or "",
            response_time,
        )

    except requests.exceptions.Timeout:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"⏱️  API timeout for @{username}")
        if retry_count < RETRY_ATTEMPTS:
            time.sleep((2 ** retry_count) * RATE_LIMIT_DELAY)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)
        return "Unknown", "low", "API timeout", response_time

    except requests.exceptions.RequestException as e:
        response_time = int((time.time() - start_time) * 1000)
        err_str = str(e)
        if "401" in err_str or "Unauthorized" in err_str:
            logger.error(f"❌ Authentication failed for @{username} - check DEEPSEEK_API_KEY")
            return "Unknown", "low", "API authentication failed", response_time

        logger.error(f"❌ API error for @{username}: {e}")
        if retry_count < RETRY_ATTEMPTS:
            time.sleep((2 ** retry_count) * RATE_LIMIT_DELAY)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)
        return "Unknown", "low", f"API error: {err_str}", response_time

    except json.JSONDecodeError as e:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ JSON decode error for @{username}: {e}")
        return "Unknown", "low", "Invalid API response format", response_time

    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Unexpected error for @{username}: {e}")
        return "Unknown", "low", str(e), response_time


# ====== CLASSIFY SINGLE USER (WORKER) ======
def classify_user(user: Dict, run_id: str) -> Dict:
    """
    Worker function to classify a single user.
    Returns a Dictionary (document) ready for indexing.
    """
    # Defensive checks: ensure `user` is a dict and contains a user_id
    if not user or not isinstance(user, dict):
        raise ValueError("Invalid user object passed to classify_user")

    user_id = user.get("user_id")
    username = user.get("username", "unknown")
    name = user.get("name", "unknown")

    if not user_id:
        raise ValueError(f"Missing user_id for user: {username}")

    # 1. Enrich Data (Get Bio and Tweets)
    bio, recent_tweets = enrich_user_with_details(user_id)

    # 2. Classify (Using the Party Logic)
    party, confidence, reasoning, response_time = classify_with_deepseek(
        name=name,
        username=username,
        bio=bio,
        recent_tweets=recent_tweets
    )

    # 3. Construct the Document Dictionary
    doc = {
        "user_id": user_id,
        "username": username,
        "name": name,
        "party": party,
        "confidence": confidence,
        "reasoning": reasoning,
        "tweet_count": user.get("doc_count", 0),
        "user_bio": bio,
        # Save up to 1200 characters of the sample tweets
        "sample_tweets": recent_tweets[:1200] if recent_tweets else "",
        "classified_at": datetime.utcnow().isoformat(),
        "classification_run_id": run_id,
        "api_response_time_ms": response_time,
    }

    return doc

# ====== PARALLEL CLASSIFICATION ======
def classify_users_parallel(users: List[Dict], run_id: str) -> List[Dict]:
    """Classify users in parallel with progress tracking"""
    classified = []
    logger.info(f"🚀 Starting parallel classification with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks; keep mapping of future -> original_user for safe error reporting
        futures = {executor.submit(classify_user, user, run_id): user for user in users}

        for future in tqdm(as_completed(futures), total=len(users), desc="Classifying users", unit="user"):
            original_user = futures.get(future) or {}
            username_safe = original_user.get("username") if isinstance(original_user, dict) else None
            try:
                result = future.result()
                classified.append(result)
                # Jitter delay
                time.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                # Avoid calling .get on None; use safe defaults
                try:
                    user_id_safe = original_user.get("user_id") if isinstance(original_user, dict) else None
                    username_safe = original_user.get("username") if isinstance(original_user, dict) else "unknown"
                except Exception:
                    user_id_safe = None
                    username_safe = "unknown"

                logger.error(f"❌ Failed to classify @{username_safe}: {e}")
                # Add fallback so we don't lose the record
                classified.append({
                    "user_id": user_id_safe,
                    "username": username_safe,
                    "party": "Unknown",
                    "reasoning": f"System Error: {str(e)}",
                    "classification_run_id": run_id,
                    "classified_at": datetime.utcnow().isoformat()
                })

    return classified
# ====== SAVE TO OPENSEARCH ======
def save_classifications_bulk(classified_users: List[Dict]):
    """Save classifications to OpenSearch in bulk"""
    if not classified_users:
        logger.warning("⚠️  No classifications to save")
        return
    logger.info(f"💾 Saving {len(classified_users)} classifications to '{TARGET_INDEX}'...")

    actions = [
        {"_index": TARGET_INDEX, "_id": user["user_id"], "_source": user} for user in classified_users
    ]
    try:
        success, failed = helpers.bulk(
            es,
            actions,
            chunk_size=100,
            request_timeout=120,
            raise_on_error=False,
        )
        logger.info(f"✅ Bulk save reported success={success}")
        if failed:
            logger.warning(f"⚠️  Some bulk items failed (see server logs)")
    except Exception as e:
        logger.error(f"❌ Bulk save error: {e}")


# ====== PRINT STATISTICS ======
def print_classification_summary(classified_users: List[Dict]):
    """Print detailed classification statistics"""
    party_counts = Counter()
    confidence_counts = Counter()
    avg_response_times = defaultdict(list)

    for user in classified_users:
        party_counts[user.get("party", "Unknown")] += 1
        confidence_counts[user.get("confidence", "low")] += 1
        avg_response_times[user.get("confidence", "low")].append(user.get("api_response_time_ms", 0))

    total = len(classified_users)
    print("\n" + "=" * 80)
    print("📊 CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"\n📈 Total Users Classified: {total}")

    print("\n🎯 By Political Party:")
    print("-" * 60)
    for party, count in sorted(party_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total else 0
        bar = "█" * int(percentage / 2)
        print(f"  {party:30s} │ {count:4d} ({percentage:5.1f}%) {bar}")

    print("\n🎯 By Confidence Level:")
    print("-" * 60)
    for conf, count in sorted(confidence_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100 if total else 0
        avg_time = int(sum(avg_response_times[conf]) / len(avg_response_times[conf])) if avg_response_times[conf] else 0
        print(f"  {conf:10s} │ {count:4d} ({percentage:5.1f}%) │ Avg: {avg_time}ms")

    print("\n📝 Sample Classifications:")
    print("-" * 80)
    for party in list(party_counts.keys())[:5]:
        samples = [u for u in classified_users if u.get("party") == party][:2]
        if samples:
            print(f"\n  {party}:")
            for s in samples:
                print(f"    • @{s.get('username')} ({s.get('name')})")
                print(f"      └─ {s.get('reasoning', '')[:150]}...")
    print("\n" + "=" * 80)


# ====== PRINT CLASSIFICATION RESULTS ======
def print_classification_results(classified_users: List[Dict]):
    """Print detailed classification results for each user."""
    print("\n" + "=" * 80)
    print("📋 DETAILED CLASSIFICATION RESULTS")
    print("=" * 80)

    for user in classified_users:
        user_id = user.get("user_id", "N/A")
        username = user.get("username", "N/A")
        party = user.get("party", "Unknown")
        confidence = user.get("confidence", "low")
        reasoning = user.get("reasoning", "No reasoning provided")

        print(f"\n👤 User: @{username} (ID: {user_id})")
        print(f"   - Party: {party}")
        print(f"   - Confidence: {confidence}")
        print(f"   - Reasoning: {reasoning}")


# ====== VALIDATE INDEX MAPPING ======
def validate_index_mapping(index: str, fields: List[str]) -> None:
    """
    Validate that the required fields exist in the index mapping and are properly configured.
    Logs warnings for any missing or improperly mapped fields.
    """
    try:
        mapping = es_call_with_retries(es.indices.get_mapping, index=index)
        properties = mapping.get(index, {}).get("mappings", {}).get("properties", {})

        for field in fields:
            if field not in properties:
                logger.warning(f"⚠️  Field '{field}' is missing in the index mapping for '{index}'.")
            elif properties[field].get("type") != "date":
                logger.warning(f"⚠️  Field '{field}' in index '{index}' is not mapped as a 'date' field.")
            else:
                logger.info(f"✅ Field '{field}' is correctly mapped as 'date' in index '{index}'.")
    except Exception as e:
        logger.error(f"❌ Failed to validate index mapping for '{index}': {e}")

# Validate the required fields in the source index
validate_index_mapping(SOURCE_INDEX, ["created_at", "post_created_at", "timestamp"])

# ====== MAIN EXECUTION ======
def main():
    """Main execution pipeline"""
    start_time = time.time()
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"🆔 Classification Run ID: {run_id}")

    create_classified_index(force_recreate=False)

    users = fetch_all_users(limit=MAX_USERS)
    if not users:
        logger.error("❌ No users found to classify")
        return

    # Filter out invalid entries (None or missing user_id)
    total_before = len(users)
    users = [u for u in users if isinstance(u, dict) and u.get("user_id")]
    filtered_out = total_before - len(users)
    if filtered_out:
        logger.warning(f"⚠️  Filtered out {filtered_out} invalid user entries before classification")

    classified = classify_users_parallel(users, run_id)
    save_classifications_bulk(classified)
    print_classification_summary(classified)
    print_classification_results(classified)

    elapsed = time.time() - start_time
    per_user = elapsed / len(users) if users else 0
    logger.info(f"\n⏱️  Total execution time: {elapsed:.1f}s ({per_user:.2f}s per user)")
    logger.info(f"✅ Results saved to OpenSearch index: '{TARGET_INDEX}'")
    logger.info(f"🔍 Query example: GET {TARGET_INDEX}/_search?q=classification_run_id:{run_id}")


# ====== ENTRY POINT ======
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Classification interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        raise