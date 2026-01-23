"""
DeepSeek AI classification service.
Handles political party classification using DeepSeek API.
"""

import re
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from ..config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL,
    LEBANESE_PARTIES_PROMPT,
    MAX_WORKERS,
    RETRY_ATTEMPTS,
    RATE_LIMIT_DELAY,
    get_logger
)
from .users import enrich_user_with_details

logger = get_logger(__name__)


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

    # Find content between first '{' and last '}'
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
    Classify user with DeepSeek API.
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
{{   "party": "Exact party name from list above",   "confidence": "high|medium|low",   "reasoning": "Brief 2-3 sentence explanation" }}"""

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
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response_time = int((time.time() - start_time) * 1000)

        # Rate limiting handling
        if response.status_code == 429 and retry_count < RETRY_ATTEMPTS:
            wait_time = (2 ** retry_count) * RATE_LIMIT_DELAY
            logger.warning(f"Rate limited by DeepSeek, waiting {wait_time:.1f}s (retry {retry_count + 1})")
            time.sleep(wait_time + random.random() * 0.5)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)

        response.raise_for_status()
        result = response.json()

        # Validate structure
        choices = result.get("choices") or []
        if not choices:
            logger.error(f"No choices returned from DeepSeek for @{username}: {result}")
            return "Unknown", "low", "No API choices", response_time

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            content = json.dumps(result)

        # Extract JSON from content
        classification = extract_json_from_text(content)
        if not classification:
            logger.error(f"JSON parse error for @{username}: {content[:400]}")
            return "Unknown", "low", "Invalid API response format", response_time

        return (
            classification.get("party", "Unknown"),
            classification.get("confidence", "low"),
            classification.get("reasoning", "") or "",
            response_time,
        )

    except requests.exceptions.Timeout:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"API timeout for @{username}")
        if retry_count < RETRY_ATTEMPTS:
            time.sleep((2 ** retry_count) * RATE_LIMIT_DELAY)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)
        return "Unknown", "low", "API timeout", response_time

    except requests.exceptions.RequestException as e:
        response_time = int((time.time() - start_time) * 1000)
        err_str = str(e)
        if "401" in err_str or "Unauthorized" in err_str:
            logger.error(f"Authentication failed for @{username} - check DEEPSEEK_API_KEY")
            return "Unknown", "low", "API authentication failed", response_time

        logger.error(f"API error for @{username}: {e}")
        if retry_count < RETRY_ATTEMPTS:
            time.sleep((2 ** retry_count) * RATE_LIMIT_DELAY)
            return classify_with_deepseek(name, username, bio, recent_tweets, retry_count + 1)
        return "Unknown", "low", f"API error: {err_str}", response_time

    except json.JSONDecodeError as e:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"JSON decode error for @{username}: {e}")
        return "Unknown", "low", "Invalid API response format", response_time

    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        logger.error(f"Unexpected error for @{username}: {e}")
        return "Unknown", "low", str(e), response_time


def classify_user(user: Dict, run_id: str) -> Dict:
    """
    Worker function to classify a single user.
    Returns a dictionary (document) ready for indexing.
    """
    if not user or not isinstance(user, dict):
        raise ValueError("Invalid user object passed to classify_user")

    user_id = user.get("user_id")
    username = user.get("username", "unknown")
    name = user.get("name", "unknown")

    if not user_id:
        raise ValueError(f"Missing user_id for user: {username}")

    # 1. Enrich Data (Get Bio and Tweets)
    bio, recent_tweets = enrich_user_with_details(user_id)

    # 2. Classify
    party, confidence, reasoning, response_time = classify_with_deepseek(
        name=name,
        username=username,
        bio=bio,
        recent_tweets=recent_tweets
    )

    # 3. Construct the Document
    doc = {
        "user_id": user_id,
        "username": username,
        "name": name,
        "party": party,
        "confidence": confidence,
        "reasoning": reasoning,
        "tweet_count": user.get("doc_count", 0),
        "user_bio": bio,
        "sample_tweets": recent_tweets[:1200] if recent_tweets else "",
        "classified_at": datetime.utcnow().isoformat(),
        "classification_run_id": run_id,
        "api_response_time_ms": response_time,
    }

    return doc


def classify_users_parallel(users: List[Dict], run_id: str) -> List[Dict]:
    """Classify users in parallel with progress tracking."""
    classified = []
    logger.info(f"Starting parallel classification with {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(classify_user, user, run_id): user for user in users}

        for future in tqdm(as_completed(futures), total=len(users), desc="Classifying users", unit="user"):
            original_user = futures.get(future) or {}
            try:
                result = future.result()
                classified.append(result)
                # Jitter delay
                time.sleep(random.uniform(0.1, 0.5))
            except Exception as e:
                try:
                    user_id_safe = original_user.get("user_id") if isinstance(original_user, dict) else None
                    username_safe = original_user.get("username") if isinstance(original_user, dict) else "unknown"
                except Exception:
                    user_id_safe = None
                    username_safe = "unknown"

                logger.error(f"Failed to classify @{username_safe}: {e}")
                # Add fallback record
                classified.append({
                    "user_id": user_id_safe,
                    "username": username_safe,
                    "party": "Unknown",
                    "reasoning": f"System Error: {str(e)}",
                    "classification_run_id": run_id,
                    "classified_at": datetime.utcnow().isoformat()
                })

    return classified
