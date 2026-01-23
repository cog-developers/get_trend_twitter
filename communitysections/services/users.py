"""
User fetching and enrichment service.
Handles fetching users from OpenSearch and enriching with bio/tweets.
"""

from typing import List, Dict, Tuple
from functools import lru_cache

from ..config.settings import SOURCE_INDEX, get_logger
from ..database.opensearch import get_client, es_call_with_retries

logger = get_logger(__name__)


def fetch_all_users_scroll(limit: int = 0) -> List[Dict]:
    """Fetch users using scroll API (more robust, works with any mapping)."""
    client = get_client()
    logger.info(f"Fetching users from '{SOURCE_INDEX}' using scroll...")

    query = {
        "size": 100,
        "_source": ["user_details", "user_name", "user_username"],
        "query": {"match_all": {}},
    }

    users_map = {}

    try:
        response = es_call_with_retries(client.search, index=SOURCE_INDEX, body=query, scroll="2m")
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
                response = es_call_with_retries(client.scroll, scroll_id=scroll_id, scroll="2m")
                scroll_id = response.get("_scroll_id")
                hits = response.get("hits", {}).get("hits", [])
            except Exception as e:
                logger.error(f"Error during scroll fetch: {e}")
                break

        # Clear scroll
        try:
            if scroll_id:
                es_call_with_retries(client.clear_scroll, scroll_id=scroll_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Error in scroll fetch: {e}")
        return []

    users = list(users_map.values())
    if limit > 0:
        users = users[:limit]

    logger.info(f"Found {len(users)} unique users")
    return users


def fetch_all_users(limit: int = 0) -> List[Dict]:
    """Fetch all unique users from the source index using aggregations, fallback to scroll."""
    client = get_client()
    logger.info(f"Fetching users from '{SOURCE_INDEX}'...")

    try:
        mapping = es_call_with_retries(client.indices.get_mapping, index=SOURCE_INDEX)
        properties = mapping.get(SOURCE_INDEX, {}).get("mappings", {}).get("properties", {})
        user_details = properties.get("user_details", {}).get("properties", {})

        # Determine field names
        name_field = "user_details.name.keyword" if "name" in user_details else "user_name.keyword"
        username_field = "user_details.username.keyword" if "username" in user_details else "user_username.keyword"
        user_id_field = "user_details.user_id" if "user_id" in user_details else "user_id"

        if not properties.get(name_field.split('.')[0], {}).get('type') in ('keyword', 'text'):
            name_field = "user_name"

        if not properties.get(username_field.split('.')[0], {}).get('type') in ('keyword', 'text'):
            username_field = "user_username"

        logger.info(f"Using fields: user_id={user_id_field}, username={username_field}, name={name_field}")

        count_response = es_call_with_retries(client.count, index=SOURCE_INDEX)
        total_docs = count_response.get("count", 0)
        logger.info(f"Total documents in index: {total_docs}")

        if total_docs == 0:
            logger.warning(f"No documents found in '{SOURCE_INDEX}'")
            return []

    except Exception as e:
        logger.warning(f"Could not detect field mappings: {e}, falling back to scroll method")
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
                response = es_call_with_retries(client.search, index=SOURCE_INDEX, body=agg_query)
                buckets = response.get("aggregations", {}).get("unique_users", {}).get("buckets", [])

                if not buckets:
                    if iteration == 1:
                        logger.warning("No buckets returned from aggregation, trying scroll method")
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
                logger.error(f"Error fetching users with aggregation: {e}")
                if iteration == 1:
                    logger.warning("Aggregation failed, falling back to scroll method")
                    return fetch_all_users_scroll(limit)
                break

        logger.info(f"Found {len(users)} unique users via aggregation")
        return users

    except Exception as e:
        logger.warning(f"Aggregation failed: {e}, falling back to scroll method")
        return fetch_all_users_scroll(limit)


@lru_cache(maxsize=2000)
def enrich_user_with_details(user_id: str) -> Tuple[str, str]:
    """
    Enrich user with bio and recent tweets.
    Returns: (bio, tweets_text)
    """
    client = get_client()

    # Get sortable date fields
    date_sort_fields = []
    mapping = client.indices.get_mapping(index=SOURCE_INDEX)
    props = mapping[SOURCE_INDEX]['mappings']['properties']

    for f in ["created_at", "post_created_at", "timestamp"]:
        if f in props:
            date_sort_fields.append(f)

    if not date_sort_fields:
        logger.warning(f"No sortable date fields found for user {user_id}")

    query = {
        "query": {"term": {"user_details.user_id": user_id}},
        "size": 15,
        "_source": ["user_details.user_bio", "post_text"],
    }

    if date_sort_fields:
        query["sort"] = [{field: {"order": "desc"}} for field in date_sort_fields]

    try:
        response = es_call_with_retries(client.search, index=SOURCE_INDEX, body=query)
    except Exception as e:
        logger.error(f"Failed to fetch tweets for user {user_id}: {e}")
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
