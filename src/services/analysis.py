"""
Trending topics analysis service.
Handles cluster analysis, scoring, and topic identification.
"""

from datetime import datetime
from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from ..config.settings import (
    MIN_CLUSTER_SIZE,
    MAX_TOPICS,
    get_logger
)
from ..processing.text import clean_text, get_post_timestamp_millis
from .topics import generate_cluster_topic

logger = get_logger(__name__)


def analyze_trending_topics(
    docs: List[Dict],
    labels: np.ndarray,
    embeddings_reduced: np.ndarray,
    embeddings_raw: np.ndarray,
    max_topics: int = None
) -> List[Dict]:
    """
    Analyze clusters and identify trending topics.

    If more than `max_topics` clusters are found, clusters are merged down to
    `max_topics` using KMeans on cluster centroids (in reduced embedding space),
    so that each topic contains more posts.
    """
    max_topics = max_topics or MAX_TOPICS
    logger.info("Analyzing trending topics...")

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
        logger.info(f"Merging {len(valid_clusters)} clusters -> {max_topics} topics (MAX_TOPICS)")
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
            logger.warning(f"KMeans merge failed, falling back to top-{max_topics} clusters: {e}")
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

    logger.info(f"Analyzing {len(merged_clusters)} clusters after merge...")

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
        timestamp_ms = int(current_time.timestamp() * 1000)

        # Track max post timestamp in this topic (for incremental state)
        post_ts = []
        for p in posts:
            t = get_post_timestamp_millis(p)
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

    logger.info(f"Identified {len(trending_topics)} trending topics")
    return trending_topics
