"""Trending topics analysis service."""

import re
from datetime import datetime
from collections import Counter
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from src.config.settings import MIN_CLUSTER_SIZE, MAX_TOPICS
from src.common.text_utils import clean_text, get_post_timestamp_millis
from src.services.topic_generation_service import generate_cluster_topic
from src.logging.logger import get_logger

logger = get_logger(__name__)


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
    
    # Group documents by cluster, skip noise
    for idx, (doc, label) in enumerate(zip(docs, labels)):
        if label == -1:  # Skip noise
            continue
        
        if label not in clusters:
            clusters[label] = {
                "posts": [],
                "indices": []
            }
        
        clusters[label]["posts"].append(doc)
        clusters[label]["indices"].append(idx)
    
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
        
        # Use cosine similarity on centroids instead of KMeans for better semantic merging
        centroids = []
        for cid in cluster_ids:
            idxs = valid_clusters[cid]["indices"]
            centroids.append(embeddings_reduced[idxs].mean(axis=0))
        centroids = np.vstack(centroids)
        
        # Normalize centroids for cosine similarity
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        centroids_norm = centroids / norms
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(centroids_norm)
        
        # Use hierarchical clustering approach: merge most similar clusters first
        # until we reach max_topics
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            # Make symmetric and ensure diagonal is 0
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # Convert to condensed form for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='average')
            cluster_assignments = fcluster(linkage_matrix, max_topics, criterion='maxclust')
            
            # Group clusters by assignment
            grouped = {}
            for cid, assignment in zip(cluster_ids, cluster_assignments):
                grouped.setdefault(int(assignment), {"posts": [], "indices": []})
                grouped[int(assignment)]["posts"].extend(valid_clusters[cid]["posts"])
                grouped[int(assignment)]["indices"].extend(valid_clusters[cid]["indices"])
            
            merged_clusters = grouped
            logger.info(f"✅ Merged clusters using hierarchical clustering based on cosine similarity")
            
        except ImportError:
            logger.warning("⚠️ scipy not available, falling back to KMeans")
            # Fallback to KMeans
            try:
                km = KMeans(n_clusters=max_topics, random_state=42, n_init=10)
                assignments = km.fit_predict(centroids_norm)
            except Exception as e:
                logger.warning(f"⚠️ KMeans merge failed, falling back to top-{max_topics} clusters: {e}")
                # Fallback: keep largest clusters only
                top = sorted(
                    valid_clusters.items(),
                    key=lambda kv: len(kv[1]["posts"]),  # Sort by cluster size only
                    reverse=True
                )[:max_topics]
                merged_clusters = {i: v for i, (_, v) in enumerate(top)}
            else:
                grouped = {}
                for cid, g in zip(cluster_ids, assignments):
                    grouped.setdefault(int(g), {"posts": [], "indices": []})
                    grouped[int(g)]["posts"].extend(valid_clusters[cid]["posts"])
                    grouped[int(g)]["indices"].extend(valid_clusters[cid]["indices"])
                merged_clusters = grouped
        except Exception as e:
            logger.warning(f"⚠️ Hierarchical clustering failed: {e}, falling back to top-{max_topics} clusters")
            # Fallback: keep largest clusters only
            top = sorted(
                valid_clusters.items(),
                key=lambda kv: len(kv[1]["posts"]),
                reverse=True
            )[:max_topics]
            merged_clusters = {i: v for i, (_, v) in enumerate(top)}

    logger.info(f"📈 Analyzing {len(merged_clusters)} clusters after merge...")
    
    trending_topics = []
    filtered_count = 0
    
    for cluster_id, cluster_data in merged_clusters.items():
        posts = cluster_data["posts"]
        size = len(posts)
        
        # Validate cluster quality: check intra-cluster similarity
        cluster_embeds_reduced = embeddings_reduced[cluster_data["indices"]]
        centroid_reduced = cluster_embeds_reduced.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeds_reduced, centroid_reduced).flatten()
        
        # Calculate average similarity to centroid (should be high for good clusters)
        avg_similarity = float(np.mean(sims))
        min_similarity = float(np.min(sims))
        
        # Filter out clusters with poor content correlation
        MIN_CLUSTER_SIMILARITY = 0.65  # Minimum average similarity to centroid
        if avg_similarity < MIN_CLUSTER_SIMILARITY:
            logger.warning(
                f"⚠️ Filtering out cluster {cluster_id}: low average similarity {avg_similarity:.3f} "
                f"(min: {min_similarity:.3f}) - posts are not well correlated"
            )
            filtered_count += 1
            continue
        
        if avg_similarity < 0.75:
            logger.info(
                f"ℹ️ Cluster {cluster_id} has moderate similarity: {avg_similarity:.3f} "
                f"(min: {min_similarity:.3f})"
            )
        
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
        
        # Calculate trending score based only on cluster size (content similarity)
        # Larger clusters = more posts discussing similar content = more trending
        trending_score = size
        
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
            t = get_post_timestamp_millis(p)
            if t is not None:
                post_ts.append(t)
        last_post_timestamp_ms = max(post_ts) if post_ts else None
        
        trending_topics.append({
            "topic": topic,
            "cluster_id": f"cluster_{cluster_id}",
            "post_count": size,
            "engagement_score": 0,  # Not used - topics based on content only
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
    
    if filtered_count > 0:
        logger.info(f"🔍 Filtered out {filtered_count} low-quality clusters (poor content correlation)")
    
    logger.info(f"✅ Identified {len(trending_topics)} trending topics")
    return trending_topics
