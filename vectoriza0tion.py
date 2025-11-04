import os
import re
import time
import requests
import numpy as np
from dotenv import load_dotenv
from collections import Counter
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import urllib3
from typing import List, Dict, Tuple

# ====== LOAD ENV ======
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== CONFIG ======
ELASTICSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
ELASTICSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "searched-tweets-index")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# clustering knobs
MIN_CLUSTER_SIZE = 10          # minimum posts per final cluster you accept
HDBSCAN_MIN_CLUSTER_SIZE = 8   # lower: allows small organic topics
HDBSCAN_MIN_SAMPLES = 5        # higher = stricter, lower = more inclusive
MERGE_COSINE_THRESHOLD = 0.92  # if two cluster centroids are > this, merge
COHERENCE_MEAN_THRESHOLD = 0.62
COHERENCE_P10_THRESHOLD = 0.45

ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "إلى", "مع", "عن",
    "ما", "لا", "هذا", "هذه", "هو", "هي"
}

# ====== VALIDATE ENV ======
if not all([ELASTICSEARCH_NODE, ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD]):
    raise ValueError("❌ OpenSearch credentials are missing in .env file.")

# ====== INIT OPENSEARCH ======
es = OpenSearch(
    [ELASTICSEARCH_NODE],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    timeout=90
)

try:
    if not es.ping():
        raise ConnectionError("❌ Cannot connect to OpenSearch.")
    print("✅ Connected to OpenSearch successfully.")
except Exception as e:
    raise ConnectionError(f"❌ OpenSearch connection failed: {e}")


# ====== UTILITIES ======
def ensure_cluster_fields_mapping(index_name: str):
    try:
        current = es.indices.get_mapping(index=index_name)
        desired_props = {
            "group_topic": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "cluster_id": {"type": "keyword"},
            "cluster_size": {"type": "integer"},
            "cluster_keywords": {"type": "keyword"},
            "fetched_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
            "updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
        }

        existing_props = current.get(index_name, {}).get("mappings", {}).get("properties", {})
        missing = {k: v for k, v in desired_props.items() if k not in existing_props}

        if not missing:
            print("ℹ️ Mapping already up to date.")
            return

        es.indices.put_mapping(index=index_name, body={"properties": missing})
        print("✅ Mapping updated successfully.")
    except Exception as e:
        print("❌ Failed to update mapping:", e)


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    # URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)

    # Arabic normalize
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)   # diacritics
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)

    # strip emoji / pictos
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # keep only Arabic letters, numbers, _
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [
        w for w in text.split()
        if len(w) > 2 and w not in ARABIC_STOPWORDS
    ]
    return " ".join(words).strip()


def fetch_all_docs() -> List[Dict]:
    from opensearchpy.helpers import scan
    query = {"query": {"match_all": {}}, "_source": ["topic", "post_text", "core_highlight", "text", "author", "created_at"]}
    docs = []
    for d in scan(es, query=query, index=SOURCE_INDEX, size=500, scroll="10m"):
        src = d["_source"]
        text = src.get("topic") or src.get("post_text") or src.get("core_highlight") or src.get("text")
        if not text or len(text.strip()) < 10:
            continue
        cleaned = clean_text(text)
        if cleaned and len(cleaned) > 10:
            docs.append({
                "id": d["_id"],
                "text": text.strip(),
                "cleaned": cleaned,
                "author": src.get("author"),
                "created_at": src.get("created_at"),
            })
    print(f"✅ Loaded {len(docs)} valid docs.")
    return docs


def vectorize_docs(docs: List[Dict]) -> np.ndarray:
    print("⚙️ Encoding posts...")
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    texts = [d["cleaned"] for d in docs]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )
    return embeddings


def reduce_dimensionality(embeddings: np.ndarray, target_dim: int = 100) -> Tuple[np.ndarray, PCA]:
    """PCA keeps global geometry smoother than UMAP for clustering stability."""
    print("📉 Reducing dimensionality with PCA...")
    n_comp = min(target_dim, embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(embeddings)
    print(f"✅ Reduced to {reduced.shape[1]} dims")
    return reduced, pca


# ---------- CLUSTERING CORE ----------

def hdbscan_cluster(embeddings: np.ndarray) -> np.ndarray:
    """
    Returns labels from HDBSCAN. -1 means noise.
    """
    print("🤖 Trying HDBSCAN first (variable density clustering)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean',  # PCA output is Euclidean space
        cluster_selection_method='eom',
        prediction_data=False
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set([l for l in labels if l != -1]))
    noise = np.sum(labels == -1)
    print(f"   • HDBSCAN produced {n_clusters} clusters (+ {noise} noise)")
    return labels


def dbscan_fallback(embeddings: np.ndarray, min_samples: int = 10) -> np.ndarray:
    """
    Fallback if HDBSCAN gives nothing useful.
    We'll estimate eps based on k-distances.
    """
    print("🔁 Falling back to DBSCAN...")
    # estimate eps dynamically
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric='cosine', n_jobs=-1)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    k_distances = np.sort(distances[:, -1])
    mean_d, std_d = np.mean(k_distances), np.std(k_distances)
    eps = float(np.clip(mean_d + 0.5 * std_d, 0.05, 0.30))
    print(f"   • Adaptive eps = {eps:.4f}")

    from sklearn.cluster import DBSCAN
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', n_jobs=-1)
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set([l for l in labels if l != -1]))
    noise = np.sum(labels == -1)
    print(f"   • DBSCAN produced {n_clusters} clusters (+ {noise} noise)")
    return labels


def coherence_scores(cluster_embeds: np.ndarray) -> Tuple[float, float]:
    """
    mean_sim: avg cosine similarity to centroid
    p10_sim: 10th percentile similarity (are there totally off-topic posts?)
    """
    if len(cluster_embeds) < 2:
        return 0.0, 0.0
    centroid = cluster_embeds.mean(axis=0, keepdims=True)
    sims = cosine_similarity(cluster_embeds, centroid).flatten()
    mean_sim = float(np.mean(sims))
    p10_sim = float(np.percentile(sims, 10))
    return mean_sim, p10_sim


def build_raw_clusters(docs: List[Dict], embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
    """
    returns:
    {
      cluster_label: {
        "members": [doc_obj, ...],
        "indices": [int, ...],           # index in docs / embeddings
        "centroid": np.ndarray(1,d)
      },
      ...
    }
    """
    clusters: Dict[int, Dict] = {}
    for idx, (doc, lab) in enumerate(zip(docs, labels)):
        if lab == -1:
            continue
        if lab not in clusters:
            clusters[lab] = {"members": [], "indices": []}
        clusters[lab]["members"].append(doc)
        clusters[lab]["indices"].append(idx)

    # compute centroid for each cluster
    for lab, data in clusters.items():
        vecs = embeddings[data["indices"], :]
        centroid = vecs.mean(axis=0, keepdims=True)
        clusters[lab]["centroid"] = centroid
        clusters[lab]["embeds"] = vecs
    return clusters


def filter_low_quality_clusters(raw_clusters: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Drop clusters that are too small or incoherent.
    """
    filtered = {}
    for lab, data in raw_clusters.items():
        size = len(data["members"])
        if size < MIN_CLUSTER_SIZE:
            continue

        mean_sim, p10_sim = coherence_scores(data["embeds"])
        data["coherence_mean"] = mean_sim
        data["coherence_p10"] = p10_sim

        if (mean_sim >= COHERENCE_MEAN_THRESHOLD) or (p10_sim >= COHERENCE_P10_THRESHOLD):
            filtered[lab] = data
            print(f"   ✓ cluster {lab}: size={size}, mean={mean_sim:.3f}, p10={p10_sim:.3f}")
        else:
            print(f"   ✗ drop {lab}: size={size}, mean={mean_sim:.3f}, p10={p10_sim:.3f}")
    return filtered


def merge_similar_clusters(clusters: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    If two clusters have almost identical centroids (cosine > MERGE_COSINE_THRESHOLD),
    merge them into one.
    """
    if not clusters:
        return clusters

    labels = list(clusters.keys())
    centroids = np.vstack([clusters[l]["centroid"] for l in labels])
    sim_matrix = cosine_similarity(centroids)

    parent = {l: l for l in labels}  # union-find representative

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # union clusters that are "almost same topic"
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if sim_matrix[i, j] >= MERGE_COSINE_THRESHOLD:
                union(labels[i], labels[j])

    # group by representative
    grouped = {}
    for l in labels:
        root = find(l)
        grouped.setdefault(root, {"members": [], "indices": [], "embeds": []})

        grouped[root]["members"].extend(clusters[l]["members"])
        grouped[root]["indices"].extend(clusters[l]["indices"])
        grouped[root]["embeds"].append(clusters[l]["embeds"])

    # finalize centroid + embeds
    merged_clusters = {}
    new_id = 0
    for root, data in grouped.items():
        embeds_concat = np.vstack(data["embeds"])
        centroid = embeds_concat.mean(axis=0, keepdims=True)
        merged_clusters[new_id] = {
            "members": data["members"],
            "indices": data["indices"],
            "embeds": embeds_concat,
            "centroid": centroid
        }
        new_id += 1

    print(f"🔗 Merged {len(labels)} → {len(merged_clusters)} clusters after similarity check")
    return merged_clusters


def recluster_noise(docs: List[Dict], embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
    """
    Take noise points (-1) and try a looser DBSCAN to rescue mini-topics.
    """
    noise_mask = labels == -1
    if np.sum(noise_mask) < MIN_CLUSTER_SIZE:
        return {}

    noise_docs = [d for d, m in zip(docs, noise_mask) if m]
    noise_embeds = embeddings[noise_mask]

    print(f"🗑 Trying to salvage noise: {len(noise_docs)} posts...")

    # very loose DBSCAN on noise only
    nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
    nbrs.fit(noise_embeds)
    distances, _ = nbrs.kneighbors(noise_embeds)
    k_distances = np.sort(distances[:, -1])
    mean_d, std_d = np.mean(k_distances), np.std(k_distances)
    eps = float(np.clip(mean_d + 0.3 * std_d, 0.07, 0.35))
    print(f"   • Salvage eps={eps:.4f}")

    from sklearn.cluster import DBSCAN
    sub = DBSCAN(eps=eps, min_samples=5, metric='cosine', n_jobs=-1)
    sub_labels = sub.fit_predict(noise_embeds)

    sub_clusters = build_raw_clusters(noise_docs, noise_embeds, sub_labels)
    sub_clusters = filter_low_quality_clusters(sub_clusters)

    # reindex their indices to the global docs list:
    # we lost original idx mapping, but content is what matters downstream (OpenSearch updates by _id)
    return sub_clusters


def cluster_all(docs: List[Dict], embeddings_reduced: np.ndarray) -> Dict[int, Dict]:
    # 1. try HDBSCAN
    labels = hdbscan_cluster(embeddings_reduced)

    # if HDBSCAN gave 0 or 1 clusters, fallback to DBSCAN
    unique_clusters = [c for c in set(labels) if c != -1]
    if len(unique_clusters) <= 1:
        print("⚠️ HDBSCAN not informative, using DBSCAN fallback")
        labels = dbscan_fallback(embeddings_reduced)

    # 2. build clusters
    raw_clusters = build_raw_clusters(docs, embeddings_reduced, labels)

    # 3. filter by coherence / size
    filtered_clusters = filter_low_quality_clusters(raw_clusters)

    # 4. salvage noise
    rescued = recluster_noise(docs, embeddings_reduced, labels)

    # merge main + rescued
    combined = {**filtered_clusters}
    offset = len(combined)
    for k, v in rescued.items():
        combined[offset + k] = v

    # 5. merge near-duplicate clusters
    merged = merge_similar_clusters(combined)

    # final: return as {cluster_id_int: [doc, ...]}
    final_clusters = {}
    for cid, data in merged.items():
        final_clusters[cid] = data["members"]

    print(f"🏁 Final usable clusters: {len(final_clusters)}")
    return final_clusters


# ---------- TOPIC LABELLING / KEYWORDS / INDEXING ----------

def select_representative_texts(members: List[Dict], embeds: np.ndarray, centroid: np.ndarray, top_k: int = 12) -> List[str]:
    """
    Pick posts closest to centroid (most 'on-topic') instead of random slice.
    """
    sims = cosine_similarity(embeds, centroid).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [members[i]["text"] for i in top_idx]


def get_cluster_topic(cluster_members: List[Dict], cluster_embeds: np.ndarray, centroid: np.ndarray) -> str:
    if not cluster_members:
        return "موضوع غير محدد"

    reps = select_representative_texts(cluster_members, cluster_embeds, centroid, top_k=12)
    sample = "\n---\n".join(reps)

    prompt = f"""أنت محلل خبير للمحتوى العربي على وسائل التواصل الاجتماعي.
مهمتك: أعطني عنواناً قصيراً وواضحاً يصف الموضوع الأساسي المشترك بين هذه المنشورات. 
العنوان يجب أن يكون جملة كاملة ومفهومة تلخص الموضوع بشكل دقيق.

المنشورات:
{sample}

اكتب العنوان فقط كجملة كاملة ومفيدة:"""

    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 100  # Increased to allow complete sentences
    }
    try:
        r = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        topic = r.json()["choices"][0]["message"]["content"].strip()
        
        # Only remove leading numbers/bullets, but keep the complete sentence
        topic = re.sub(r'^[\d\.\-\s:]+', '', topic)
        
        # Remove surrounding quotes if present
        topic = topic.strip('"').strip("'").strip("«").strip("»")
        
        # Take only the first line if multiple lines returned
        topic = topic.split("\n")[0].strip()
        
        if not topic or len(topic) < 5:
            return "موضوع غير محدد"
        
        return topic
        
    except Exception as e:
        print(f"⚠️ Error getting topic from API: {e}")
        return "موضوع غير محدد"
def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    words = []
    for text in texts:
        words.extend([
            w for w in clean_text(text).split()
            if w not in ARABIC_STOPWORDS and len(w) > 2
        ])
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def update_clusters_in_opensearch(final_clusters: Dict[int, List[Dict]], docs: List[Dict], embeddings_reduced: np.ndarray, merged_struct: Dict[int, Dict]):
    """
    merged_struct is the dict from merge_similar_clusters BEFORE we flattened to final_clusters.
    We need it here to compute centroids + pick reps.
    We'll reconstruct a parallel mapping cid -> (members, embeds, centroid).
    """
    actions = []
    total = 0
    clustered_ids = set()

    print("\n" + "="*70)
    print("📊 CLUSTER SUMMARY")
    print("="*70)

    # build helper map for cid -> info
    cluster_info = {}
    for cid, data in merged_struct.items():
        cluster_info[cid] = {
            "members": data["members"],
            "embeds": data["embeds"],
            "centroid": data["centroid"]
        }

    # sort clusters by size desc
    cluster_items_sorted = sorted(cluster_info.items(), key=lambda x: len(x[1]["members"]), reverse=True)

    for i, (cid, cdata) in enumerate(cluster_items_sorted):
        members = cdata["members"]
        embeds_local = cdata["embeds"]
        centroid = cdata["centroid"]

        cluster_texts = [m["text"] for m in members]
        topic = get_cluster_topic(members, embeds_local, centroid)
        keywords = extract_keywords(cluster_texts, top_n=5)
        size = len(members)

        print(f" • group_{i}: size={size}, topic={topic[:60]}... keywords={keywords}")

        for m in members:
            actions.append({
                "_op_type": "update",
                "_index": SOURCE_INDEX,
                "_id": m["id"],
                "doc": {
                    "group_topic": topic,
                    "cluster_id": f"group_{i}",
                    "cluster_size": size,
                    "cluster_keywords": keywords
                }
            })
            clustered_ids.add(m["id"])
            total += 1

            if len(actions) >= 300:
                helpers.bulk(es, actions, request_timeout=90)
                actions = []

    # fallback for singletons / uncaptured
    for doc in docs:
        if doc["id"] not in clustered_ids:
            solo_keywords = extract_keywords([doc["text"]], top_n=3)
            actions.append({
                "_op_type": "update",
                "_index": SOURCE_INDEX,
                "_id": doc["id"],
                "doc": {
                    "group_topic": doc["text"][:80],
                    "cluster_id": None,
                    "cluster_size": 1,
                    "cluster_keywords": solo_keywords
                }
            })
            total += 1

            if len(actions) >= 300:
                helpers.bulk(es, actions, request_timeout=90)
                actions = []

    if actions:
        helpers.bulk(es, actions, request_timeout=90)

    print(f"\n✅ Total updated: {total} docs")


# ====== MAIN PIPELINE ======
if __name__ == "__main__":
    print("🎯 TREND CLUSTERING PIPELINE v4.0")

    # Step 1: load
    print("📥 Step 1: Fetching documents...")
    docs = fetch_all_docs()
    if len(docs) < MIN_CLUSTER_SIZE:
        print("❌ Not enough docs.")
        raise SystemExit(1)

    # Step 2: embed
    print("🔢 Step 2: Vectorizing...")
    embeddings = vectorize_docs(docs)
    print(f"✅ Created {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Step 3: reduce
    embeddings_reduced, pca_model = reduce_dimensionality(embeddings, target_dim=100)

    # Step 4: cluster
    print("🎯 Step 4: Clustering with adaptive pipeline...")
    # cluster_all returns final_clusters (dict cid -> [docs])
    # but we also need the merged_struct (before flatten) for centroids/embeds,
    # so let's slightly refactor to keep both.

    # We'll inline the logic from cluster_all here to keep merged_struct:
    labels = hdbscan_cluster(embeddings_reduced)
    unique_clusters = [c for c in set(labels) if c != -1]
    if len(unique_clusters) <= 1:
        print("⚠️ HDBSCAN not informative, using DBSCAN fallback")
        labels = dbscan_fallback(embeddings_reduced)

    raw_clusters = build_raw_clusters(docs, embeddings_reduced, labels)
    filtered_clusters = filter_low_quality_clusters(raw_clusters)
    rescued = recluster_noise(docs, embeddings_reduced, labels)

    combined = {**filtered_clusters}
    offset = len(combined)
    for k, v in rescued.items():
        combined[offset + k] = v

    merged_struct = merge_similar_clusters(combined)

    final_clusters = {cid: data["members"] for cid, data in merged_struct.items()}

    if not final_clusters:
        print("❌ No valid clusters found.")
        raise SystemExit(1)

    print(f"✅ Final clusters: {len(final_clusters)} groups")

    # Step 5: index back
    print("📤 Step 5: Updating OpenSearch...")
    ensure_cluster_fields_mapping(SOURCE_INDEX)
    update_clusters_in_opensearch(final_clusters, docs, embeddings_reduced, merged_struct)

    print("🎉 Done. All clusters labeled and indexed.")
