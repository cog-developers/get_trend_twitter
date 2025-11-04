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

GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3")

# clustering knobs
MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 5
MERGE_COSINE_THRESHOLD = 0.92
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

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)

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
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [w for w in text.split() if len(w) > 2 and w not in ARABIC_STOPWORDS]
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
    print("📉 Reducing dimensionality with PCA...")
    n_comp = min(target_dim, embeddings.shape[1])
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(embeddings)
    print(f"✅ Reduced to {reduced.shape[1]} dims")
    return reduced, pca


def hdbscan_cluster(embeddings: np.ndarray) -> np.ndarray:
    print("🤖 Trying HDBSCAN first (variable density clustering)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=False
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set([l for l in labels if l != -1]))
    noise = np.sum(labels == -1)
    print(f"   • HDBSCAN produced {n_clusters} clusters (+ {noise} noise)")
    return labels


def dbscan_fallback(embeddings: np.ndarray, min_samples: int = 10) -> np.ndarray:
    print("🔁 Falling back to DBSCAN...")
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
    if len(cluster_embeds) < 2:
        return 0.0, 0.0
    centroid = cluster_embeds.mean(axis=0, keepdims=True)
    sims = cosine_similarity(cluster_embeds, centroid).flatten()
    return float(np.mean(sims)), float(np.percentile(sims, 10))


def build_raw_clusters(docs: List[Dict], embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
    clusters: Dict[int, Dict] = {}
    for idx, (doc, lab) in enumerate(zip(docs, labels)):
        if lab == -1:
            continue
        clusters.setdefault(lab, {"members": [], "indices": []})
        clusters[lab]["members"].append(doc)
        clusters[lab]["indices"].append(idx)
    for lab, data in clusters.items():
        vecs = embeddings[data["indices"], :]
        clusters[lab]["centroid"] = vecs.mean(axis=0, keepdims=True)
        clusters[lab]["embeds"] = vecs
    return clusters


def filter_low_quality_clusters(raw_clusters: Dict[int, Dict]) -> Dict[int, Dict]:
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
    if not clusters:
        return clusters
    labels = list(clusters.keys())
    centroids = np.vstack([clusters[l]["centroid"] for l in labels])
    sim_matrix = cosine_similarity(centroids)

    parent = {l: l for l in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if sim_matrix[i, j] >= MERGE_COSINE_THRESHOLD:
                union(labels[i], labels[j])

    grouped = {}
    for l in labels:
        root = find(l)
        grouped.setdefault(root, {"members": [], "indices": [], "embeds": []})
        grouped[root]["members"].extend(clusters[l]["members"])
        grouped[root]["indices"].extend(clusters[l]["indices"])
        grouped[root]["embeds"].append(clusters[l]["embeds"])

    merged_clusters = {}
    new_id = 0
    for root, data in grouped.items():
        embeds_concat = np.vstack(data["embeds"])
        merged_clusters[new_id] = {
            "members": data["members"],
            "indices": data["indices"],
            "embeds": embeds_concat,
            "centroid": embeds_concat.mean(axis=0, keepdims=True)
        }
        new_id += 1

    print(f"🔗 Merged {len(labels)} → {len(merged_clusters)} clusters after similarity check")
    return merged_clusters


def recluster_noise(docs: List[Dict], embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, Dict]:
    noise_mask = labels == -1
    if np.sum(noise_mask) < MIN_CLUSTER_SIZE:
        return {}
    noise_docs = [d for d, m in zip(docs, noise_mask) if m]
    noise_embeds = embeddings[noise_mask]

    print(f"🗑 Trying to salvage noise: {len(noise_docs)} posts...")
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
    return sub_clusters


def select_representative_texts(members: List[Dict], embeds: np.ndarray, centroid: np.ndarray, top_k: int = 12) -> List[str]:
    sims = cosine_similarity(embeds, centroid).flatten()
    top_idx = np.argsort(-sims)[:top_k]
    return [members[i]["text"] for i in top_idx]


GROK_API_KEY = os.getenv("XAI_API_KEY").strip()
GROK_MODEL = "grok-4"

def get_cluster_topic(cluster_members: List[Dict], cluster_embeds: np.ndarray, centroid: np.ndarray) -> str:
    if not cluster_members:
        return "موضوع غير محدد"

    # Prepare sample posts
    sample = "\n---\n".join([m.get("text", "") for m in cluster_members[:5]])

    prompt = f"""أنت محلل خبير للمحتوى العربي على وسائل التواصل الاجتماعي.
مهمتك: أعطني عنواناً قصيراً يصف الموضوع الأساسي المشترك بين هذه المنشورات. ركّز على الحدث/القضية/الشخص/المكان، بدون شرح إضافي.
المنشورات:
{sample}

العنوان فقط بدون أي شرح:"""

    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": "You are Grok, a highly intelligent, helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    r = None  # ✅ ensure defined before try/except

    try:
        r = requests.post("http://185.217.126.143/v1/chat/completions", headers=headers, json=payload, timeout=60)
        r.raise_for_status()

        data = r.json()
        if not data or "choices" not in data or not data["choices"]:
            print("⚠️ Invalid or empty response from Grok:", data)
            return "موضوع غير محدد"

        topic = data["choices"][0]["message"]["content"].strip()
        topic = re.sub(r'^[\d\.\-\s:"«»]+', '', topic)
        topic = re.sub(r'[\d\.\-\s:"«»]+$', '', topic)
        topic = topic.split("\n")[0].strip().strip('"').strip("'").strip("«").strip("»")

        return topic if topic else "موضوع غير محدد"

    except Exception as e:
        print(f"⚠️ Grok API error: {e}")
        if r is not None:
            print("Response text:", getattr(r, "text", "no response"))
        else:
            print("⚠️ No HTTP response received (request likely failed before connection).")
        return "موضوع غير محدد"
    
def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
    words = []
    for text in texts:
        words.extend([w for w in clean_text(text).split() if w not in ARABIC_STOPWORDS and len(w) > 2])
    counter = Counter(words)
    return [w for w, _ in counter.most_common(top_n)]


def update_clusters_in_opensearch(final_clusters: Dict[int, List[Dict]], docs: List[Dict], embeddings_reduced: np.ndarray, merged_struct: Dict[int, Dict]):
    actions = []
    total = 0
    clustered_ids = set()

    print("\n" + "="*70)
    print("📊 CLUSTER SUMMARY")
    print("="*70)

    cluster_info = {cid: {"members": d["members"], "embeds": d["embeds"], "centroid": d["centroid"]} for cid, d in merged_struct.items()}
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

    for doc in docs:
        if doc["id"] not in clustered_ids:
            solo_keywords = extract_keywords([doc["text"]], top_n=3)
            actions.append({
                "_op_type": "update",  "_index": SOURCE_INDEX,
                "_id": doc["id"],
                "doc": {
                    "group_topic": "موضوع منفصل",
                    "cluster_id": f"solo_{doc['id']}",
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

    print(f"\n✅ Indexed/updated {total} documents in OpenSearch")


# ====== MAIN PIPELINE ======
def main():
    print("🚀 Starting clustering pipeline with Grok...")

    ensure_cluster_fields_mapping(SOURCE_INDEX)

    # Step 1: Fetch
    docs = fetch_all_docs()
    if not docs:
        print("⚠️ No documents found.")
        return

    # Step 2: Vectorize
    embeddings = vectorize_docs(docs)

    # Step 3: PCA
    embeddings_reduced, _ = reduce_dimensionality(embeddings)

    # Step 4: Clustering (HDBSCAN → fallback DBSCAN)
    labels = hdbscan_cluster(embeddings_reduced)
    n_clusters = len(set([l for l in labels if l != -1]))
    if n_clusters == 0:
        print("⚠️ No clusters found via HDBSCAN, trying DBSCAN fallback.")
        labels = dbscan_fallback(embeddings_reduced)

    # Step 5: Raw clustering
    raw_clusters = build_raw_clusters(docs, embeddings_reduced, labels)
    print(f"🧩 Raw clusters count: {len(raw_clusters)}")

    # Step 6: Filter coherence
    filtered_clusters = filter_low_quality_clusters(raw_clusters)
    print(f"✅ Retained {len(filtered_clusters)} coherent clusters")

    # Step 7: Merge similar clusters
    merged_clusters = merge_similar_clusters(filtered_clusters)

    # Step 8: Re-cluster noise
    print("♻️ Checking for noise re-clustering...")
    noise_clusters = recluster_noise(docs, embeddings_reduced, labels)
    all_clusters = {**merged_clusters, **noise_clusters}
    print(f"✅ Final cluster count: {len(all_clusters)}")

    # Step 9: Save to OpenSearch
    update_clusters_in_opensearch(all_clusters, docs, embeddings_reduced, merged_clusters)

    print("\n🏁 All done!")


# ====== ENTRYPOINT ======
if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        print(f"\n⏱️ Total runtime: {time.time() - t0:.2f} seconds")