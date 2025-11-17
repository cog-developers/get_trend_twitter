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
from sklearn.cluster import DBSCAN
import hdbscan
import urllib3
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== LOAD ENV ======
load_dotenv()

# ====== CONFIG ======
@dataclass
class Config:
    """Centralized configuration management"""
    # OpenSearch
    opensearch_node: str
    opensearch_username: str
    opensearch_password: str
    source_index: str
    
    # DeepSeek API
    deepseek_api_key: str
    deepseek_model: str
    deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions"
    
    # Clustering parameters
    min_cluster_size: int = 10
    hdbscan_min_cluster_size: int = 8
    hdbscan_min_samples: int = 5
    merge_cosine_threshold: float = 0.92
    coherence_mean_threshold: float = 0.62
    coherence_p10_threshold: float = 0.45
    
    # Processing parameters
    pca_target_dim: int = 100
    batch_size: int = 300
    embedding_batch_size: int = 32
    
    # Text processing
    min_text_length: int = 10
    min_word_length: int = 2
    top_keywords: int = 5
    representative_texts: int = 12
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            opensearch_node=os.getenv("OPENSEARCH_NODE"),
            opensearch_username=os.getenv("OPENSEARCH_USERNAME"),
            opensearch_password=os.getenv("OPENSEARCH_PASSWORD"),
            source_index=os.getenv("OPENSEARCH_INDEX", "searched-tweets-index"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            min_cluster_size=int(os.getenv("MIN_CLUSTER_SIZE", "10")),
            hdbscan_min_cluster_size=int(os.getenv("HDBSCAN_MIN_CLUSTER_SIZE", "8")),
            hdbscan_min_samples=int(os.getenv("HDBSCAN_MIN_SAMPLES", "5")),
            merge_cosine_threshold=float(os.getenv("MERGE_COSINE_THRESHOLD", "0.92")),
            pca_target_dim=int(os.getenv("PCA_TARGET_DIM", "100")),
        )
    
    def validate(self):
        """Validate required configuration"""
        required = {
            "opensearch_node": self.opensearch_node,
            "opensearch_username": self.opensearch_username,
            "opensearch_password": self.opensearch_password,
            "deepseek_api_key": self.deepseek_api_key
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"❌ Missing required config: {', '.join(missing)}")
        
        logger.info("✅ Configuration validated")


ARABIC_STOPWORDS = {
    "و", "في", "من", "على", "إلى", "مع", "عن",
    "ما", "لا", "هذا", "هذه", "هو", "هي", "أن", 
    "إن", "كان", "كل", "قد", "لم", "لن", "ثم", 
    "أو", "بل", "لكن", "حتى", "عند", "بعد", "قبل"
}


# ====== OPENSEARCH CLIENT ======
class OpenSearchClient:
    """Wrapper for OpenSearch operations with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = self._create_client()
    
    def _create_client(self) -> OpenSearch:
        """Create and test OpenSearch connection"""
        try:
            client = OpenSearch(
                [self.config.opensearch_node],
                http_auth=(self.config.opensearch_username, self.config.opensearch_password),
                verify_certs=False,
                timeout=90,
                max_retries=3,
                retry_on_timeout=True
            )
            
            if not client.ping():
                raise ConnectionError("Cannot ping OpenSearch")
            
            logger.info("✅ Connected to OpenSearch successfully")
            return client
            
        except Exception as e:
            logger.error(f"❌ OpenSearch connection failed: {e}")
            raise
    
    def ensure_cluster_fields_mapping(self):
        """Ensure required fields exist in index mapping"""
        try:
            current = self.client.indices.get_mapping(index=self.config.source_index)
            desired_props = {
                "group_topic": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "cluster_id": {"type": "keyword"},
                "cluster_size": {"type": "integer"},
                "cluster_keywords": {"type": "keyword"},
                "fetched_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                },
                "updated_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||epoch_millis"
                }
            }

            existing_props = current.get(
                self.config.source_index, {}
            ).get("mappings", {}).get("properties", {})
            
            missing = {k: v for k, v in desired_props.items() if k not in existing_props}

            if not missing:
                logger.info("ℹ️ Mapping already up to date")
                return

            self.client.indices.put_mapping(
                index=self.config.source_index,
                body={"properties": missing}
            )
            logger.info("✅ Mapping updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to update mapping: {e}")
            raise
    
    def fetch_all_docs(self) -> List[Dict]:
        """Fetch all documents from OpenSearch"""
        logger.info("📥 Fetching documents from OpenSearch...")
        
        query = {
            "query": {"match_all": {}},
            "_source": ["topic", "post_text", "core_highlight", "text", "author", "created_at"]
        }
        
        docs = []
        try:
            for d in helpers.scan(
                self.client,
                query=query,
                index=self.config.source_index,
                size=500,
                scroll="10m"
            ):
                src = d["_source"]
                text = (
                    src.get("topic") or 
                    src.get("post_text") or 
                    src.get("core_highlight") or 
                    src.get("text")
                )
                
                if not text or len(text.strip()) < self.config.min_text_length:
                    continue
                
                cleaned = TextProcessor.clean_text(text)
                if cleaned and len(cleaned) > self.config.min_text_length:
                    docs.append({
                        "id": d["_id"],
                        "text": text.strip(),
                        "cleaned": cleaned,
                        "author": src.get("author"),
                        "created_at": src.get("created_at"),
                    })
            
            logger.info(f"✅ Loaded {len(docs)} valid documents")
            return docs
            
        except Exception as e:
            logger.error(f"❌ Error fetching documents: {e}")
            raise
    
    def bulk_update(self, actions: List[Dict]):
        """Perform bulk update with error handling"""
        if not actions:
            return 0, []
        
        try:
            success, errors = helpers.bulk(
                self.client,
                actions,
                request_timeout=90,
                raise_on_error=False,
                raise_on_exception=False
            )
            
            if errors:
                logger.warning(f"⚠️ {len(errors)} errors during bulk update")
            
            return success, errors
            
        except Exception as e:
            logger.error(f"❌ Bulk update failed: {e}")
            raise
    
    def close(self):
        """Close OpenSearch connection"""
        try:
            self.client.close()
            logger.info("🔌 OpenSearch connection closed")
        except Exception as e:
            logger.warning(f"⚠️ Error closing connection: {e}")


# ====== TEXT PROCESSING ======
class TextProcessor:
    """Text cleaning and normalization utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize Arabic text"""
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs, mentions, hashtags
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"@\w+", " ", text)
        text = re.sub(r"#\w+", " ", text)

        # Arabic normalization
        text = re.sub(r"[ًٌٍَُِّْـ]", "", text)  # diacritics
        text = re.sub(r"[إأآ]", "ا", text)
        text = re.sub(r"ى", "ي", text)

        # Remove emojis
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

        # Keep only Arabic letters, numbers, underscore
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Remove stopwords and short words
        words = [
            w for w in text.split()
            if len(w) > 2 and w not in ARABIC_STOPWORDS
        ]
        
        return " ".join(words).strip()
    
    @staticmethod
    def extract_keywords(texts: List[str], top_n: int = 5) -> List[str]:
        """Extract most common keywords from texts"""
        words = []
        for text in texts:
            words.extend([
                w for w in TextProcessor.clean_text(text).split()
                if w not in ARABIC_STOPWORDS and len(w) > 2
            ])
        
        counter = Counter(words)
        return [w for w, _ in counter.most_common(top_n)]


# ====== EMBEDDING & DIMENSIONALITY REDUCTION ======
class EmbeddingProcessor:
    """Handle text embeddings and dimensionality reduction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.pca_model = None
    
    def load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        if self.model is None:
            logger.info("⚙️ Loading sentence transformer model...")
            self.model = SentenceTransformer(
                'sentence-transformers/distiluse-base-multilingual-cased-v2'
            )
            logger.info("✅ Model loaded")
        return self.model
    
    def vectorize_docs(self, docs: List[Dict]) -> np.ndarray:
        """Convert documents to embeddings"""
        logger.info(f"🔢 Encoding {len(docs)} documents...")
        
        model = self.load_model()
        texts = [d["cleaned"] for d in docs]
        
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=self.config.embedding_batch_size,
                normalize_embeddings=True
            )
            
            logger.info(f"✅ Created embeddings: shape={embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings: {e}")
            raise
    
    def reduce_dimensionality(self, embeddings: np.ndarray) -> Tuple[np.ndarray, PCA]:
        """Reduce dimensionality using PCA"""
        logger.info("📉 Reducing dimensionality with PCA...")
        
        try:
            n_comp = min(self.config.pca_target_dim, embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(embeddings)
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            logger.info(
                f"✅ Reduced to {reduced.shape[1]} dims "
                f"(variance explained: {variance_explained:.2%})"
            )
            
            self.pca_model = pca
            return reduced, pca
            
        except Exception as e:
            logger.error(f"❌ Error in dimensionality reduction: {e}")
            raise


# ====== CLUSTERING ENGINE ======
class ClusteringEngine:
    """Advanced clustering with quality control"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def hdbscan_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering"""
        logger.info("🤖 Running HDBSCAN clustering...")
        
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=False
            )
            
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set([l for l in labels if l != -1]))
            noise = np.sum(labels == -1)
            
            logger.info(f"   • HDBSCAN: {n_clusters} clusters, {noise} noise points")
            return labels
            
        except Exception as e:
            logger.error(f"❌ HDBSCAN failed: {e}")
            raise
    
    def dbscan_fallback(self, embeddings: np.ndarray, min_samples: int = 10) -> np.ndarray:
        """Fallback DBSCAN with adaptive epsilon"""
        logger.info("🔁 Falling back to DBSCAN...")
        
        try:
            # Estimate epsilon dynamically
            nbrs = NearestNeighbors(
                n_neighbors=min_samples,
                metric='cosine',
                n_jobs=-1
            )
            nbrs.fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            k_distances = np.sort(distances[:, -1])
            
            mean_d, std_d = np.mean(k_distances), np.std(k_distances)
            eps = float(np.clip(mean_d + 0.5 * std_d, 0.05, 0.30))
            
            logger.info(f"   • Adaptive eps = {eps:.4f}")

            clusterer = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='cosine',
                n_jobs=-1
            )
            labels = clusterer.fit_predict(embeddings)
            
            n_clusters = len(set([l for l in labels if l != -1]))
            noise = np.sum(labels == -1)
            
            logger.info(f"   • DBSCAN: {n_clusters} clusters, {noise} noise points")
            return labels
            
        except Exception as e:
            logger.error(f"❌ DBSCAN failed: {e}")
            raise
    
    @staticmethod
    def coherence_scores(cluster_embeds: np.ndarray) -> Tuple[float, float]:
        """Calculate cluster coherence metrics"""
        if len(cluster_embeds) < 2:
            return 0.0, 0.0
        
        centroid = cluster_embeds.mean(axis=0, keepdims=True)
        sims = cosine_similarity(cluster_embeds, centroid).flatten()
        
        mean_sim = float(np.mean(sims))
        p10_sim = float(np.percentile(sims, 10))
        
        return mean_sim, p10_sim
    
    def build_raw_clusters(
        self,
        docs: List[Dict],
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """Build cluster structures from labels"""
        clusters: Dict[int, Dict] = {}
        
        for idx, (doc, lab) in enumerate(zip(docs, labels)):
            if lab == -1:
                continue
            
            if lab not in clusters:
                clusters[lab] = {"members": [], "indices": []}
            
            clusters[lab]["members"].append(doc)
            clusters[lab]["indices"].append(idx)

        # Compute centroids and embeddings
        for lab, data in clusters.items():
            vecs = embeddings[data["indices"], :]
            centroid = vecs.mean(axis=0, keepdims=True)
            clusters[lab]["centroid"] = centroid
            clusters[lab]["embeds"] = vecs
        
        return clusters
    
    def filter_low_quality_clusters(self, raw_clusters: Dict[int, Dict]) -> Dict[int, Dict]:
        """Filter clusters by size and coherence"""
        logger.info("🔍 Filtering low-quality clusters...")
        
        filtered = {}
        dropped = 0
        
        for lab, data in raw_clusters.items():
            size = len(data["members"])
            
            if size < self.config.min_cluster_size:
                dropped += 1
                continue

            mean_sim, p10_sim = self.coherence_scores(data["embeds"])
            data["coherence_mean"] = mean_sim
            data["coherence_p10"] = p10_sim

            is_coherent = (
                mean_sim >= self.config.coherence_mean_threshold or
                p10_sim >= self.config.coherence_p10_threshold
            )
            
            if is_coherent:
                filtered[lab] = data
            else:
                dropped += 1
        
        logger.info(f"   • Kept {len(filtered)}, dropped {dropped} clusters")
        return filtered
    
    def merge_similar_clusters(self, clusters: Dict[int, Dict]) -> Dict[int, Dict]:
        """Merge clusters with similar centroids"""
        if not clusters:
            return clusters

        logger.info("🔗 Merging similar clusters...")
        
        labels = list(clusters.keys())
        centroids = np.vstack([clusters[l]["centroid"] for l in labels])
        sim_matrix = cosine_similarity(centroids)

        # Union-Find for merging
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

        # Union similar clusters
        merge_count = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if sim_matrix[i, j] >= self.config.merge_cosine_threshold:
                    union(labels[i], labels[j])
                    merge_count += 1

        # Group by representative
        grouped = {}
        for l in labels:
            root = find(l)
            if root not in grouped:
                grouped[root] = {"members": [], "indices": [], "embeds": []}

            grouped[root]["members"].extend(clusters[l]["members"])
            grouped[root]["indices"].extend(clusters[l]["indices"])
            grouped[root]["embeds"].append(clusters[l]["embeds"])

        # Finalize merged clusters
        merged_clusters = {}
        for new_id, (root, data) in enumerate(grouped.items()):
            embeds_concat = np.vstack(data["embeds"])
            centroid = embeds_concat.mean(axis=0, keepdims=True)
            
            merged_clusters[new_id] = {
                "members": data["members"],
                "indices": data["indices"],
                "embeds": embeds_concat,
                "centroid": centroid
            }

        logger.info(
            f"   • Merged {len(labels)} → {len(merged_clusters)} clusters"
        )
        return merged_clusters
    
    def recluster_noise(
        self,
        docs: List[Dict],
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """Attempt to cluster noise points"""
        noise_mask = labels == -1
        noise_count = np.sum(noise_mask)
        
        if noise_count < self.config.min_cluster_size:
            logger.info(f"   • Skipping noise reclustering ({noise_count} points)")
            return {}

        logger.info(f"🗑 Attempting to salvage {noise_count} noise points...")

        noise_docs = [d for d, m in zip(docs, noise_mask) if m]
        noise_embeds = embeddings[noise_mask]

        try:
            # Loose DBSCAN on noise
            nbrs = NearestNeighbors(n_neighbors=5, metric='cosine', n_jobs=-1)
            nbrs.fit(noise_embeds)
            distances, _ = nbrs.kneighbors(noise_embeds)
            k_distances = np.sort(distances[:, -1])
            
            mean_d, std_d = np.mean(k_distances), np.std(k_distances)
            eps = float(np.clip(mean_d + 0.3 * std_d, 0.07, 0.35))
            
            logger.info(f"   • Salvage eps={eps:.4f}")

            sub = DBSCAN(eps=eps, min_samples=5, metric='cosine', n_jobs=-1)
            sub_labels = sub.fit_predict(noise_embeds)

            sub_clusters = self.build_raw_clusters(noise_docs, noise_embeds, sub_labels)
            sub_clusters = self.filter_low_quality_clusters(sub_clusters)

            logger.info(f"   • Salvaged {len(sub_clusters)} clusters from noise")
            return sub_clusters
            
        except Exception as e:
            logger.warning(f"⚠️ Noise reclustering failed: {e}")
            return {}
    
    def cluster_all(
        self,
        docs: List[Dict],
        embeddings_reduced: np.ndarray
    ) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]]]:
        """Complete clustering pipeline"""
        logger.info("🎯 Starting clustering pipeline...")
        
        # 1. Initial clustering
        labels = self.hdbscan_cluster(embeddings_reduced)
        
        unique_clusters = [c for c in set(labels) if c != -1]
        if len(unique_clusters) <= 1:
            logger.warning("⚠️ HDBSCAN not informative, using DBSCAN fallback")
            labels = self.dbscan_fallback(embeddings_reduced)

        # 2. Build clusters
        raw_clusters = self.build_raw_clusters(docs, embeddings_reduced, labels)

        # 3. Filter by quality
        filtered_clusters = self.filter_low_quality_clusters(raw_clusters)

        # 4. Salvage noise
        rescued = self.recluster_noise(docs, embeddings_reduced, labels)

        # Combine main + rescued
        combined = {**filtered_clusters}
        offset = len(combined)
        for k, v in rescued.items():
            combined[offset + k] = v

        # 5. Merge similar clusters
        merged_struct = self.merge_similar_clusters(combined)

        # 6. Final clusters
        final_clusters = {
            cid: data["members"]
            for cid, data in merged_struct.items()
        }

        logger.info(f"🏁 Final: {len(final_clusters)} usable clusters")
        
        return merged_struct, final_clusters


# ====== TOPIC GENERATION ======
class TopicGenerator:
    """Generate topics using DeepSeek API"""
    
    def __init__(self, config: Config):
        self.config = config
    
    @staticmethod
    def select_representative_texts(
        members: List[Dict],
        embeds: np.ndarray,
        centroid: np.ndarray,
        top_k: int = 12
    ) -> List[str]:
        """Select most representative texts from cluster"""
        sims = cosine_similarity(embeds, centroid).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        return [members[i]["text"] for i in top_idx]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_api(self, prompt: str) -> str:
        """Call DeepSeek API with retry logic"""
        headers = {
            "Authorization": f"Bearer {self.config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.15,
            "max_tokens": 150
        }
        
        response = requests.post(
            self.config.deepseek_api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"].strip()
    
    def get_cluster_topic(
        self,
        cluster_members: List[Dict],
        cluster_embeds: np.ndarray,
        centroid: np.ndarray
    ) -> str:
        """Generate topic label for cluster"""
        if not cluster_members:
            return "موضوع غير محدد"

        try:
            reps = self.select_representative_texts(
                cluster_members,
                cluster_embeds,
                centroid,
                top_k=self.config.representative_texts
            )
            
            sample = "\n---\n".join(reps)

            prompt = f"""
أنت محلل محتوى اجتماعي محترف ومتخصص في فهم منشورات المستخدمين باللغة العربية.
مهمتك: استخرج **موضوعاً واحداً واضحاً وموجزاً** يصف بدقة محتوى المنشور التالي.
- الموضوع يجب أن يكون **جملة كاملة، مفهومة، دقيقة، وغير عامة**.
- لا تستخدم رموزاً أو اختصارات.
- إذا كان المنشور **غير ذي معنى، فارغ، أو يحتوي على سب أو إهانات أو رموز فقط**، أجب: "غير قابل للتحديد".
- اجعل الموضوع قابل الفهم لأي شخص يقرأه لأول مرة.

المنشورات:
{sample}

اكتب العنوان فقط كجملة كاملة ومفيدة:"""

            topic = self.call_api(prompt)
            
            # Clean topic
            topic = re.sub(r'^[\d\.\-\s:]+', '', topic)
            topic = topic.strip('"').strip("'").strip("«").strip("»")
            topic = topic.split("\n")[0].strip()
            
            if not topic or len(topic) < 5:
                return "موضوع غير محدد"
            
            return topic
            
        except Exception as e:
            logger.error(f"⚠️ Error generating topic: {e}")
            return "موضوع غير محدد"


# ====== INDEXING MANAGER ======
class IndexingManager:
    """Manage OpenSearch indexing operations"""
    
    def __init__(self, os_client: OpenSearchClient, config: Config):
        self.os_client = os_client
        self.config = config
        self.topic_gen = TopicGenerator(config)
    
    def update_clusters_in_opensearch(
        self,
        merged_struct: Dict[int, Dict],
        docs: List[Dict]
    ):
        """Update all documents with cluster information"""
        logger.info("\n" + "="*70)
        logger.info("📊 CLUSTER SUMMARY & INDEXING")
        logger.info("="*70)
        
        actions = []
        clustered_ids = set()
        total_updated = 0

        # Sort by size descending
        cluster_items_sorted = sorted(
            merged_struct.items(),
            key=lambda x: len(x[1]["members"]),
            reverse=True
        )

        for i, (cid, cdata) in enumerate(cluster_items_sorted):
            members = cdata["members"]
            embeds_local = cdata["embeds"]
            centroid = cdata["centroid"]

            cluster_texts = [m["text"] for m in members]
            topic = self.topic_gen.get_cluster_topic(members, embeds_local, centroid)
            keywords = TextProcessor.extract_keywords(cluster_texts, top_n=self.config.top_keywords)
            size = len(members)

            logger.info(f"Cluster {i}: size={size}, topic={topic[:60]}...")

            for m in members:
                actions.append({
                    "_op_type": "update",
                    "_index": self.config.source_index,
                    "_id": m["id"],
                    "doc": {
                        "group_topic": topic,
                        "cluster_id": f"group_{i}",
                        "cluster_size": size,
                        "cluster_keywords": keywords
                    }
                })
                clustered_ids.add(m["id"])
                total_updated += 1

                if len(actions) >= self.config.batch_size:
                    success, _ = self.os_client.bulk_update(actions)
                    actions = []

        # Handle unclustered documents (singletons)
        logger.info("📝 Processing unclustered documents...")
        singleton_count = 0
        
        for doc in docs:
            if doc["id"] not in clustered_ids:
                solo_keywords = TextProcessor.extract_keywords([doc["text"]], top_n=3)
                actions.append({
                    "_op_type": "update",
                    "_index": self.config.source_index,
                    "_id": doc["id"],
                    "doc": {
                        "group_topic": doc["text"][:80],
                        "cluster_id": None,
                        "cluster_size": 1,
                        "cluster_keywords": solo_keywords
                    }
                })
                total_updated += 1
                singleton_count += 1

                if len(actions) >= self.config.batch_size:
                    success, _ = self.os_client.bulk_update(actions)
                    actions = []

        # Final batch
        if actions:
            success, _ = self.os_client.bulk_update(actions)

        logger.info(f"\n✅ Total updated: {total_updated} documents")
        logger.info(f"   • Clustered: {len(clustered_ids)}")
        logger.info(f"   • Singletons: {singleton_count}")
        logger.info("="*70)


# ====== PIPELINE ORCHESTRATOR ======
class ClusteringPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.os_client = None
        self.embedding_processor = None
        self.clustering_engine = None
        self.indexing_manager = None
    
    def setup(self):
        """Initialize all components"""
        logger.info("🎯 TREND CLUSTERING PIPELINE v5.0")
        logger.info("="*70)
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.os_client = OpenSearchClient(self.config)
        self.embedding_processor = EmbeddingProcessor(self.config)
        self.clustering_engine = ClusteringEngine(self.config)
        self.indexing_manager = IndexingManager(self.os_client, self.config)
        
        logger.info("✅ All components initialized")
    
    def run(self):
        """Execute the complete clustering pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Fetch documents
            logger.info("\n📥 STEP 1: Fetching documents...")
            docs = self.os_client.fetch_all_docs()
            
            if len(docs) < self.config.min_cluster_size:
                logger.error(f"❌ Not enough documents ({len(docs)} < {self.config.min_cluster_size})")
                return
            
            # Step 2: Create embeddings
            logger.info("\n🔢 STEP 2: Creating embeddings...")
            embeddings = self.embedding_processor.vectorize_docs(docs)
            
            # Step 3: Reduce dimensionality
            logger.info("\n📉 STEP 3: Reducing dimensionality...")
            embeddings_reduced, pca_model = self.embedding_processor.reduce_dimensionality(embeddings)
            
            # Step 4: Cluster documents
            logger.info("\n🎯 STEP 4: Clustering documents...")
            merged_struct, final_clusters = self.clustering_engine.cluster_all(docs, embeddings_reduced)
            
            if not final_clusters:
                logger.error("❌ No valid clusters found")
                return
            
            logger.info(f"✅ Found {len(final_clusters)} clusters")
            
            # Step 5: Update OpenSearch
            logger.info("\n📤 STEP 5: Updating OpenSearch...")
            self.os_client.ensure_cluster_fields_mapping()
            self.indexing_manager.update_clusters_in_opensearch(merged_struct, docs)
            
            # Summary
            elapsed = time.time() - start_time
            logger.info("\n" + "="*70)
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
            logger.info(f"📊 Documents processed: {len(docs)}")
            logger.info(f"🎯 Clusters created: {len(final_clusters)}")
            logger.info(f"⚡ Avg time per document: {elapsed/len(docs):.3f} seconds")
            logger.info("="*70)
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
        
        finally:
            # Cleanup
            if self.os_client:
                self.os_client.close()
    
    def get_cluster_statistics(self, final_clusters: Dict[int, List[Dict]]):
        """Generate cluster statistics"""
        sizes = [len(members) for members in final_clusters.values()]
        
        stats = {
            "total_clusters": len(final_clusters),
            "total_documents": sum(sizes),
            "min_cluster_size": min(sizes) if sizes else 0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "avg_cluster_size": np.mean(sizes) if sizes else 0,
            "median_cluster_size": np.median(sizes) if sizes else 0
        }
        
        logger.info("\n📈 CLUSTER STATISTICS:")
        for key, value in stats.items():
            logger.info(f"   • {key}: {value:.2f}" if isinstance(value, float) else f"   • {key}: {value}")
        
        return stats


# ====== MAIN ENTRY POINT ======
def main():
    """Main entry point"""
    try:
        # Load configuration
        config = Config.from_env()
        
        # Create and run pipeline
        pipeline = ClusteringPipeline(config)
        pipeline.setup()
        pipeline.run()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()