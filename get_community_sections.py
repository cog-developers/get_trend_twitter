import os
import re
import requests
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict, Counter
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import urllib3
from datetime import datetime
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch

# ====== SETUP LOGGING ======
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====== LOAD ENVIRONMENT VARIABLES ======
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== CONFIG ======
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "searched-tweets-index")
STANCE_INDEX = os.getenv("STANCE_INDEX", "stance-analysis-index")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", 0.85))
KEYWORD_MATCH_MIN = int(os.getenv("KEYWORD_MATCH_MIN", 2))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))

# ====== VALIDATE ENVIRONMENT ======
if not all([OPENSEARCH_NODE, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD]):
    raise ValueError("❌ Missing OpenSearch credentials in .env file.")

# ====== INITIALIZE OPENSEARCH CLIENT ======
es = OpenSearch(
    [OPENSEARCH_NODE],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    verify_certs=False,
    timeout=180,
    max_retries=5,
    retry_on_timeout=True
)

if not es.ping():
    raise ConnectionError("❌ Cannot connect to OpenSearch.")
logger.info("✅ Connected to OpenSearch.")

# ====== CREATE STANCE ANALYSIS INDEX ======
def create_stance_index():
    """Create index for storing stance analysis results."""
    index_body = {
        "settings": {
            "number_of_shards": 2,
            "number_of_replicas": 1,
            "index": {
                "refresh_interval": "5s"
            }
        },
        "mappings": {
            "properties": {
                "original_post_id": {"type": "keyword"},
                "post_text": {"type": "text", "analyzer": "standard"},
                "party": {"type": "keyword"},
                "stance_label": {"type": "keyword"},
                "confidence": {"type": "keyword"},
                "classification_method": {"type": "keyword"},
                "similarity_score": {"type": "float"},
                "matched_keywords": {"type": "keyword"},
                "timestamp": {"type": "date"},
                "analysis_run_id": {"type": "keyword"}
            }
        }
    }
    
    if es.indices.exists(index=STANCE_INDEX):
        logger.info(f"ℹ️  Index '{STANCE_INDEX}' already exists.")
    else:
        es.indices.create(index=STANCE_INDEX, body=index_body)
        logger.info(f"✅ Created index '{STANCE_INDEX}'")

# ====== LOAD SENTENCE TRANSFORMER MODEL ======
logger.info("⚙️ Loading sentence transformer model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')
    logger.info("✅ Model moved to GPU")

# ====== PARTY-SPECIFIC CONFIGURATION ======
PARTY_CONFIG = {
    "حزب الله": {
        "aliases": ["حزب الله", "الحزب", "هزب الله", "حزبالله"],
        "context_required": True,
        "support_keywords": ["مقاومة", "صمود", "انتصار", "أبطال", "دعم المقاومة", "نصر"],
        "oppose_keywords": ["ميليشيا", "سلاح غير شرعي", "إرهاب", "احتلال", "سيطرة", "حصر السلاح"]
    },
    "تيار المستقبل": {
        "aliases": ["تيار المستقبل", "المستقبل السياسي"],
        "context_required": True,
        "support_keywords": ["سعد الحريري", "تيار", "ندعم المستقبل", "مع التيار"],
        "oppose_keywords": ["ضد التيار", "فشل التيار"]
    },
    "القوات اللبنانية": {
        "aliases": ["القوات اللبنانية", "قوات لبنانية", "سمير جعجع"],
        "context_required": True,
        "support_keywords": ["القوات اللبنانية", "دعم القوات", "مع القوات", "لائحة القوات"],
        "oppose_keywords": ["ضد القوات اللبنانية", "ميليشيات القوات"]
    },
    "حركة حماس": {
        "aliases": ["حركة حماس", "حماس"],
        "context_required": False,
        "support_keywords": ["مقاومة", "طوفان الأقصى", "دعم حماس", "أبطال حماس"],
        "oppose_keywords": ["إرهاب حماس", "ضد حماس"]
    }
}

# ====== KEYWORD DETECTION ======
def build_keyword_patterns(party_config: Dict) -> Dict:
    """Build optimized regex patterns for each party."""
    patterns = {}
    for party, config in party_config.items():
        support_kw = config.get("support_keywords", [])
        oppose_kw = config.get("oppose_keywords", [])
        
        if support_kw:
            patterns[f"{party}_support"] = re.compile(
                '|'.join(map(re.escape, support_kw)), 
                re.IGNORECASE
            )
        if oppose_kw:
            patterns[f"{party}_oppose"] = re.compile(
                '|'.join(map(re.escape, oppose_kw)), 
                re.IGNORECASE
            )
    return patterns

KEYWORD_PATTERNS = build_keyword_patterns(PARTY_CONFIG)

@lru_cache(maxsize=10000)
def party_mentioned_in_context(text: str, party: str) -> bool:
    """Check if party is actually mentioned in the text."""
    config = PARTY_CONFIG.get(party, {})
    aliases = config.get("aliases", [party])
    for alias in aliases:
        if alias in text:
            return True
    return False

# ====== ENHANCED KEYWORD LABELING WITHOUT NEUTRAL ======
@lru_cache(maxsize=10000)
def keyword_label_cached(text: str, party: str) -> Tuple[str, int, List[str]]:
    """Enhanced keyword matching to ensure classification as 'with' or 'against'."""
    if not party_mentioned_in_context(text, party):
        logger.debug(f"[Keyword Label] Party '{party}' not mentioned in text.")
        return "against", 0, []  # Default to 'against' if party is not mentioned

    support_pattern_key = f"{party}_support"
    oppose_pattern_key = f"{party}_oppose"

    matched_with = []
    matched_against = []

    if support_pattern_key in KEYWORD_PATTERNS:
        support_matches = KEYWORD_PATTERNS[support_pattern_key].findall(text)
        matched_with = list(set(support_matches))

    if oppose_pattern_key in KEYWORD_PATTERNS:
        oppose_matches = KEYWORD_PATTERNS[oppose_pattern_key].findall(text)
        matched_against = list(set(oppose_matches))

    with_score = sum(len(match) for match in matched_with)  # Weight by keyword length
    against_score = sum(len(match) for match in matched_against)

    logger.debug(f"[Keyword Label] Party: {party}, With Score: {with_score}, Against Score: {against_score}, Matched With: {matched_with}, Matched Against: {matched_against}")

    # Always classify as 'with' or 'against'
    if with_score >= against_score:
        return "with", with_score, matched_with
    else:
        return "against", against_score, matched_against

# ====== PROTOTYPES ======
def build_prototypes_for_party(party_name: str) -> Dict[str, List[str]]:
    """Build richer prototypes with more variety."""
    config = PARTY_CONFIG.get(party_name, {})
    
    prototypes = {
        "with": [
            f"أنا مع {party_name} بقوة.",
            f"أنا مع {party_name} بكل فخر.",
            f"أدعم {party_name} وموقفهم.",
            f"{party_name} يمثلنا بشكل جيد.",
            f"نحن مع {party_name}.",
            f"{party_name} على حق.",
            f"أؤيّد {party_name}.",
            f"مع {party_name} دائماً."
        ],
        "against": [
            f"أنا ضد {party_name} تماماً.",
            f"لست مع {party_name} أبداً.",
            f"أرفض {party_name} وسياساتهم.",
            f"{party_name} مرفوض عندي.",
            f"ضد سياسات {party_name}.",
            f"{party_name} يضر بالبلد.",
            f"لا أؤيّد {party_name}.",
            f"ضد {party_name} كلياً."
        ],
        "neutral": [
            f"تقرير عن {party_name}.",
            f"خبر يتعلق ب {party_name}.",
            f"معلومات عن {party_name}.",
            "هذا محتوى معلوماتي.",
            "تحليل سياسي محايد.",
            "تغطية إخبارية."
        ]
    }
    
    if party_name in config:
        for kw in config.get("support_keywords", [])[:3]:
            prototypes["with"].append(f"{party_name} و{kw}.")
        for kw in config.get("oppose_keywords", [])[:3]:
            prototypes["against"].append(f"{party_name} {kw}.")
    
    return prototypes

def embed_prototypes(parties: List[str], batch_size: int = 16) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute embeddings for all party prototypes."""
    proto_emb = {}
    model.eval()
    
    with torch.no_grad():
        for p in parties:
            prototypes = build_prototypes_for_party(p)
            proto_emb[p] = {}
            
            for label, texts in prototypes.items():
                if not texts:
                    proto_emb[p][label] = np.empty((0, model.get_sentence_embedding_dimension()))
                    continue
                
                embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                    embeddings.append(batch_emb)
                
                proto_emb[p][label] = np.vstack(embeddings)
    
    return proto_emb

# ====== BATCH EMBEDDING ======
def batch_encode_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Encode texts in batches."""
    if not texts:
        return np.array([])
    return model.encode(texts, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)

# ====== SEMANTIC DETECTION ======
def semantic_label_batch(embeddings: np.ndarray, proto_emb: Dict[str, np.ndarray], 
                         threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> List[Tuple[str, float]]:
    """Vectorized semantic labeling to ensure classification as 'with' or 'against'."""
    results = []

    for emb in embeddings:
        emb = emb.reshape(1, -1)
        scores = {}

        for label in ["with", "against"]:  # Removed 'neutral'
            sims = cosine_similarity(emb, proto_emb[label])
            scores[label] = float(np.max(sims))

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        logger.debug(f"[Semantic Label] Scores: {scores}, Best Label: {best_label}, Best Score: {best_score}")

        # Always classify as 'with' or 'against'
        results.append((best_label, best_score))

    return results

# ====== FETCH POSTS ======
def fetch_posts_for_parties_texts(parties: List[str], max_per_party: int = 500) -> Dict[str, Dict]:
    """Fetch posts with improved party-specific queries."""
    from opensearchpy.helpers import scan
    topics = defaultdict(lambda: {'doc_count': 0, 'posts': []})

    logger.info("\n🎯 FETCHING POSTS FOR LEBANESE POLITICAL PARTIES")

    for party in parties:
        config = PARTY_CONFIG.get(party, {})
        aliases = config.get("aliases", [party])
        
        should_clauses = [{"match_phrase": {"post_text": alias}} for alias in aliases]
        
        query = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            },
            "_source": ["post_text"]
        }
        
        seen_texts = set()
        count = 0
        
        for doc in scan(es, query=query, index=SOURCE_INDEX, size=500, scroll="5m"):
            text = doc.get("_source", {}).get("post_text", "") or ""
            
            if not text or len(text) < 20 or text in seen_texts:
                continue
            
            if not party_mentioned_in_context(text, party):
                continue
                
            seen_texts.add(text)
            topics[party]['doc_count'] += 1
            
            if len(topics[party]['posts']) < max_per_party:
                topics[party]['posts'].append({'id': doc["_id"], 'text': text})
            
            count += 1
        
        logger.info(f"✅ Retrieved {count} relevant posts for party '{party}'")
    
    return dict(topics)

# ====== DEEPSEEK INTEGRATION ======
def deepseek_classify(text: str, api_key: str, model: str) -> Tuple[Optional[str], Optional[float]]:
    """Classify text using DeepSeek API."""
    url = "https://api.deepseek.ai/v1/classify"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "inputs": [text]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result and "outputs" in result and len(result["outputs"]) > 0:
            label = result["outputs"][0].get("label")
            confidence = result["outputs"][0].get("confidence")
            return label, confidence
    except requests.RequestException as e:
        logger.error(f"❌ DeepSeek API error: {e}")

    return None, None

# ====== STANCE ANALYSIS ======
def analyze_stance_for_party(posts: List[Dict], party: str, 
                            proto_emb_party: Dict[str, np.ndarray], 
                            use_deepseek_if_uncertain: bool = False) -> Dict:
    """Multi-stage analysis with improved accuracy."""
    results = []
    counts = Counter()
    
    keyword_posts = []
    semantic_posts = []
    semantic_indices = []
    
    for idx, p in enumerate(posts):
        text = p['text']
        kw_label, kw_count, matched_kw = keyword_label_cached(text, party)
        
        if kw_label and kw_count >= KEYWORD_MATCH_MIN:
            counts[kw_label] += 1
            results.append({
                'id': p['id'], 
                'text': text, 
                'label': kw_label, 
                'reason': f"keyword({kw_count})",
                'confidence': 'high',
                'score': 1.0,
                'matched_keywords': matched_kw
            })
        else:
            semantic_posts.append(text)
            semantic_indices.append(idx)
    
    if semantic_posts:
        logger.info(f"  🧠 Running semantic analysis on {len(semantic_posts)} posts...")
        embeddings = batch_encode_texts(semantic_posts, batch_size=BATCH_SIZE)
        semantic_labels = semantic_label_batch(embeddings, proto_emb_party)
        
        for original_idx, (sem_label, sem_score) in zip(semantic_indices, semantic_labels):
            p = posts[original_idx]
            text = p['text']
            
            if sem_label in ("with", "against") and sem_score >= 0.75:
                confidence = 'high'
            elif sem_label in ("with", "against") and sem_score >= SEMANTIC_SIMILARITY_THRESHOLD:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            label = sem_label
            reason = f"semantic({sem_score:.3f})"
            
            # Use DeepSeek if enabled and confidence is low
            if use_deepseek_if_uncertain and confidence == 'low':
                logger.info(f"🔍 Using DeepSeek for uncertain post: {text[:50]}...")
                ds_label, ds_confidence = deepseek_classify(text, DEEPSEEK_API_KEY, DEEPSEEK_MODEL)

                if ds_label in ("with", "against") and ds_confidence:
                    confidence = "high" if ds_confidence >= 0.85 else "medium"
                    label = ds_label
                    reason = f"deepseek({ds_confidence:.3f})"

            counts[label] += 1
            results.append({
                'id': p['id'], 
                'text': text, 
                'label': label, 
                'reason': reason,
                'confidence': confidence,
                'score': sem_score,
                'matched_keywords': []
            })
    
    return {'counts': counts, 'results': results}

# ====== PARALLEL PROCESSING ======
def analyze_all_parties_parallel(topics: Dict[str, Dict], proto_emb_all: Dict, 
                                use_deepseek: bool = False) -> Dict:
    """Analyze multiple parties in parallel."""
    all_analysis = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_party = {}
        
        for party, data in topics.items():
            posts = data['posts']
            if not posts:
                continue
            future = executor.submit(
                analyze_stance_for_party, 
                posts, 
                party, 
                proto_emb_all[party], 
                use_deepseek
            )
            future_to_party[future] = party
        
        for future in tqdm(as_completed(future_to_party), total=len(future_to_party), 
                          desc="Analyzing parties"):
            party = future_to_party[future]
            try:
                analysis = future.result()
                all_analysis[party] = analysis
                logger.info(f"✅ Completed analysis for '{party}'")
            except Exception as e:
                logger.error(f"❌ Error analyzing '{party}': {e}")
                all_analysis[party] = {'counts': Counter(), 'results': []}
    
    return all_analysis

# ====== SAVE TO OPENSEARCH ======
def save_to_opensearch(all_party_analysis: Dict[str, Dict], run_id: str):
    """Save stance analysis results to OpenSearch index."""
    logger.info(f"\n💾 Saving results to OpenSearch index '{STANCE_INDEX}'...")
    
    actions = []
    timestamp = datetime.utcnow().isoformat()
    
    for party, analysis in all_party_analysis.items():
        for result in analysis['results']:
            doc = {
                "_index": STANCE_INDEX,
                "_source": {
                    "original_post_id": result['id'],
                    "post_text": result['text'],
                    "party": party,
                    "stance_label": result['label'],
                    "confidence": result['confidence'],
                    "classification_method": result['reason'].split('(')[0],
                    "similarity_score": result.get('score', 0.0),
                    "matched_keywords": result.get('matched_keywords', []),
                    "timestamp": timestamp,
                    "analysis_run_id": run_id
                }
            }
            actions.append(doc)
    
    if actions:
        try:
            success, failed = helpers.bulk(es, actions, chunk_size=500, request_timeout=60)
            logger.info(f"✅ Successfully indexed {success} documents")
            if failed:
                logger.warning(f"⚠️  Failed to index {len(failed)} documents")
        except Exception as e:
            logger.error(f"❌ Error during bulk indexing: {e}")
    else:
        logger.warning("⚠️  No documents to index")

# ====== SUMMARY ======
def print_stance_summary(all_party_analysis: Dict[str, Dict], show_samples: bool = True):
    """Print analysis summary."""
    print("\n" + "="*80)
    print("📢 STANCE ANALYSIS SUMMARY")
    print("="*80)
    
    for party, analysis in all_party_analysis.items():
        counts = analysis['counts']
        total = sum(counts.values())
        
        if total == 0:
            print(f"\n🎯 Party: {party}")
            print(f"   ⚠️  No relevant posts found")
            continue
            
        support_pct = (counts.get('with', 0) / total) * 100
        oppose_pct = (counts.get('against', 0) / total) * 100
        neutral_pct = (counts.get('neutral', 0) / total) * 100
        
        print(f"\n{'='*80}")
        print(f"🎯 PARTY: {party}")
        print(f"{'='*80}")
        print(f"📊 Total Posts: {total}")
        print(f"   ✅ With:    {counts.get('with',0):4d} ({support_pct:5.1f}%)")
        print(f"   ❌ Against: {counts.get('against',0):4d} ({oppose_pct:5.1f}%)")
        print(f"   ⚪ Neutral: {counts.get('neutral',0):4d} ({neutral_pct:5.1f}%)")
        
        if show_samples:
            print(f"\n📝 Sample Posts:")
            results = analysis['results']
            for label_type in ['with', 'against', 'neutral']:
                label_posts = [r for r in results if r['label'] == label_type][:3]
                if label_posts:
                    emoji_map = {'with': '✅', 'against': '❌', 'neutral': '⚪'}
                    print(f"\n{emoji_map[label_type]} {label_type.upper()} samples:")
                    for idx, post in enumerate(label_posts, 1):
                        print(f"   {idx}. [{post['confidence']}] {post['text'][:100]}...")
        
        print(f"{'='*80}\n")

# ====== MAIN ======
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Create stance analysis index
    create_stance_index()
    
    # Generate unique run ID
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"🆔 Analysis Run ID: {run_id}")
    
    # Use party names from config
    parties = list(PARTY_CONFIG.keys())
    logger.info(f"📋 Analyzing parties: {', '.join(parties)}")
    
    # Fetch posts
    logger.info("📥 Fetching posts...")
    topics = fetch_posts_for_parties_texts(parties, max_per_party=500)
    
    # Precompute prototype embeddings
    logger.info("🔧 Computing prototype embeddings...")
    proto_emb_all = embed_prototypes(parties)
    
    # Analyze with parallel processing
    logger.info("🚀 Starting parallel stance analysis...")
    all_analysis = analyze_all_parties_parallel(topics, proto_emb_all, use_deepseek=False)
    
    # Save to OpenSearch
    save_to_opensearch(all_analysis, run_id)
    
    # Print summary
    print_stance_summary(all_analysis, show_samples=True)
    
    elapsed = time.time() - start_time
    logger.info(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")
    logger.info(f"✅ Results saved to OpenSearch index: '{STANCE_INDEX}'")
    logger.info(f"🔍 Query example: GET {STANCE_INDEX}/_search?q=analysis_run_id:{run_id}")