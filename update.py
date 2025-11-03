import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers
import urllib3

# ====== LOAD ENV ======
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ====== CONFIG ======
ELASTICSEARCH_NODE = os.getenv("OPENSEARCH_NODE")
ELASTICSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME")
ELASTICSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")
SOURCE_INDEX = os.getenv("OPENSEARCH_INDEX", "searched-tweets-index")

# ====== VALIDATE ENV ======
if not all([ELASTICSEARCH_NODE, ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD]):
    raise ValueError("❌ Missing OpenSearch credentials in .env file.")

# ====== INIT OPENSEARCH CLIENT ======
es = OpenSearch(
    [ELASTICSEARCH_NODE],
    http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    verify_certs=False,
    timeout=90
)

if not es.ping():
    raise ConnectionError("❌ Cannot connect to OpenSearch.")
print("✅ Connected to OpenSearch")

# ====== RESET SCRIPT ======
def reset_cluster_fields(index_name: str):
    print(f"🧹 Resetting cluster fields in index: {index_name}")

    # Fetch all documents using a scan helper (efficient scrolling)
    from opensearchpy.helpers import scan
    query = {"query": {"match_all": {}}, "_source": False}

    docs = scan(es, index=index_name, query=query, size=500, scroll="10m")
    actions = []
    total = 0

    for d in docs:
        actions.append({
            "_op_type": "update",
            "_index": index_name,
            "_id": d["_id"],
            "doc": {
                "group_topic": None,
                "cluster_id": None,
                "cluster_size": None,
                "cluster_keywords": []
            }
        })
        total += 1

        # Bulk update in chunks of 500 for efficiency
        if len(actions) >= 500:
            helpers.bulk(es, actions, request_timeout=90)
            print(f"   → Updated {total} docs so far...")
            actions = []

    # Final flush
    if actions:
        helpers.bulk(es, actions, request_timeout=90)

    print(f"✅ Finished resetting {total} documents in index '{index_name}'.")


if __name__ == "__main__":
    reset_cluster_fields(SOURCE_INDEX)
