import os
from opensearchpy import OpenSearch

# ====== CONFIGURATION ======
PORT = 3005
OPENSEARCH_USERNAME = 'elastic'
OPENSEARCH_PASSWORD = 'k2CR1mWRXCc9=-sZc2zu'
OPENSEARCH_NODE = 'https://190.92.128.60:9200'
INDEX_NAME = "twiter_output_index"

# ====== CONNECT TO OPENSEARCH ======
client = OpenSearch(
    hosts=[OPENSEARCH_NODE],
    http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,      # set to True if you have a valid SSL certificate
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

# ====== DELETE INDEX ======
try:
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
        print(f"✅ Index '{INDEX_NAME}' deleted successfully.")
    else:
        print(f"❌ Index '{INDEX_NAME}' does not exist.")
except Exception as e:
    print(f"⚠️ Error deleting index: {e}")
