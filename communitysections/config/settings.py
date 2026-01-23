"""
Configuration and settings for Community Sections.
All environment variables, constants, and Lebanese political parties data.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====== LOGGING SETUP ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ====== OPENSEARCH CONFIG ======
OPENSEARCH_NODE = os.getenv("OPENSEARCH_NODE", "http://localhost:9200")
OPENSEARCH_USERNAME = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD", "admin")
SOURCE_INDEX = os.getenv("SOURCE_INDEX", "past-month-tweets-index")
TARGET_INDEX = os.getenv("TARGET_INDEX", "classified-users-index")

# ====== DEEPSEEK API CONFIG ======
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# ====== PROCESSING CONFIG ======
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 10))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
MAX_USERS = int(os.getenv("MAX_USERS", 0))  # 0 = all users

# ====== RETRY CONFIG ======
RETRY_ATTEMPTS = 4
RATE_LIMIT_DELAY = 1.5  # seconds between API calls
ES_RETRY_DELAY = 1.0

# ====== LEBANESE POLITICAL PARTIES PROMPT ======
LEBANESE_PARTIES_PROMPT = """
Major Lebanese Political Parties and Groups:
**Christian Parties:**
1. FPM (Free Patriotic Movement/التيار الوطني الحر) - Led by Gebran Bassil, founded by Michel Aoun
2. Lebanese Forces (القوات اللبنانية/LF) - Led by Samir Geagea
3. Kataeb (حزب الكتائب اللبنانية) - Led by Samy Gemayel
4. Marada Movement (تيار المردة) - Led by Sleiman Frangieh
**Shia Parties:**
5. Hezbollah (حزب الله) - Led by Hassan Nasrallah/Naim Qassem
6. Amal Movement (حركة أمل) - Led by Nabih Berri
**Sunni Parties:**
7. Future Movement (تيار المستقبل) - Led by Saad Hariri (currently suspended)
**Druze Parties:**
8. PSP (Progressive Socialist Party/الحزب التقدمي الاشتراكي) - Led by Walid Jumblatt/Taymour Jumblatt
**Independent/Opposition:**
9. MMFD (Citizens in a State/مواطنون ومواطنات في دولة) - Led by Charbel Nahas
10. LCP (Lebanese Communist Party/الحزب الشيوعي اللبناني)
11. October 17 Revolution Groups (مجموعات ثورة 17 تشرين) - Various independent activists
12. Sabaa Party (حزب سبعة) - Political reform movement
**Other Categories:**
13. Media/Journalist (إعلامي) - Professional journalists and media figures
14. Independent (مستقل/محايد) - No clear party affiliation
15. Unknown (غير معروف) - Insufficient information
"""


def validate_config():
    """Validate required environment variables."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-api-key-here":
        logger.error("DEEPSEEK_API_KEY not set in environment!")
        raise ValueError("Missing DEEPSEEK_API_KEY")

    logger.info(f"Using DeepSeek API Key: {DEEPSEEK_API_KEY[:8]}...{DEEPSEEK_API_KEY[-4:]}")
    logger.info(f"API Endpoint: {DEEPSEEK_API_URL}")
    logger.info(f"Model: {DEEPSEEK_MODEL}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
