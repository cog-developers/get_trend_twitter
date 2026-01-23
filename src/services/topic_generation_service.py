"""Topic generation service using DeepSeek API."""

import re
from tenacity import retry, stop_after_attempt, wait_exponential
from src.infra.deepseek_client import generate_topic
from src.logging.logger import get_logger

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def generate_cluster_topic(posts_text: str) -> str:
    """Generate topic for a cluster of posts."""
    prompt = f"""أنت محلل محتوى اجتماعي محترف.
مهمتك: استخرج موضوعاً واحداً واضحاً وموجزاً يصف بدقة محتوى المنشورات التالية.

⚠️ القواعد:
- اكتب الموضوع مباشرة دون مقدمات
- جملة واحدة كاملة فقط (10-15 كلمة)
- لا تبدأ بـ: "الموضوع"، "المنشورات"، "يتناول"
- لا تستخدم نقاط التعليق (...)
- إذا كان المحتوى غير ذي معنى، أجب بـ: "غير قابل للتحديد"

المنشورات:
{posts_text[:2000]}

الموضوع:"""
    
    try:
        topic = generate_topic(prompt)
        
        # Clean topic
        topic = re.sub(r'^الموضوعات?\s*(المستخرجة|هو|هي)?\s*:?\s*', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^[\d\.\-\s:•]+', '', topic)
        topic = topic.strip('"').strip("'").strip()
        topic = re.sub(r'\.{2,}$', '', topic)
        
        return topic if len(topic) > 10 else "غير قابل للتحديد"
    except Exception as e:
        logger.error(f"❌ Error generating topic: {e}")
        return "غير قابل للتحديد"
