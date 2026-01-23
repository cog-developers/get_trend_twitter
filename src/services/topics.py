"""
Topic generation service.
Handles LLM-based topic generation using DeepSeek API.
"""

import re

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_API_URL,
    get_logger
)

logger = get_logger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def generate_cluster_topic(posts_text: str) -> str:
    """Generate topic for a cluster of posts using DeepSeek API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

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

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 200
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        topic = response.json()["choices"][0]["message"]["content"].strip()

        # Clean topic
        topic = re.sub(r'^الموضوعات?\s*(المستخرجة|هو|هي)?\s*:?\s*', '', topic, flags=re.IGNORECASE)
        topic = re.sub(r'^[\d\.\-\s:•]+', '', topic)
        topic = topic.strip('"').strip("'").strip()
        topic = re.sub(r'\.{2,}$', '', topic)

        return topic if len(topic) > 10 else "غير قابل للتحديد"
    except Exception as e:
        logger.error(f"Error generating topic: {e}")
        return "غير قابل للتحديد"
