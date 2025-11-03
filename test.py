# import os
# import requests
# import json
# from urllib.parse import urlparse

# # ====== CONFIG ======
# XAI_API_KEY = os.getenv("XAI_API_KEY") or "YOUR_XAI_API_KEY_HERE"
# # proxy may be set here as a default, but we won't force using it unless APPLY_PROXY=1
# DEFAULT_PROXY = os.getenv("DEFAULT_PROXY") or "http://customer-cogdev_KLWHh-cc-ca-sessid-0342479676-sesstime-1440:cog_Devs24leb@pr.oxylabs.io:7777"

# # control flags
# APPLY_PROXY = os.getenv("APPLY_PROXY", "").lower() in ("1", "true", "yes")
# MOCK_GROK = os.getenv("MOCK_GROK", "").lower() in ("1", "true", "yes")

# # If API key is missing, fall back to mock mode unless FORCE_NETWORK is set
# FORCE_NETWORK = os.getenv("FORCE_NETWORK", "").lower() in ("1", "true", "yes")
# if (not XAI_API_KEY or XAI_API_KEY == "YOUR_XAI_API_KEY_HERE") and not FORCE_NETWORK:
#     print("⚠️ XAI_API_KEY is not set or is a placeholder. Falling back to MOCK_GROK mode.")
#     print("Set XAI_API_KEY and unset MOCK_GROK/FORCE_NETWORK to call the real API.")
#     MOCK_GROK = True

# if MOCK_GROK:
#     print("ℹ️ MOCK_GROK enabled — returning canned topic without calling the API.")
#     print("✅ Generated Topic:")
#     print("نقاش حول الانتخابات والمشاركة في التصويت.")
#     raise SystemExit(0)

# # ====== PROXY SETUP ======
# # Use proxies only when APPLY_PROXY is true. Otherwise let requests use system env or direct route.
# proxies = None
# if APPLY_PROXY:
#     proxies = {"http": DEFAULT_PROXY, "https": DEFAULT_PROXY}
#     print(f"ℹ️ APPLY_PROXY enabled — using proxy {DEFAULT_PROXY}")
# else:
#     print("ℹ️ Not applying custom proxy from script. To force, set APPLY_PROXY=1 in env.")

# # ====== API ENDPOINT ======
# url = "https://api.x.ai/v1/chat/completions"

# # ====== PAYLOAD ======
# payload = {
#     "messages": [
#         {"role": "system", "content": "You are Grok, a highly intelligent Arabic content analyst."},
#         {"role": "user", "content": (
#             "أنت محلل خبير للمحتوى العربي على وسائل التواصل الاجتماعي.\n"
#             "مهمتك: صِغ جملة وصفية كاملة ومختصرة تصف الموضوع الأساسي المشترك بين هذه المنشورات. "
#             "ركّز على الحدث/القضية/الشخص/المكان، ولا تضف تفسيرات طويلة — جملة واحدة فقط.\n"
#             "المنشورات:\n"
#             "الانتخابات القادمة في لبنان ستؤثر على الوضع السياسي.\n"
#             "تساؤلات حول مشاركة المغتربين في التصويت.\n"
#             "الجدل السياسي حول البرلمان مستمر.\n\n"
#             "أجب بجملة واحدة كاملة فقط (تنتهي بنقطة) بدون أي شرح إضافي:"
#         )}
#     ],
#     "model": "grok-3",
#     "stream": False
# }

# # ====== HEADERS ======
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {XAI_API_KEY}"
# }

# # ====== SEND REQUEST ======
# try:
#     response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=3600)
#     # If proxy rejected (403) and we set APPLY_PROXY, give a helpful hint and retry without proxy
#     if response.status_code == 403 and proxies is not None:
#         print("⚠️ Received HTTP 403 when using the configured proxy. This may mean the proxy blocks the request.")
#         print("Trying again without the script-applied proxy to see if the proxy is the cause...")
#         response = requests.post(url, headers=headers, json=payload, timeout=3600)

#     response.raise_for_status()  # raise exception if HTTP error
#     data = response.json()
#     # Extract the generated content safely
#     topic = None
#     try:
#         topic = data.get("choices", [])[0].get("message", {}).get("content", "").strip()
#     except Exception:
#         # fallback older structure
#         topic = data.get("choices", [])[0].get("text", "").strip() if data.get("choices") else ""

#     if not topic:
#         print("⚠️ API returned no topic. Full response:")
#         print(json.dumps(data, ensure_ascii=False, indent=2))
#     else:
#         print("✅ Generated Topic:")
#         print(topic)

# except requests.exceptions.RequestException as e:
#     # show helpful debugging info
#     print("⚠️ Grok REST API Error:", e)
#     if proxies is not None:
#         print("Hint: the configured proxy may be blocking this request. Try running without APPLY_PROXY or unset HTTP(S)_PROXY environment variables.")
import os
import requests
import json
from typing import List

def get_cluster_topic(posts: List[str]) -> str:
    """
    Generate a concise Arabic topic sentence describing the main theme of a list of posts
    using Grok API. Falls back to mock topic if API key is missing or network fails.

    Args:
        posts (List[str]): List of Arabic text strings (social media posts).

    Returns:
        str: Single-line topic describing the common theme of the posts.
    """

    # ====== CONFIG ======
    XAI_API_KEY = os.getenv("XAI_API_KEY") or "YOUR_XAI_API_KEY_HERE"
    DEFAULT_PROXY = os.getenv("DEFAULT_PROXY") or \
        "http://customer-cogdev_KLWHh-cc-ca-sessid-0342479676-sesstime-1440:cog_Devs24leb@pr.oxylabs.io:7777"
    APPLY_PROXY = os.getenv("APPLY_PROXY", "").lower() in ("1", "true", "yes")
    MOCK_GROK = os.getenv("MOCK_GROK", "").lower() in ("1", "true", "yes")
    FORCE_NETWORK = os.getenv("FORCE_NETWORK", "").lower() in ("1", "true", "yes")

    # Use mock if API key missing and FORCE_NETWORK is not set
    if (not XAI_API_KEY or XAI_API_KEY == "YOUR_XAI_API_KEY_HERE") and not FORCE_NETWORK:
        MOCK_GROK = True

    if MOCK_GROK:
        return "نقاش حول الانتخابات والمشاركة في التصويت."

    # ====== PROXY SETUP ======
    proxies = {"http": DEFAULT_PROXY, "https": DEFAULT_PROXY} if APPLY_PROXY else None

    # ====== API ENDPOINT ======
    url = "https://api.x.ai/v1/chat/completions"

    # ====== PAYLOAD ======
    joined_posts = "\n".join(posts)
    payload = {
        "messages": [
            {"role": "system", "content": "You are Grok, a highly intelligent Arabic content analyst."},
            {"role": "user", "content": (
                "أنت محلل خبير للمحتوى العربي على وسائل التواصل الاجتماعي.\n"
                "مهمتك: صِغ جملة وصفية كاملة ومختصرة تصف الموضوع الأساسي المشترك بين هذه المنشورات. "
                "ركّز على الحدث/القضية/الشخص/المكان، ولا تضف تفسيرات طويلة — جملة واحدة فقط.\n"
                f"المنشورات:\n{joined_posts}\n\n"
                "أجب بجملة واحدة كاملة فقط (تنتهي بنقطة) بدون أي شرح إضافي:"
            )}
        ],
        "model": "grok-3",
        "stream": False
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}"
    }

    # ====== SEND REQUEST ======
    try:
        response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=60)

        # Retry without proxy if 403
        if response.status_code == 403 and proxies is not None:
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        response.raise_for_status()
        data = response.json()

        # Extract topic safely
        try:
            topic = data.get("choices", [])[0].get("message", {}).get("content", "").strip()
        except Exception:
            topic = data.get("choices", [])[0].get("text", "").strip() if data.get("choices") else ""

        if not topic:
            return "⚠️ Grok API returned no topic."
        return topic

    except requests.exceptions.RequestException as e:
        print("⚠️ Grok REST API Error:", e)
        return "⚠️ Failed to generate topic due to network/API error."

# ====== Example usage ======
if __name__ == "__main__":
    posts = [
        "الانتخابات القادمة في لبنان ستؤثر على الوضع السياسي.",
        "تساؤلات حول مشاركة المغتربين في التصويت.",
        "الجدل السياسي حول البرلمان مستمر."
    ]
    topic = get_cluster_topic(posts)
    print("✅ Generated Topic:", topic)
