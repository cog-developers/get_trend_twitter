import requests
from src.config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    DEEPSEEK_MODEL
)

def generate_topic(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.15,
        "max_tokens": 200
    }

    r = requests.post(
        DEEPSEEK_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"].strip()
