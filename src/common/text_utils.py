"""Text processing utilities for Arabic content."""

import re
from typing import Optional
from datetime import datetime, timezone
from src.common.constants import ARABIC_STOPWORDS


def clean_text(text: str) -> str:
    """Clean and normalize Arabic text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\u2600-\u26FF]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)
    
    # Keep only Arabic letters, numbers, and spaces
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords and short words
    words = [w for w in text.split() if len(w) > 2 and w not in ARABIC_STOPWORDS]
    
    return " ".join(words)


def parse_datetime_to_epoch_millis(value) -> Optional[int]:
    """
    Best-effort parse for OpenSearch date-ish values.
    
    Supports:
    - epoch millis (int/float)
    - ISO strings with/without 'Z'
    - datetime-like strings that datetime.fromisoformat can parse
    Returns epoch millis (UTC) or None.
    """
    if value is None:
        return None

    # epoch millis
    if isinstance(value, (int, float)):
        # If it looks like seconds, convert to millis (heuristic)
        if value < 10_000_000_000:  # < ~2286-11-20 in seconds
            return int(value * 1000)
        return int(value)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # Handle "Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

    return None


def get_post_timestamp_millis(post: dict) -> Optional[int]:
    """Extract the best available post timestamp as epoch millis."""
    for field in ("timestamp", "created_at", "post_created_at"):
        ts = parse_datetime_to_epoch_millis(post.get(field))
        if ts is not None:
            return ts
    return None
