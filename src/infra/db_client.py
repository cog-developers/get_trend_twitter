"""HTTP client for fetching data from the API service."""

import os
import json
import time
from typing import List, Dict, Optional
import requests
from src.logging.logger import get_logger

logger = get_logger(__name__)

# API configuration
API_BASE_URL = "https://dev-data.teledeck.news"

# Proxy configuration
PROXY_URL = "http://user-cogdev_aDG27:cog_Devs24leb@isp.oxylabs.io:8001"
PROXIES = {
    "http": PROXY_URL,
    "https": PROXY_URL,
}


def validate_db_config():
    """Validate API configuration by checking connectivity."""
    url = f"{API_BASE_URL}/api/tracking-inputs/active"
    try:
        response = requests.get(url, proxies=PROXIES, timeout=10)
        response.raise_for_status()
        logger.info("API connectivity validated: %s", API_BASE_URL)
    except requests.RequestException as exc:
        raise ValueError(f"Cannot connect to API at {API_BASE_URL}: {exc}")


def fetch_active_inputs() -> List[Dict]:
    """Fetch active user tracking inputs from the API."""
    retry_delay = float(os.getenv("API_RETRY_DELAY_SECONDS", "5"))
    url = f"{API_BASE_URL}/api/tracking-inputs/active"

    while True:
        try:
            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                proxies=PROXIES,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("API request failed; retrying in %ss: %s", retry_delay, exc)
            time.sleep(retry_delay)


def normalize_source_ids(raw_accounts) -> Optional[List[str]]:
    """Normalize accounts field into a list of source IDs."""
    if raw_accounts is None:
        return None
    if isinstance(raw_accounts, list):
        return raw_accounts or None
    if isinstance(raw_accounts, str):
        try:
            parsed = json.loads(raw_accounts)
            return parsed or None
        except json.JSONDecodeError:
            return None
    return None
