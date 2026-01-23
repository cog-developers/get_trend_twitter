"""MySQL database client for worker service."""

import os
import json
import time
from typing import List, Dict, Optional
import pymysql
from src.logging.logger import get_logger

logger = get_logger(__name__)


def validate_db_config():
    """Validate database configuration from environment variables."""
    required_vars = ["DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing DB env vars: {', '.join(missing)}")


def get_db_connection():
    """Create a MySQL database connection, retrying on failure."""
    retry_delay = float(os.getenv("DB_RETRY_DELAY_SECONDS", "5"))
    while True:
        try:
            return pymysql.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                port=int(os.getenv("DB_PORT", "3306")),
                cursorclass=pymysql.cursors.DictCursor,
            )
        except pymysql.MySQLError as exc:
            logger.warning("DB connection failed; retrying in %ss: %s", retry_delay, exc)
            time.sleep(retry_delay)


def fetch_active_inputs() -> List[Dict]:
    """Fetch active user tracking inputs ordered by creation time."""
    query = """
        SELECT id, user_id, country, keywords, accounts, active, created_at, updated_at
        FROM user_tracking_inputs
        WHERE active = 1
        ORDER BY created_at ASC
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        connection.close()


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
