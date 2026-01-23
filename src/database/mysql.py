"""
MySQL database operations.
Connection management and user tracking input queries.
"""

import json
import time
from typing import List, Dict, Optional

import pymysql

from ..config.settings import (
    DB_HOST,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
    DB_PORT,
    DB_RETRY_DELAY_SECONDS,
    get_logger
)

logger = get_logger(__name__)


def get_db_connection():
    """Create a MySQL database connection, retrying on failure."""
    while True:
        try:
            return pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except pymysql.MySQLError as exc:
            logger.warning("DB connection failed; retrying in %ss: %s", DB_RETRY_DELAY_SECONDS, exc)
            time.sleep(DB_RETRY_DELAY_SECONDS)


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
