"""
Simple nba_api fetcher:
- JSON disk cache (reuse if present)
- retries + backoff via tenacity
- basic rate limiting (min delay between requests)
- cache includes small, useful metadata
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Type

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# ----------------------------
# Config
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw")
MIN_INTERVAL_S = 1.2
TIMEOUT_S = 15
MAX_RETRIES = 5

# Mimic a browser to avoid 403 Forbidden errors
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Sec-Ch-Ua": '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
}

_last_request_t = 0.0


# ----------------------------
# Core helpers
# ----------------------------

def _cache_path(endpoint_name: str, params: Dict[str, Any]) -> Path:
    """Stable filename based on endpoint + params."""
    key = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{endpoint_name}|{key}".encode("utf-8")).hexdigest()[:20]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{endpoint_name}_{digest}.json"


def _rate_limit() -> None:
    """Sleep if we are calling too fast."""
    global _last_request_t
    now = time.monotonic()
    wait = MIN_INTERVAL_S - (now - _last_request_t)
    if wait > 0:
        time.sleep(wait)
    _last_request_t = time.monotonic()


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry=retry_if_exception_type(
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _call_endpoint(endpoint_cls: Type[Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Actual API call with retries and headers."""
    _rate_limit()
    logger.info(f"Calling endpoint: {endpoint_cls.__name__} with params: {params}")
    ep = endpoint_cls(**params, timeout=TIMEOUT_S, headers=HEADERS)
    return ep.get_dict()


def fetch(endpoint_cls: Type[Any], params: Dict[str, Any], *, use_cache: bool = True) -> Dict[str, Any] | None:
    """
    Fetch nba_api endpoint data with JSON cache.
    Returns the cached file contents or None if fetching fails.
    """
    endpoint_name = endpoint_cls.__name__
    path = _cache_path(endpoint_name, params)

    if use_cache and path.exists():
        logger.info(f"CACHE HIT: {endpoint_name} for params: {params}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    try:
        started = time.time()
        data = _call_endpoint(endpoint_cls, params)
        ended = time.time()

        payload = {
            "meta": {
                "endpoint": endpoint_name,
                "params": params,
                "fetched_at_unix": ended,
                "duration_ms": int((ended - started) * 1000),
                "retries": MAX_RETRIES,
                "timeout_s": TIMEOUT_S,
                "min_interval_s": MIN_INTERVAL_S,
            },
            "data": data,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"SUCCESS: Fetched and cached {endpoint_name} for params: {params}")
        return payload

    except Exception as e:
        logger.error(f"FETCH FAILED: for {endpoint_name} with params {params}. Error: {e}")
        return None
