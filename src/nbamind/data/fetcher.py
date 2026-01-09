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
import time
from pathlib import Path
from typing import Any, Dict, Type

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

# ----------------------------
# Config
# ----------------------------

CACHE_DIR = Path("data/raw")   # change if you want
MIN_INTERVAL_S = 1.2          # simple pacing
TIMEOUT_S = 15                # request timeout (nba_api passes this through)
MAX_RETRIES = 5

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
)
def _call_endpoint(endpoint_cls: Type[Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Actual API call with retries."""
    _rate_limit()
    ep = endpoint_cls(**params, timeout=TIMEOUT_S)
    return ep.get_dict()


def fetch(endpoint_cls: Type[Any], params: Dict[str, Any], *, use_cache: bool = True) -> Dict[str, Any]:
    """
    Fetch nba_api endpoint data with JSON cache.
    Returns the cached file contents: {"meta": {...}, "data": {...}}.
    """
    endpoint_name = endpoint_cls.__name__
    path = _cache_path(endpoint_name, params)

    if use_cache and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    started = time.time()
    data = _call_endpoint(endpoint_cls, params)
    ended = time.time()

    payload = {
        "meta": {
            "endpoint": endpoint_name,
            "params": params,
            "fetched_at_unix": ended,
            "duration_ms": int((ended - started) * 1000),
            "timeout_s": TIMEOUT_S,
            "min_interval_s": MIN_INTERVAL_S,
            # "cache_version": 1,
        },
        "data": data,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload
    # return data

# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    # from nba_api.stats.endpoints import commonplayerinfo

    # result = fetch(commonplayerinfo.CommonPlayerInfo, {"player_id": 2544})
    # print("endpoint:", result["meta"]["endpoint"])
    # print("resultSets:", len(result["data"].get("resultSets", [])))