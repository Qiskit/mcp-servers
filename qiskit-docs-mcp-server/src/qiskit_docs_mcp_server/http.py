# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""HTTP client management, caching, and low-level fetch operations."""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

import httpx

from qiskit_docs_mcp_server.constants import (
    CACHE_TTL,
    HTTP_TIMEOUT,
    SEARCH_CACHE_TTL,
)


logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_RETRY_DELAY = 1.0


class _TTLCache:
    """Simple in-memory cache with TTL and LRU eviction."""

    def __init__(self, ttl: float = 3600.0, max_size: int = 128):
        self._ttl = ttl
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            timestamp, value = self._cache[key]
            if time.monotonic() - timestamp < self._ttl:
                self._cache.move_to_end(key)  # LRU touch
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Evict LRU — O(1)
        self._cache[key] = (time.monotonic(), value)

    def clear(self) -> None:
        self._cache.clear()


_text_cache = _TTLCache(ttl=CACHE_TTL)
_json_cache = _TTLCache(ttl=SEARCH_CACHE_TTL)

_client_holder: dict[str, httpx.AsyncClient] = {}


def set_http_client(client: httpx.AsyncClient) -> None:
    """Set the shared HTTP client (called by server lifespan)."""
    _client_holder["client"] = client


def clear_http_client() -> None:
    """Clear the shared HTTP client (called on server shutdown)."""
    _client_holder.clear()


def _get_http_client() -> httpx.AsyncClient:
    """Get or create a shared HTTP client."""
    client = _client_holder.get("client")
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True)
        _client_holder["client"] = client
    return client


async def _fetch_with_retry(url: str) -> httpx.Response | None:
    """Fetch a URL with retry for transient errors (5xx, timeouts).

    Args:
        url: The URL to fetch

    Returns:
        The httpx Response on success, or None if all attempts fail
    """
    client = _get_http_client()
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response
        except httpx.TimeoutException as e:  # noqa: PERF203
            last_error = e
            if attempt < _MAX_RETRIES - 1:
                logger.warning(
                    "Timeout fetching %s (attempt %d), retrying...",
                    url,
                    attempt + 1,
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < _MAX_RETRIES - 1 and e.response.status_code >= 500:
                logger.warning(
                    "Server error %d fetching %s (attempt %d), retrying...",
                    e.response.status_code,
                    url,
                    attempt + 1,
                )
                await asyncio.sleep(_RETRY_DELAY)
                continue
            break  # 4xx errors — don't retry
        except httpx.HTTPError as e:
            logger.error("Failed to fetch %s: %s", url, e)
            return None
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", url, e)
            return None

    logger.error("Failed to fetch %s after %d attempts: %s", url, _MAX_RETRIES, last_error)
    return None


async def fetch_text(url: str) -> str | None:
    """Fetch text content from a URL using httpx.

    Retries on transient errors (5xx status codes and timeouts).

    Args:
        url: The URL to fetch

    Returns:
        The text content of the page, or None if fetch fails
    """
    cached: str | None = _text_cache.get(url)
    if cached is not None:
        return cached

    response = await _fetch_with_retry(url)
    if response is None:
        return None

    result = response.text
    _text_cache.set(url, result)
    return result


async def fetch_text_json(url: str) -> list[dict[str, Any]] | None:
    """Fetch JSON content from a URL using httpx.

    Retries on transient errors (5xx status codes and timeouts).

    Args:
        url: The URL to fetch

    Returns:
        The JSON content as a list of dicts, or None if fetch fails
    """
    cached: list[dict[str, Any]] | None = _json_cache.get(url)
    if cached is not None:
        return cached

    response = await _fetch_with_retry(url)
    if response is None:
        return None

    result: list[dict[str, Any]] = response.json()
    _json_cache.set(url, result)
    return result
