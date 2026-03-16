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

"""Caching layer for documentation fetches.

Provides in-memory LRU cache with TTL, persistent disk cache, and a layered
coordinator that combines both for transparent caching with stale fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

CacheValue = str | list[dict[str, Any]]


@dataclass
class _CacheEntry:
    """A single in-memory cache entry."""

    value: CacheValue
    expires_at: float
    content_type: str


@dataclass
class DiskCacheResult:
    """Result from a disk cache lookup."""

    value: CacheValue
    content_type: str
    is_stale: bool
    fetched_at: float


class TTLCache:
    """Thread-safe in-memory LRU cache with TTL expiration.

    Args:
        maxsize: Maximum number of entries.
        ttl: Time-to-live in seconds.
    """

    def __init__(self, maxsize: int = 128, ttl: float = 3600.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._last_hit = False

    @property
    def last_was_hit(self) -> bool:
        """Whether the last get() call was a cache hit."""
        return self._last_hit

    def get(self, key: str) -> _CacheEntry | None:
        """Get entry if it exists and is not expired. Moves to end (most recent).

        Args:
            key: Cache key (URL string).

        Returns:
            Cache entry or None if not found / expired.
        """
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                self._last_hit = False
                return None
            if time.monotonic() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                self._last_hit = False
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            self._last_hit = True
            return entry

    def put(self, key: str, value: CacheValue, content_type: str) -> None:
        """Store an entry. Evicts LRU if at capacity.

        Args:
            key: Cache key (URL string).
            value: The fetched content.
            content_type: Either "text" or "json".
        """
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self._ttl,
                content_type=content_type,
            )
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with hits, misses, size, maxsize, and ttl.
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "ttl": self._ttl,
            }


class DiskCache:
    """File-based persistent cache with JSON-serialized entries.

    Args:
        cache_dir: Directory for cache files.
        ttl: Time-to-live in seconds for freshness (stale entries still readable).
    """

    def __init__(self, cache_dir: Path, ttl: float = 3600.0) -> None:
        self._cache_dir = cache_dir
        self._ttl = ttl
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _url_to_filename(url: str) -> str:
        """Convert URL to a safe, unique filename."""
        return hashlib.sha256(url.encode()).hexdigest() + ".json"

    def _entry_path(self, url: str) -> Path:
        return self._cache_dir / self._url_to_filename(url)

    def get(self, url: str) -> DiskCacheResult | None:
        """Read from disk. Returns None if not found.

        Args:
            url: The URL cache key.

        Returns:
            DiskCacheResult with is_stale flag, or None.
        """
        path = self._entry_path(url)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = float(data["fetched_at"])
            is_stale = (time.time() - fetched_at) > self._ttl
            return DiskCacheResult(
                value=data["value"],
                content_type=data["content_type"],
                is_stale=is_stale,
                fetched_at=fetched_at,
            )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Corrupted cache file {path}, removing: {e}")
            path.unlink(missing_ok=True)
            return None

    def get_stale(self, url: str) -> DiskCacheResult | None:
        """Return entry even if expired (for fallback on network failure).

        Args:
            url: The URL cache key.

        Returns:
            DiskCacheResult (always with is_stale=True if expired) or None.
        """
        return self.get(url)

    def put(self, url: str, value: CacheValue, content_type: str) -> None:
        """Write entry to disk atomically.

        Args:
            url: The URL cache key.
            value: The fetched content.
            content_type: Either "text" or "json".
        """
        data = {
            "url": url,
            "content_type": content_type,
            "value": value,
            "fetched_at": time.time(),
            "ttl": self._ttl,
        }
        path = self._entry_path(url)
        try:
            fd, tmp_path_str = tempfile.mkstemp(dir=self._cache_dir, suffix=".tmp")
            tmp = Path(tmp_path_str)
            try:
                os.close(fd)
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                tmp.replace(path)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except OSError as e:
            logger.warning(f"Failed to write cache file {path}: {e}")

    def clear(self) -> None:
        """Delete all cache files."""
        for path in self._cache_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        logger.info(f"Disk cache cleared: {self._cache_dir}")

    def info(self) -> dict[str, Any]:
        """Return disk cache stats.

        Returns:
            Dict with file_count, total_size_bytes, and directory.
        """
        files = list(self._cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files if f.exists())
        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "directory": str(self._cache_dir),
        }


class LayeredCache:
    """Coordinates in-memory and on-disk caches.

    Args:
        memory_cache: The in-memory TTL cache.
        disk_cache: Optional disk cache (None for memory-only mode).
    """

    def __init__(
        self,
        memory_cache: TTLCache,
        disk_cache: DiskCache | None = None,
    ) -> None:
        self._memory = memory_cache
        self._disk = disk_cache

    def get(self, url: str) -> tuple[CacheValue | None, bool]:
        """Look up a URL in memory, then disk.

        Args:
            url: The URL cache key.

        Returns:
            Tuple of (value_or_none, is_stale). If not found, (None, False).
        """
        # Check memory first
        mem_entry = self._memory.get(url)
        if mem_entry is not None:
            return mem_entry.value, False

        # Check disk
        if self._disk is not None:
            disk_result = self._disk.get(url)
            if disk_result is not None:
                if not disk_result.is_stale:
                    # Promote fresh disk entry to memory
                    self._memory.put(url, disk_result.value, disk_result.content_type)
                return disk_result.value, disk_result.is_stale

        return None, False

    def put(self, url: str, value: CacheValue, content_type: str) -> None:
        """Write to both memory and disk caches.

        Args:
            url: The URL cache key.
            value: The fetched content.
            content_type: Either "text" or "json".
        """
        self._memory.put(url, value, content_type)
        if self._disk is not None:
            self._disk.put(url, value, content_type)

    def get_stale_fallback(self, url: str) -> CacheValue | None:
        """Return stale disk entry for network failure fallback.

        Args:
            url: The URL cache key.

        Returns:
            Cached value (possibly stale) or None.
        """
        if self._disk is not None:
            result = self._disk.get_stale(url)
            if result is not None:
                return result.value
        return None

    def clear(self) -> None:
        """Clear both memory and disk caches."""
        self._memory.clear()
        if self._disk is not None:
            self._disk.clear()

    def info(self) -> dict[str, Any]:
        """Return combined cache statistics.

        Returns:
            Dict with memory and optional disk cache stats.
        """
        result: dict[str, Any] = {"memory": self._memory.info()}
        if self._disk is not None:
            result["disk"] = self._disk.info()
        return result
