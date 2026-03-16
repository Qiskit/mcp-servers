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

"""Sync command to pre-download all documentation for offline use."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from qiskit_docs_mcp_server.cache import DiskCache
from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_GUIDES,
    AVAILABLE_MODULES,
    HTTP_TIMEOUT,
    QISKIT_DOCS_BASE,
)


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


def _build_sync_urls() -> list[tuple[str, str, str]]:
    """Build list of (url, content_type, label) tuples for all known docs pages."""
    urls: list[tuple[str, str, str]] = [
        (f"{QISKIT_DOCS_BASE}api/qiskit/{module}", "text", f"module:{module}")
        for module in AVAILABLE_MODULES
    ]
    urls.extend(
        (f"{QISKIT_DOCS_BASE}api/qiskit-addon-{addon}", "text", f"addon:{addon}")
        for addon in AVAILABLE_ADDONS
    )
    urls.extend(
        (f"{QISKIT_DOCS_BASE}guides/{guide}", "text", f"guide:{guide}")
        for guide in AVAILABLE_GUIDES
    )
    urls.append((f"{QISKIT_DOCS_BASE}errors", "text", "errors"))
    return urls


async def sync_all_docs(
    cache_dir: str,
    ttl: float = 3600.0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Download all known documentation pages to the disk cache.

    Args:
        cache_dir: Path to the cache directory.
        ttl: TTL for cached entries in seconds.
        progress_callback: Optional callback(current, total, label) for progress.

    Returns:
        Summary dict with total, success, and failed counts.
    """
    disk_cache = DiskCache(Path(cache_dir), ttl=ttl)
    urls = _build_sync_urls()
    total = len(urls)
    success = 0
    failed = 0

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, follow_redirects=True) as client:
        for i, (url, content_type, label) in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, total, label)
            try:
                response = await client.get(url)
                response.raise_for_status()
                value: str | list[dict[str, Any]]
                if content_type == "json":
                    value = response.json()
                else:
                    value = response.text
                disk_cache.put(url, value, content_type)
                success += 1
            except Exception as e:
                logger.error(f"Failed to sync {label} ({url}): {e}")
                failed += 1

    return {"total": total, "success": success, "failed": failed}
