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

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from qiskit_docs_mcp_server.cache import DiskCache
from qiskit_docs_mcp_server.constants import CACHE_DIR, CACHE_TTL
from qiskit_docs_mcp_server.sync import sync_all_docs

from . import server


def main() -> None:
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description="Qiskit Documentation MCP Server")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached documentation and exit.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Download all documentation to the cache for offline use.",
    )
    args = parser.parse_args()

    if args.clear_cache:
        if CACHE_DIR:
            disk = DiskCache(Path(CACHE_DIR))
            disk.clear()
            print(f"Cache cleared: {CACHE_DIR}")
        else:
            print("No cache directory configured (set QISKIT_DOCS_CACHE_DIR).")
        sys.exit(0)

    if args.sync:
        if not CACHE_DIR:
            print("Error: QISKIT_DOCS_CACHE_DIR must be set for --sync.")
            sys.exit(1)

        def progress(current: int, total: int, label: str) -> None:
            print(f"  [{current}/{total}] Syncing {label}...")

        result = asyncio.run(sync_all_docs(CACHE_DIR, ttl=CACHE_TTL, progress_callback=progress))
        print(
            f"\nSync complete: {result['success']}/{result['total']} pages downloaded"
            f" ({result['failed']} failed)."
        )
        sys.exit(0)

    server.mcp.run(transport="stdio", show_banner=False)


# Optionally expose other important items at package level
__all__ = ["main", "server"]
