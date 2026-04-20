#!/usr/bin/env python3
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

"""Fetch the live sitemap and print updated fallback constant lists.

Run this script periodically to keep the hardcoded fallback values in
``constants.py`` in sync with the live Qiskit documentation sitemap.

Usage:
    cd qiskit-docs-mcp-server
    uv run python scripts/update_fallback_constants.py
"""

from __future__ import annotations

import sys

import httpx


sys.path.insert(0, "src")
from qiskit_docs_mcp_server.constants import SITEMAP_URL
from qiskit_docs_mcp_server.sitemap import _parse_sitemap_xml


def _format_list(name: str, values: list[str]) -> str:
    """Format a Python list constant."""
    items = ",\n".join(f'    "{v}"' for v in values)
    return f"{name}: list[str] = [\n{items},\n]"


def main() -> None:
    print(f"Fetching sitemap from {SITEMAP_URL} ...")
    response = httpx.get(SITEMAP_URL, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    xml_text = response.text

    pages = _parse_sitemap_xml(xml_text)

    print("\nDiscovered:")
    for category, items in pages.items():
        print(f"  {category}: {len(items)} entries")

    print("\n" + "=" * 72)
    print("Copy the following into constants.py (fallback lists section):")
    print("=" * 72 + "\n")

    mapping = {
        "AVAILABLE_MODULES": "modules",
        "AVAILABLE_ADDONS": "addons",
        "AVAILABLE_API_PACKAGES": "api_packages",
        "AVAILABLE_GUIDES": "guides",
        "AVAILABLE_TUTORIALS": "tutorials",
    }

    for const_name, key in mapping.items():
        print(_format_list(const_name, pages[key]))
        print()


if __name__ == "__main__":
    main()
