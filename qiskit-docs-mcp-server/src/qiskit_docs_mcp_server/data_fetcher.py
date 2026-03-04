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

import difflib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import html2text
import httpx

from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_GUIDES,
    AVAILABLE_MODULES,
    BASE_URL,
    ERROR_CODE_CATEGORIES,
    HTTP_TIMEOUT,
    QISKIT_DOCS_BASE,
    SEARCH_PATH,
)


logger = logging.getLogger(__name__)


def _find_similar(query: str, available: list[str], cutoff: float = 0.6) -> list[str]:
    """
    Find similar strings using difflib.get_close_matches.

    Args:
        query: The query string to match
        available: List of available options to search
        cutoff: Similarity threshold (0.0-1.0), default 0.6

    Returns:
        List of similar strings, sorted by similarity (highest first)
    """
    if not query or not available:
        return []

    matches = difflib.get_close_matches(query, available, n=3, cutoff=cutoff)
    return matches


def convert_html_to_markdown(html: str) -> str:
    """
    Convert HTML content to Markdown format.

    Args:
        html: HTML content string

    Returns:
        Markdown formatted content
    """
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.ignore_images = False
    return h.handle(html)


async def fetch_text(url: str) -> str | None:
    """
    Fetch text content from a URL using httpx.

    Args:
        url: The URL to fetch

    Returns:
        The text content of the page, or None if fetch fails
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


async def fetch_text_json(url: str) -> list[dict[str, Any]] | None:
    """
    Fetch JSON content from a URL using httpx.

    Args:
        url: The URL to fetch

    Returns:
        The JSON content as a list of dicts, or None if fetch fails
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


async def get_component_docs(component: str) -> dict[str, Any]:
    """
    Fetch documentation for a Qiskit SDK module and convert to Markdown.

    Args:
        component: Module name (e.g., 'circuit', 'primitives', 'transpiler')

    Returns:
        The documentation content in Markdown format or error status
    """
    if component not in AVAILABLE_MODULES:
        suggestions = _find_similar(component, AVAILABLE_MODULES)
        error_response = {
            "status": "error",
            "message": f"Module '{component}' not found.",
            "available_modules": AVAILABLE_MODULES,
        }
        if suggestions:
            error_response["suggestions"] = suggestions
        return error_response

    url = f"{QISKIT_DOCS_BASE}api/qiskit/{component}"
    logger.info(f"Fetching component docs for {component} from {url}")
    html = await fetch_text(url)
    docs = convert_html_to_markdown(html) if html else None

    return {
        "status": "success",
        "module": component,
        "documentation": docs,
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "markdown",
            "content_length": len(docs) if docs else 0,
        },
    }


async def get_guide_docs(guide: str) -> dict[str, Any]:
    """
    Fetch documentation for a Qiskit guide or best practice and convert to Markdown.

    Args:
        guide: Guide name (e.g., 'quick-start', 'transpile', 'configure-error-mitigation')

    Returns:
        The documentation content in Markdown format or error status
    """
    if guide not in AVAILABLE_GUIDES:
        suggestions = _find_similar(guide, AVAILABLE_GUIDES)
        error_response = {
            "status": "error",
            "message": f"Guide '{guide}' not found.",
            "available_guides": AVAILABLE_GUIDES,
        }
        if suggestions:
            error_response["suggestions"] = suggestions
        return error_response

    url = f"{QISKIT_DOCS_BASE}guides/{guide}"
    logger.info(f"Fetching style docs for {guide} from {url}")
    html = await fetch_text(url)
    docs = convert_html_to_markdown(html) if html else None

    return {
        "status": "success",
        "guide": guide,
        "documentation": docs,
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "markdown",
            "content_length": len(docs) if docs else 0,
        },
    }


async def search_qiskit_docs(query: str, module: str = "documentation") -> dict[str, Any]:
    """
    Search Qiskit documentation for relevant results.

    Args:
        query: Search query string (e.g., 'circuit', 'error mitigation')
        module: Search scope (e.g., 'documentation', 'API')

    Returns:
        Search results with matching entries, total count, and metadata
    """

    url = f"{BASE_URL}{SEARCH_PATH}?query={query}&module={module}"
    logger.info(f"Querying from {query} which gives {url} from {module}")

    results = await fetch_text_json(url)

    return {
        "status": "success",
        "query": query,
        "module": module,
        "results": results if results else [],
        "total_results": len(results) if results else 0,
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "json",
        },
    }


async def lookup_error_code(code: str) -> dict[str, Any]:
    """
    Look up a Qiskit error code and return its message and solution.

    Args:
        code: Error code string (e.g., '1002', '7001')

    Returns:
        Error code details including message and solution, or error status
    """
    if not re.fullmatch(r"\d{4}", code):
        return {
            "status": "error",
            "message": f"Invalid error code format: '{code}'. Expected a 4-digit code (e.g., '1002').",
        }

    url = f"{QISKIT_DOCS_BASE}errors"
    logger.info(f"Fetching error code {code} from {url}")
    html = await fetch_text(url)

    if not html:
        return {
            "status": "error",
            "message": "Failed to fetch the error code registry.",
            "metadata": {"url": url},
        }

    docs = convert_html_to_markdown(html)

    # Search for the error code in the markdown content
    # Error codes appear as table rows: | code | message | solution |
    lines = docs.split("\n")
    matching_lines: list[str] = []
    for i, line in enumerate(lines):
        if re.search(rf"\b{code}\b", line):
            # Grab surrounding context (table header + matching row)
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            matching_lines.extend(lines[start:end])
            break

    if matching_lines:
        return {
            "status": "success",
            "code": code,
            "details": "\n".join(matching_lines).strip(),
            "metadata": {
                "url": f"{url}#{code[0]}xxx",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "content_type": "markdown",
            },
        }

    return {
        "status": "error",
        "message": f"Error code '{code}' not found in the registry.",
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def get_list_of_modules() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules."""
    return {"status": "success", "modules": AVAILABLE_MODULES}


async def get_addon_docs(addon: str) -> dict[str, Any]:
    """
    Fetch documentation for a Qiskit addon module and convert to Markdown.

    Args:
        addon: Addon name (e.g., 'sqd', 'cutting', 'mpf')

    Returns:
        The documentation content in Markdown format or error status
    """
    if addon not in AVAILABLE_ADDONS:
        suggestions = _find_similar(addon, AVAILABLE_ADDONS)
        error_response = {
            "status": "error",
            "message": f"Addon '{addon}' not found.",
            "available_addons": AVAILABLE_ADDONS,
        }
        if suggestions:
            error_response["suggestions"] = suggestions
        return error_response

    url = f"{QISKIT_DOCS_BASE}api/qiskit-addon-{addon}"
    logger.info(f"Fetching addon docs for {addon} from {url}")
    html = await fetch_text(url)
    docs = convert_html_to_markdown(html) if html else None

    return {
        "status": "success",
        "addon": addon,
        "documentation": docs,
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "markdown",
            "content_length": len(docs) if docs else 0,
        },
    }


def get_list_of_addons() -> dict[str, Any]:
    """Get list of all Qiskit addon modules."""
    return {"status": "success", "addons": AVAILABLE_ADDONS}


def get_list_of_guides() -> dict[str, Any]:
    """Get list of Qiskit guides and best practices."""
    return {"status": "success", "guides": AVAILABLE_GUIDES}


def get_list_of_error_code_categories() -> dict[str, Any]:
    """Get list of IBM Quantum error code categories."""
    return {
        "status": "success",
        "categories": ERROR_CODE_CATEGORIES,
        "registry_url": f"{QISKIT_DOCS_BASE}errors",
    }
