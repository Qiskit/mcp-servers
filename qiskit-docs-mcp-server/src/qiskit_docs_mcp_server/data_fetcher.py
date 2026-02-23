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
import os
from datetime import datetime
from functools import lru_cache
from typing import Any

import html2text
import httpx


logger = logging.getLogger(__name__)


# Environment variable configuration
def _get_env_float(name: str, default: float) -> float:
    """
    Get environment variable as float with fallback to default.

    Args:
        name: Environment variable name
        default: Default value if not set or invalid

    Returns:
        Float value from environment or default
    """
    try:
        value = os.getenv(name)
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Invalid {name} value: {os.getenv(name)}, using default {default}")
        return default


# Qiskit documentation bases (configurable via environment variables)
QISKIT_DOCS_BASE = os.getenv("QISKIT_DOCS_BASE", "https://docs.quantum.ibm.com/")
QISKIT_SDK_DOCS = os.getenv("QISKIT_SDK_DOCS", "https://docs.quantum.ibm.com/")
QISKIT_RUNTIME_DOCS = os.getenv("QISKIT_RUNTIME_DOCS", "https://docs.quantum.ibm.com/run/")
BASE_URL = os.getenv("QISKIT_SEARCH_BASE_URL", "https://quantum.cloud.ibm.com/")

# HTTP timeout configuration (in seconds)
HTTP_TIMEOUT = _get_env_float("QISKIT_HTTP_TIMEOUT", 10.0)

# Qiskit modules and their documentation paths
QISKIT_MODULES = {
    "circuit": "api/qiskit/circuit",
    "primitives": "api/qiskit/primitives",
    "transpiler": "api/qiskit/transpiler",
    "quantum_info": "api/qiskit/quantum_info",
    "result": "api/qiskit/result",
    "visualization": "api/qiskit/visualization",
}

QISKIT_ADDON_MODULES = {
    "addon-opt-mapper": "guides/qaoa-mapper",
    "addon-qpe": "guides/qpe",
    "addon-vqe": "guides/vqe",
}

SEARCH_PATH = "endpoints-docs-learning/api/search"


def _find_similar(query: str, available: list[str], cutoff: float = 0.6) -> list[str]:
    """
    Find similar strings using difflib SequenceMatcher.

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


@lru_cache(maxsize=50)
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


@lru_cache(maxsize=100)
def fetch_text(url: str) -> str | None:
    """
    Fetch text content from a URL using httpx.

    Args:
        url: The URL to fetch

    Returns:
        The text content of the page, or None if fetch fails
    """
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


def get_component_docs(component: str) -> dict[str, Any]:
    """
    Fetch documentation for a Qiskit SDK module and convert to Markdown.

    Args:
        component: Module name (e.g., 'circuit', 'primitives', 'transpiler')

    Returns:
        The documentation content in Markdown format or error status
    """
    if component not in QISKIT_MODULES:
        suggestions = _find_similar(component, list(QISKIT_MODULES.keys()))
        error_response = {
            "status": "error",
            "message": f"Module '{component}' not found.",
            "available_modules": list(QISKIT_MODULES.keys()),
        }
        if suggestions:
            error_response["suggestions"] = suggestions
        return error_response

    path = QISKIT_MODULES[component]
    url = f"{QISKIT_SDK_DOCS}{path}"
    logger.info(f"Fetching component docs for {component} from {url}")
    html = fetch_text(url)
    if html:
        docs = convert_html_to_markdown(html)
    else:
        docs = None

    return {
        "status": "success",
        "module": component,
        "documentation": docs,
        "metadata": {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "content_type": "markdown",
            "content_length": len(docs) if docs else 0,
        },
    }


def get_guide_docs(guide: str) -> dict[str, Any]:
    """
    Fetch documentation for a Qiskit guide or best practice and convert to Markdown.

    Args:
        guide: Guide name (e.g., 'optimization', 'error-mitigation')

    Returns:
        The documentation content in Markdown format or error status
    """
    guide_paths = {
        "optimization": "guides/optimization",
        "quantum-circuits": "guides/circuits",
        "error-mitigation": "guides/error-mitigation",
        "dynamic-circuits": "guides/dynamic-circuits",
        "parametric-compilation": "guides/parametric-compilation",
        "performance-tuning": "guides/performance-tuning",
    }

    if guide not in guide_paths:
        suggestions = _find_similar(guide, list(guide_paths.keys()))
        error_response = {
            "status": "error",
            "message": f"Guide '{guide}' not found.",
            "available_guides": list(guide_paths.keys()),
        }
        if suggestions:
            error_response["suggestions"] = suggestions
        return error_response

    path = guide_paths[guide]
    url = f"{QISKIT_DOCS_BASE}{path}"
    logger.info(f"Fetching style docs for {guide} from {url}")
    html = fetch_text(url)
    if html:
        docs = convert_html_to_markdown(html)
    else:
        docs = None

    return {
        "status": "success",
        "guide": guide,
        "documentation": docs,
        "metadata": {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "content_type": "markdown",
            "content_length": len(docs) if docs else 0,
        },
    }


def search_qiskit_docs(query: str, module: str = "documentation") -> dict[str, Any]:
    """
    Search Qiskit documentation for relevant results.

    Args:
        query: Search query string
        module: Search module string

    Returns:
        List of relevant documentation entries with name and description
    """

    url = f"{BASE_URL}{SEARCH_PATH}?query={query}&module={module}"
    logger.info(f"Querying from {query} which gives {url} from {module}")

    results = fetch_text_json(url)

    return {
        "status": "success",
        "query": query,
        "module": module,
        "results": results if results else [],
        "total_results": len(results) if results else 0,
        "metadata": {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "content_type": "json",
        },
    }


def fetch_text_json(url: str) -> list[dict]:
    """
    Fetch text content from a URL using httpx.

    Args:
        url: The URL to fetch

    Returns:
        The text content of the page, or None if fetch fails
    """
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e} because of a HTTP error.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {e}")
        return None


def get_list_of_modules() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules."""
    modules = list(QISKIT_MODULES.keys())
    return {"status": "success", "modules": modules}


def get_list_of_addons() -> dict[str, Any]:
    """Get list of all Qiskit addon modules and tutorials."""
    addons = list(QISKIT_ADDON_MODULES.keys())
    return {"status": "success", "addons": addons}


def get_list_of_guides() -> dict[str, Any]:
    """Get list of Qiskit guides and best practices."""
    guides = [
        "optimization",
        "quantum-circuits",
        "error-mitigation",
        "dynamic-circuits",
        "parametric-compilation",
        "performance-tuning",
    ]
    return {"status": "success", "guides": guides}
