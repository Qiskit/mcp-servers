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

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

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

# Allowed hostname for URL validation (derived from configurable QISKIT_DOCS_BASE)
_ALLOWED_HOST = urlparse(QISKIT_DOCS_BASE).netloc


def _strip_html_tags(text: str) -> str:
    """Strip HTML tags from a string.

    Args:
        text: String potentially containing HTML tags

    Returns:
        String with all HTML tags removed
    """
    return re.sub(r"<[^>]+>", "", text)


def convert_html_to_markdown(html: str) -> str:
    """Convert HTML content to Markdown format.

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


def _truncate_content(content: str, max_length: int = 20000, offset: int = 0) -> dict[str, Any]:
    """Truncate content with pagination metadata.

    Args:
        content: The full content string
        max_length: Maximum number of characters to return (0 for unlimited)
        offset: Character offset to start from (negative values clamped to 0)

    Returns:
        Dict with 'content', 'has_more', 'offset', 'next_offset', 'total_length'
    """
    # Clamp invalid inputs
    offset = max(0, offset)
    max_length = max(0, max_length)

    total_length = len(content)

    # Clamp offset to content length
    offset = min(offset, total_length)

    if max_length <= 0:
        return {
            "content": content[offset:] if offset > 0 else content,
            "has_more": False,
            "offset": offset,
            "next_offset": None,
            "total_length": total_length,
        }

    # Apply offset
    sliced = content[offset:]

    if len(sliced) <= max_length:
        return {
            "content": sliced,
            "has_more": False,
            "offset": offset,
            "next_offset": None,
            "total_length": total_length,
        }

    # Truncate at a line boundary if possible
    truncated = sliced[:max_length]
    last_newline = truncated.rfind("\n")
    if last_newline > max_length * 0.8:  # Only snap to newline if reasonably close
        truncated = truncated[: last_newline + 1]

    next_offset = offset + len(truncated)

    return {
        "content": truncated,
        "has_more": True,
        "offset": offset,
        "next_offset": next_offset,
        "total_length": total_length,
    }


def _resolve_url(url: str) -> str:
    """Resolve a URL or relative path to a full documentation URL.

    Args:
        url: Full URL or path relative to docs base
            (e.g., 'guides/transpile' or 'api/qiskit/circuit')

    Returns:
        Full resolved URL

    Raises:
        ValueError: If the URL is outside the allowed documentation domain
    """
    # If it's already a full URL, validate the domain
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        if parsed.netloc != _ALLOWED_HOST:
            msg = (
                f"URL domain '{parsed.netloc}' is not allowed. "
                f"Only URLs from '{_ALLOWED_HOST}' are supported."
            )
            raise ValueError(msg)
        return url

    # Relative path — resolve against the docs base
    # Strip leading slash if present
    path = url.lstrip("/")
    base = QISKIT_DOCS_BASE.rstrip("/")
    return f"{base}/{path}"


async def fetch_text(url: str) -> str | None:
    """Fetch text content from a URL using httpx.

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
    """Fetch JSON content from a URL using httpx.

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


async def get_page_docs(url: str, max_length: int = 20000, offset: int = 0) -> dict[str, Any]:
    """Fetch any Qiskit documentation page and return as markdown.

    Accepts full URLs or relative paths. Validates that the URL is within
    the allowed documentation domain. Supports pagination for large pages.

    Args:
        url: Full URL or relative path (e.g., 'guides/transpile',
            'api/qiskit/circuit', 'api/qiskit/qiskit.circuit.QuantumCircuit')
        max_length: Maximum number of characters to return (0 for unlimited)
        offset: Character offset to start from for pagination

    Returns:
        Documentation content in markdown with metadata, or error status
    """
    try:
        resolved_url = _resolve_url(url)
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e),
        }

    logger.info(f"Fetching page docs from {resolved_url}")
    html = await fetch_text(resolved_url)

    if html is None:
        return {
            "status": "error",
            "message": (
                f"Failed to fetch '{url}'. The page may not exist. "
                "Try using search_docs_tool to find the correct URL."
            ),
            "metadata": {
                "url": resolved_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    docs = convert_html_to_markdown(html)

    # Apply pagination
    paginated = _truncate_content(docs, max_length=max_length, offset=offset)

    return {
        "status": "success",
        "url": resolved_url,
        "documentation": paginated["content"],
        "has_more": paginated["has_more"],
        "next_offset": paginated["next_offset"],
        "total_length": paginated["total_length"],
        "metadata": {
            "url": resolved_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "markdown",
            "content_length": len(paginated["content"]),
        },
    }


_VALID_SCOPES = {"all", "documentation", "api", "learning", "tutorials"}


async def search_qiskit_docs(query: str, scope: str = "all") -> dict[str, Any]:
    """Search Qiskit documentation for relevant results.

    Args:
        query: Search query string
        scope: Search scope filter. Valid values (case-sensitive):
            'all', 'documentation', 'api', 'learning', 'tutorials'

    Returns:
        Search results with matching entries, total count, and metadata
    """
    if scope not in _VALID_SCOPES:
        return {
            "status": "error",
            "message": (
                f"Invalid scope '{scope}'. Valid values: {', '.join(sorted(_VALID_SCOPES))}."
            ),
        }

    url = f"{BASE_URL}{SEARCH_PATH}?query={quote(query)}&module={quote(scope)}"
    logger.info(f"Searching docs for '{query}' in scope '{scope}'")

    results = await fetch_text_json(url)

    if results is None:
        return {
            "status": "error",
            "message": f"Failed to search documentation for query '{query}'.",
            "metadata": {
                "url": url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

    # Strip HTML tags from search result fields (shallow copy to avoid mutating cached data)
    cleaned = []
    for item in results:
        entry = dict(item)
        if "title" in entry:
            entry["title"] = _strip_html_tags(entry["title"])
        if "text" in entry:
            entry["text"] = _strip_html_tags(entry["text"])
        cleaned.append(entry)

    return {
        "status": "success",
        "query": query,
        "scope": scope,
        "results": cleaned,
        "total_results": len(cleaned),
        "metadata": {
            "url": url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_type": "json",
        },
    }


async def lookup_error_code(code: str) -> dict[str, Any]:
    """Look up a Qiskit error code and return its message and solution.

    Args:
        code: Error code string (e.g., '1002', '7001')

    Returns:
        Error code details including message and solution, or error status
    """
    if not re.fullmatch(r"\d{4}", code):
        return {
            "status": "error",
            "message": (
                f"Invalid error code format: '{code}'. Expected a 4-digit code (e.g., '1002')."
            ),
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
    lines = docs.split("\n")
    matching_lines: list[str] = []
    for i, line in enumerate(lines):
        if re.search(rf"\b{code}\b", line):
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
    """Get list of all Qiskit SDK modules with descriptions and URL paths."""
    return {
        "status": "success",
        "modules": [
            {"name": name, "description": desc, "url_path": f"api/qiskit/{name}"}
            for name, desc in AVAILABLE_MODULES.items()
        ],
    }


def get_list_of_addons() -> dict[str, Any]:
    """Get list of all Qiskit addon modules with descriptions and URL paths."""
    return {
        "status": "success",
        "addons": [
            {"name": name, "description": desc, "url_path": f"api/qiskit-addon-{name}"}
            for name, desc in AVAILABLE_ADDONS.items()
        ],
    }


def get_list_of_guides() -> dict[str, Any]:
    """Get list of Qiskit guides and best practices with descriptions and URL paths."""
    return {
        "status": "success",
        "guides": [
            {"name": name, "description": desc, "url_path": f"guides/{name}"}
            for name, desc in AVAILABLE_GUIDES.items()
        ],
    }


def get_list_of_error_code_categories() -> dict[str, Any]:
    """Get list of IBM Quantum error code categories."""
    return {
        "status": "success",
        "categories": ERROR_CODE_CATEGORIES,
        "registry_url": f"{QISKIT_DOCS_BASE}errors",
    }
