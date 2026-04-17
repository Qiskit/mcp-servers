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

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

import html2text
import httpx
from bs4 import BeautifulSoup

from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_API_PACKAGES,
    AVAILABLE_GUIDES,
    AVAILABLE_MODULES,
    AVAILABLE_TUTORIALS,
    BASE_URL,
    CACHE_TTL,
    ERROR_CODE_CATEGORIES,
    HTTP_TIMEOUT,
    QISKIT_DOCS_BASE,
    SEARCH_CACHE_TTL,
    SEARCH_PATH,
    SITEMAP_URL,
)


logger = logging.getLogger(__name__)

# Allowed hostname for URL validation (derived from configurable QISKIT_DOCS_BASE)
_ALLOWED_HOST = urlparse(QISKIT_DOCS_BASE).netloc

# Retry configuration for transient HTTP failures
_MAX_RETRIES = 2  # Total attempts (1 initial + 1 retry)
_RETRY_DELAY = 1.0  # Seconds between retries


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
_sitemap_cache = _TTLCache(ttl=CACHE_TTL)

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_SITEMAP_CACHE_KEY = "sitemap_pages"

# Regex for versioned paths (e.g., /0.46, /1.0, /2.1, /dev)
_VERSION_SEGMENT_RE = re.compile(r"/(?:\d+\.\d+|dev)(?:/|$)")

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


def _classify_page(path: str, buckets: dict[str, set[str]]) -> None:
    """Classify a single English doc path into the appropriate bucket.

    Args:
        path: Path relative to ``/docs/en/`` (e.g. ``guides/transpile``).
        buckets: Mutable dict of sets to add the slug to.
    """
    for prefix, key in (("guides/", "guides"), ("tutorials/", "tutorials")):
        if path.startswith(prefix):
            slug = path[len(prefix) :]
            if slug and "/" not in slug:
                buckets[key].add(slug)
            return

    if not path.startswith("api/"):
        return

    if path.startswith("api/qiskit-addon-"):
        rest = path.removeprefix("api/qiskit-addon-")
        if rest and "/" not in rest:
            buckets["addons"].add(rest)
    elif path.startswith("api/qiskit/"):
        slug = path.removeprefix("api/qiskit/")
        if (
            slug
            and "/" not in slug
            and not slug.startswith("qiskit.")
            and slug not in {"release-notes", "root"}
        ):
            buckets["modules"].add(slug)
    else:
        slug = path.removeprefix("api/")
        if slug and "/" not in slug and slug != "qiskit":
            buckets["api_packages"].add(slug)


def _parse_sitemap_xml(xml_text: str) -> dict[str, list[str]]:
    """Parse sitemap XML and categorize English page paths.

    Extracts ``/en/`` pages from the sitemap and groups them into:
    modules, addons, api_packages, guides, and tutorials.

    Args:
        xml_text: Raw XML string from the sitemap

    Returns:
        Dict with keys 'modules', 'addons', 'api_packages', 'guides',
        'tutorials', each mapping to a sorted list of slug strings.
    """
    root = ET.fromstring(xml_text)

    buckets: dict[str, set[str]] = {
        "modules": set(),
        "addons": set(),
        "api_packages": set(),
        "guides": set(),
        "tutorials": set(),
    }

    en_marker = "/docs/en/"
    for loc in root.iter(f"{_SITEMAP_NS}loc"):
        url = loc.text
        if url is None:
            continue
        idx = url.find(en_marker)
        if idx == -1:
            continue
        path = url[idx + len(en_marker) :]
        if _VERSION_SEGMENT_RE.search(path):
            continue
        _classify_page(path, buckets)

    return {key: sorted(values) for key, values in buckets.items()}


async def _fetch_sitemap_pages() -> dict[str, list[str]] | None:
    """Fetch and parse the documentation sitemap for dynamic page discovery.

    Results are cached using the standard TTL cache.

    Returns:
        Categorized page lists, or None if the sitemap cannot be fetched.
    """
    cached: dict[str, list[str]] | None = _sitemap_cache.get(_SITEMAP_CACHE_KEY)
    if cached is not None:
        return cached

    try:
        client = _get_http_client()
        response = await client.get(SITEMAP_URL, follow_redirects=True)
        response.raise_for_status()
        result = _parse_sitemap_xml(response.text)
        _sitemap_cache.set(_SITEMAP_CACHE_KEY, result)
        logger.info(
            "Sitemap loaded: %d modules, %d addons, %d api_packages, %d guides, %d tutorials",
            len(result["modules"]),
            len(result["addons"]),
            len(result["api_packages"]),
            len(result["guides"]),
            len(result["tutorials"]),
        )
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch sitemap, using fallback constants: {e}")
        return None


def _strip_html_tags(text: str) -> str:
    """Strip HTML tags from a string.

    Args:
        text: String potentially containing HTML tags

    Returns:
        String with all HTML tags removed
    """
    return re.sub(r"<[^>]+>", "", text)


def extract_main_content(html: str) -> str:
    """Extract main content from HTML, removing navigation chrome.

    Strips nav, header, footer, aside elements and ARIA-role navigation,
    then returns the <main>, <article>, or role='main' content. Falls back
    to <body> (with chrome removed) if no semantic main content is found.

    Args:
        html: Full HTML page content

    Returns:
        HTML string with only the main content
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove structural chrome elements
    for tag_name in ["nav", "header", "footer", "aside"]:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Remove ARIA-role navigation elements
    for role in ["navigation", "banner", "contentinfo", "complementary"]:
        for element in soup.find_all(attrs={"role": role}):
            element.decompose()

    # Remove skip-to-content links
    for element in soup.find_all("a", class_=lambda c: c and "skip" in c.lower()):
        element.decompose()
    for element in soup.find_all(
        "a",
        string=lambda s: s and "skip to" in s.lower(),  # type: ignore[call-overload]
    ):
        element.decompose()

    # Return the best semantic container
    main_content = soup.find("main")
    if main_content:
        return str(main_content)

    article = soup.find("article")
    if article:
        return str(article)

    main_role = soup.find(attrs={"role": "main"})
    if main_role:
        return str(main_role)

    body = soup.find("body")
    if body:
        return str(body)

    return str(soup)


_html2text_converter = html2text.HTML2Text()
_html2text_converter.ignore_links = False
_html2text_converter.body_width = 0
_html2text_converter.ignore_images = False


def convert_html_to_markdown(html: str) -> str:
    """Convert HTML content to Markdown format.

    Strips navigation chrome (header, footer, nav, aside) before conversion
    to produce cleaner markdown output.

    Args:
        html: HTML content string

    Returns:
        Markdown formatted content
    """
    content_html = extract_main_content(html)
    return _html2text_converter.handle(content_html)


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

    logger.info("Fetching page docs from %s", resolved_url)
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
    query = query.strip()
    if not query:
        return {
            "status": "error",
            "message": "Please provide a search query.",
        }
    if scope not in _VALID_SCOPES:
        return {
            "status": "error",
            "message": (
                f"Invalid scope '{scope}'. Valid values: {', '.join(sorted(_VALID_SCOPES))}."
            ),
        }

    url = f"{BASE_URL}{SEARCH_PATH}?query={quote(query)}&module={quote(scope)}"
    logger.info("Searching docs for '%s' in scope '%s'", query, scope)

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
    base = QISKIT_DOCS_BASE.rstrip("/")
    for item in results:
        entry = dict(item)
        if "title" in entry:
            entry["title"] = _strip_html_tags(entry["title"])
        if "text" in entry:
            entry["text"] = _strip_html_tags(entry["text"])
        # Normalize URL to full URL if relative
        url_val = entry.get("url")
        if url_val:
            parsed = urlparse(url_val)
            if not parsed.scheme and not parsed.netloc:
                entry["url"] = f"{base}/{url_val.lstrip('/')}"
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
    logger.info("Fetching error code %s from %s", code, url)
    html = await fetch_text(url)

    if not html:
        return {
            "status": "error",
            "message": "Failed to fetch the error code registry.",
            "metadata": {"url": url},
        }

    soup = BeautifulSoup(html, "html.parser")

    # Strategy 1: Search in table rows
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        row_text = " ".join(cell.get_text(strip=True) for cell in cells)
        if re.search(rf"\b{code}\b", row_text):
            details = " | ".join(cell.get_text(strip=True) for cell in cells)
            return {
                "status": "success",
                "code": code,
                "details": details,
                "metadata": {
                    "url": f"{url}#{code[0]}xxx",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_type": "text",
                },
            }

    # Strategy 2: Search in any element containing the code
    code_pattern = re.compile(rf"\b{code}\b")
    for element in soup.find_all(string=code_pattern):
        # Get the parent block element for context
        parent = element.find_parent(
            ["p", "div", "li", "dd", "section", "td", "h1", "h2", "h3", "h4", "h5", "h6"]
        )
        if parent:
            details = parent.get_text(strip=True)
            return {
                "status": "success",
                "code": code,
                "details": details,
                "metadata": {
                    "url": f"{url}#{code[0]}xxx",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content_type": "text",
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


async def get_list_of_modules() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules with URL paths.

    Tries dynamic sitemap discovery first, falls back to hardcoded constants.
    """
    base = QISKIT_DOCS_BASE.rstrip("/")
    sitemap = await _fetch_sitemap_pages()
    names = sitemap["modules"] if sitemap else AVAILABLE_MODULES
    return {
        "status": "success",
        "source": "sitemap" if sitemap else "fallback",
        "modules": [
            {
                "name": name,
                "url_path": f"api/qiskit/{name}",
                "full_url": f"{base}/api/qiskit/{name}",
            }
            for name in names
        ],
    }


async def get_list_of_addons() -> dict[str, Any]:
    """Get list of all Qiskit addon packages with URL paths.

    Tries dynamic sitemap discovery first, falls back to hardcoded constants.
    """
    base = QISKIT_DOCS_BASE.rstrip("/")
    sitemap = await _fetch_sitemap_pages()
    names = sitemap["addons"] if sitemap else AVAILABLE_ADDONS
    return {
        "status": "success",
        "source": "sitemap" if sitemap else "fallback",
        "addons": [
            {
                "name": name,
                "url_path": f"api/qiskit-addon-{name}",
                "full_url": f"{base}/api/qiskit-addon-{name}",
            }
            for name in names
        ],
    }


async def get_list_of_guides() -> dict[str, Any]:
    """Get list of Qiskit guides with URL paths.

    Tries dynamic sitemap discovery first, falls back to hardcoded constants.
    """
    base = QISKIT_DOCS_BASE.rstrip("/")
    sitemap = await _fetch_sitemap_pages()
    names = sitemap["guides"] if sitemap else AVAILABLE_GUIDES
    return {
        "status": "success",
        "source": "sitemap" if sitemap else "fallback",
        "guides": [
            {
                "name": name,
                "url_path": f"guides/{name}",
                "full_url": f"{base}/guides/{name}",
            }
            for name in names
        ],
    }


async def get_list_of_tutorials() -> dict[str, Any]:
    """Get list of Qiskit tutorials with URL paths.

    Tries dynamic sitemap discovery first, falls back to hardcoded constants.
    """
    base = QISKIT_DOCS_BASE.rstrip("/")
    sitemap = await _fetch_sitemap_pages()
    names = sitemap["tutorials"] if sitemap else AVAILABLE_TUTORIALS
    return {
        "status": "success",
        "source": "sitemap" if sitemap else "fallback",
        "tutorials": [
            {
                "name": name,
                "url_path": f"tutorials/{name}",
                "full_url": f"{base}/tutorials/{name}",
            }
            for name in names
        ],
    }


async def get_list_of_api_packages() -> dict[str, Any]:
    """Get list of all API packages (beyond SDK modules and addons) with URL paths.

    Includes qiskit-ibm-runtime, qiskit-ibm-transpiler, REST APIs, etc.
    Tries dynamic sitemap discovery first, falls back to hardcoded constants.
    """
    base = QISKIT_DOCS_BASE.rstrip("/")
    sitemap = await _fetch_sitemap_pages()
    names = sitemap["api_packages"] if sitemap else AVAILABLE_API_PACKAGES
    return {
        "status": "success",
        "source": "sitemap" if sitemap else "fallback",
        "api_packages": [
            {
                "name": name,
                "url_path": f"api/{name}",
                "full_url": f"{base}/api/{name}",
            }
            for name in names
        ],
    }


def get_list_of_error_code_categories() -> dict[str, Any]:
    """Get list of IBM Quantum error code categories."""
    return {
        "status": "success",
        "categories": ERROR_CODE_CATEGORIES,
        "registry_url": f"{QISKIT_DOCS_BASE}errors",
    }
