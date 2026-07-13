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

"""Business logic for fetching and processing Qiskit documentation."""

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlparse

from bs4 import BeautifulSoup

from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_API_PACKAGES,
    AVAILABLE_GUIDES,
    AVAILABLE_MODULES,
    AVAILABLE_TUTORIALS,
    BASE_URL,
    DEFAULT_SEARCH_TOP_K,
    ERROR_CODE_CATEGORIES,
    MAX_SEARCH_TOP_K,
    QISKIT_DOCS_BASE,
    SEARCH_PATH,
    SNIPPET_MAX_CHARS,
    VALID_SEARCH_DETAIL,
)
from qiskit_docs_mcp_server.html_processing import (
    _strip_html_tags,
    convert_html_to_markdown,
)
from qiskit_docs_mcp_server.http import fetch_text, fetch_text_json
from qiskit_docs_mcp_server.sitemap import get_sitemap_pages


logger = logging.getLogger(__name__)

_ALLOWED_HOST = urlparse(QISKIT_DOCS_BASE).netloc


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

# Metadata fields kept from each upstream search result. The full page body
# (the upstream ``text`` field) is deliberately excluded and replaced with a
# short ``snippet``; whitelisting also keeps the response small and stable if
# the upstream API later adds new (potentially large) fields.
_RESULT_META_FIELDS = ("id", "url", "title", "pageTitle", "module", "section")

# Words this short are ignored when locating the query match in a body, so that
# stop-word-like fragments don't anchor the snippet on a noisy position.
_MIN_TERM_LEN = 2

_ELLIPSIS = "…"  # single-character "..."


def _densest_window_center(positions: list[int], width: int) -> int:
    """Return the center of the densest cluster of match positions.

    Given sorted character offsets where query terms occur, find the window of
    ``width`` characters that contains the most matches and return the midpoint
    of that cluster (a good place to center a snippet). Returns -1 if there are
    no positions.
    """
    if not positions:
        return -1
    best_count = 0
    best_center = positions[0]
    left = 0
    for right, pos in enumerate(positions):
        while pos - positions[left] > width:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_center = (positions[left] + pos) // 2
    return best_center


def _trim_partial_words(window: str, *, head: bool, tail: bool) -> str:
    """Trim partial words at clipped edges and add ellipsis markers.

    Args:
        window: The raw character window sliced out of the body.
        head: True if the window was clipped at the start (text precedes it).
        tail: True if the window was clipped at the end (text follows it).
    """
    if head:
        # Drop a leading partial word (text before the first space).
        space = window.find(" ")
        if space != -1:
            window = window[space + 1 :]
    if tail:
        # Drop a trailing partial word (text after the last space).
        space = window.rfind(" ")
        if space != -1:
            window = window[:space]
    window = window.strip()
    if head:
        window = f"{_ELLIPSIS} {window}"
    if tail:
        window = f"{window} {_ELLIPSIS}"
    return window


def _make_snippet(text: str, query: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Build a compact, query-centered excerpt from a documentation body.

    Whitespace is collapsed so the character budget reflects real content. When
    the query (or its terms) appears in the body, the snippet is a window
    centered on the densest cluster of matches; otherwise it falls back to the
    start of the body. Partial words at clipped edges are trimmed and marked
    with an ellipsis.

    Args:
        text: Plain-text body (HTML already stripped).
        query: The search query, used to locate the most relevant window.
        max_chars: Maximum length of the returned snippet (excluding ellipses).

    Returns:
        A snippet no longer than roughly ``max_chars`` characters.
    """
    if not text:
        return ""

    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized

    haystack = normalized.lower()

    # Prefer the full query phrase; fall back to the densest cluster of terms.
    # Guard against an empty phrase: ``"".find`` is 0, which would otherwise
    # short-circuit to a head-anchored window and skip term clustering.
    phrase = query.strip().lower()
    anchor = haystack.find(phrase) if phrase else -1
    if anchor != -1:
        # Center on the phrase's middle so a long phrase isn't clipped.
        anchor += len(phrase) // 2
    else:
        terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) >= _MIN_TERM_LEN}
        positions = sorted(
            match.start() for term in terms for match in re.finditer(re.escape(term), haystack)
        )
        anchor = _densest_window_center(positions, max_chars)

    if anchor == -1:
        # No query term present (e.g. a semantic match) — use the body's head.
        return _trim_partial_words(normalized[:max_chars], head=False, tail=True)

    # Center a window of max_chars on the anchor, clamped to the body bounds.
    start = max(0, anchor - max_chars // 2)
    end = min(len(normalized), start + max_chars)
    start = max(0, end - max_chars)  # pull the start back if we hit the tail
    return _trim_partial_words(normalized[start:end], head=start > 0, tail=end < len(normalized))


def _normalize_result_url(url_val: Any, base: str) -> Any:
    """Resolve a relative search-result URL to a full documentation URL."""
    if not url_val or not isinstance(url_val, str):
        return url_val
    parsed = urlparse(url_val)
    if not parsed.scheme and not parsed.netloc:
        return f"{base}/{url_val.lstrip('/')}"
    return url_val


def _build_result_entry(item: dict[str, Any], query: str, detail: str, base: str) -> dict[str, Any]:
    """Build a single search-result entry.

    In ``snippet`` mode (default) the entry is trimmed to a small set of
    navigation fields plus a short ``snippet`` — token-economical for agent use.
    In ``full`` mode the original upstream fields are preserved (backwards
    compatible with the pre-snippet response) and the full page body is returned
    as ``text``. Either way the upstream ``item`` is not mutated (dict/comprehension
    copies), so cached search responses stay intact.
    """
    if detail == "full":
        # Preserve all upstream fields for backwards compatibility.
        entry: dict[str, Any] = dict(item)
        raw_text = item.get("text")
        if isinstance(raw_text, str):
            entry["text"] = _strip_html_tags(raw_text)
    else:
        entry = {key: item[key] for key in _RESULT_META_FIELDS if item.get(key) is not None}
        raw_text = item.get("text")
        clean_text = _strip_html_tags(raw_text) if isinstance(raw_text, str) else ""
        entry["snippet"] = _make_snippet(clean_text, query)

    if isinstance(entry.get("title"), str):
        entry["title"] = _strip_html_tags(entry["title"])
    if "url" in entry:
        entry["url"] = _normalize_result_url(entry["url"], base)
    return entry


def _resolve_effective_top_k(top_k: int | None, detail: str, total_results: int) -> int:
    """Resolve how many results to return.

    A ``None`` or non-positive ``top_k`` means "no explicit limit":

    * In ``snippet`` mode (token-economical default) it falls back to
      ``DEFAULT_SEARCH_TOP_K``.
    * In ``full`` mode (the backwards-compatible option for issue #218) it
      returns the *complete* result set, matching the pre-snippet behavior of
      returning every match with full text.

    An explicit positive ``top_k`` is honored. In ``snippet`` mode it is clamped
    to ``[1, MAX_SEARCH_TOP_K]`` so a single agent-loop call can't balloon; in
    ``full`` mode it is only floored at 1 (never clamped to the max), since full
    is an opt-in heavy path where the caller has asked for everything.
    """
    if top_k is None or top_k <= 0:
        # No explicit limit: full mode returns everything, snippet mode falls
        # back to the default (still clamped below).
        if detail == "full":
            return total_results
        top_k = DEFAULT_SEARCH_TOP_K
    # top_k is now a positive int. Full mode honors it as-is; snippet mode
    # clamps to [1, MAX_SEARCH_TOP_K]. The final max(1, ...) keeps the slice
    # valid even if MAX_SEARCH_TOP_K were misconfigured below 1.
    if detail == "full":
        return top_k
    return max(1, min(top_k, MAX_SEARCH_TOP_K))


async def search_qiskit_docs(
    query: str,
    scope: str = "all",
    top_k: int | None = None,
    detail: str = "snippet",
) -> dict[str, Any]:
    """Search Qiskit documentation for relevant results.

    Returns concise, query-centered snippets by default so the response stays
    small enough for repeated use inside an LLM agent loop. Use ``get_page_docs``
    with a result's ``url`` to read a page in full.

    Args:
        query: Search query string
        scope: Search scope filter. Valid values (case-sensitive):
            'all', 'documentation', 'api', 'learning', 'tutorials'
        top_k: Maximum number of results to return. When None or non-positive
            (the default), snippet mode falls back to DEFAULT_SEARCH_TOP_K and
            full mode returns every match. An explicit value is clamped to
            [1, MAX_SEARCH_TOP_K] in snippet mode and only floored at 1 in
            full mode.
        detail: 'snippet' (default) returns a short excerpt per result, a
            trimmed field set, and at most DEFAULT_SEARCH_TOP_K/top_k results;
            'full' restores the pre-snippet behavior — every match by default,
            each with its full page body as 'text' and all original upstream
            fields (backwards compatible, heavier, so prefer get_page for a
            single page's full content).

    Returns:
        Search results with matching entries, counts, and metadata.
        'total_results' is the grand total of matches found; 'returned_results'
        is the (possibly smaller) number included after the top_k cap, with a
        'truncated' flag. Each entry carries 'id', 'url', 'title', 'pageTitle',
        'module', 'section' (when available) and either a 'snippet' or, for
        detail='full', a 'text' body.
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
    if detail not in VALID_SEARCH_DETAIL:
        return {
            "status": "error",
            "message": (
                f"Invalid detail '{detail}'. Valid values: {', '.join(VALID_SEARCH_DETAIL)}."
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

    # `total_results` keeps its pre-PR meaning: the grand total of matches found.
    # `returned_results` is the (possibly smaller) count actually included here.
    total_results = len(results)
    base = QISKIT_DOCS_BASE.rstrip("/")
    effective_top_k = _resolve_effective_top_k(top_k, detail, total_results)
    selected = results[:effective_top_k]
    cleaned = [_build_result_entry(item, query, detail, base) for item in selected]
    truncated = total_results > len(cleaned)

    if detail == "snippet":
        note = "Showing snippets; call get_page_tool with a result's url for full page content."
    else:
        note = "Showing full page bodies; for a single page, get_page_tool is more economical."
    if truncated:
        note = (
            f"Showing top {len(cleaned)} of {total_results} matches. "
            "Refine the query for fewer, more relevant results. " + note
        )

    return {
        "status": "success",
        "query": query,
        "scope": scope,
        "detail": detail,
        "results": cleaned,
        "total_results": total_results,
        "returned_results": len(cleaned),
        "truncated": truncated,
        "note": note,
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
    sitemap = get_sitemap_pages()
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
    sitemap = get_sitemap_pages()
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
    sitemap = get_sitemap_pages()
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
    sitemap = get_sitemap_pages()
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
    sitemap = get_sitemap_pages()
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
