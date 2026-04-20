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

"""Dynamic sitemap discovery for automatic content adaptation."""

import logging
import re

from defusedxml.ElementTree import fromstring as parse_xml

from qiskit_docs_mcp_server.constants import SITEMAP_URL
from qiskit_docs_mcp_server.http import _get_http_client


logger = logging.getLogger(__name__)

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

_sitemap_data: dict[str, list[str]] | None = None

_VERSION_SEGMENT_RE = re.compile(r"/(?:\d+\.\d+|dev)(?:/|$)")


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
    root = parse_xml(xml_text)

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


def get_sitemap_pages() -> dict[str, list[str]] | None:
    """Return the sitemap data loaded at startup, or ``None`` if unavailable."""
    return _sitemap_data


async def load_sitemap() -> None:
    """Fetch and parse the documentation sitemap at server startup.

    Stores the result in a module-level variable so that all subsequent
    resource calls can read it synchronously.
    """
    global _sitemap_data  # noqa: PLW0603

    try:
        client = _get_http_client()
        response = await client.get(SITEMAP_URL, follow_redirects=True)
        response.raise_for_status()
        _sitemap_data = _parse_sitemap_xml(response.text)
        logger.info(
            "Sitemap loaded: %d modules, %d addons, %d api_packages, %d guides, %d tutorials",
            len(_sitemap_data["modules"]),
            len(_sitemap_data["addons"]),
            len(_sitemap_data["api_packages"]),
            len(_sitemap_data["guides"]),
            len(_sitemap_data["tutorials"]),
        )
    except Exception as e:
        logger.warning(f"Failed to fetch sitemap, using fallback constants: {e}")
        _sitemap_data = None
