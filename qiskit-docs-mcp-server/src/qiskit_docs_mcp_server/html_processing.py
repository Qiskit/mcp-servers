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

"""HTML content extraction and markdown conversion."""

import re

import html2text
from bs4 import BeautifulSoup


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
