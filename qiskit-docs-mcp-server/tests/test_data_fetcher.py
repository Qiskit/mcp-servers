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

"""Tests for data_fetcher module."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_GUIDES,
    AVAILABLE_MODULES,
    HTTP_TIMEOUT,
    _get_env_float,
)
from qiskit_docs_mcp_server.data_fetcher import (
    _resolve_url,
    _strip_html_tags,
    _truncate_content,
    convert_html_to_markdown,
    fetch_text,
    fetch_text_json,
    get_list_of_addons,
    get_list_of_error_code_categories,
    get_list_of_guides,
    get_list_of_modules,
    get_page_docs,
    lookup_error_code,
    search_qiskit_docs,
)


class TestFetchText:
    """Test fetch_text function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_success(self, mock_client_class):
        """Test successful text fetch."""
        mock_response = MagicMock()
        mock_response.text = "Sample documentation"
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text("https://example.com")
        assert result == "Sample documentation"

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_http_error(self, mock_client_class):
        """Test fetch_text with HTTP error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text("https://example.com")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_generic_exception(self, mock_client_class):
        """Test fetch_text with generic exception."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Unexpected error")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text("https://example.com")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_timeout(self, mock_client_class):
        """Test fetch_text with timeout."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text("https://example.com")
        assert result is None


class TestFetchTextJson:
    """Test fetch_text_json function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_json_success(self, mock_client_class):
        """Test successful JSON fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"key": "value"}]
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text_json("https://example.com/api")
        assert result == [{"key": "value"}]

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_json_http_error(self, mock_client_class):
        """Test fetch_text_json with HTTP error."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text_json("https://example.com/api")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_json_generic_exception(self, mock_client_class):
        """Test fetch_text_json with generic exception."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Unexpected error")
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text_json("https://example.com/api")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_json_returns_list(self, mock_client_class):
        """Test that fetch_text_json returns list."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"name": "test"}]
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        result = await fetch_text_json("https://example.com")
        assert isinstance(result, list)


class TestResolveUrl:
    """Test URL resolution and validation."""

    def test_resolve_relative_path(self):
        """Test resolving a relative path."""
        url = _resolve_url("guides/transpile")
        assert url == "https://quantum.cloud.ibm.com/docs/guides/transpile"

    def test_resolve_relative_path_with_leading_slash(self):
        """Test resolving a relative path with leading slash."""
        url = _resolve_url("/guides/transpile")
        assert url == "https://quantum.cloud.ibm.com/docs/guides/transpile"

    def test_resolve_api_path(self):
        """Test resolving an API path."""
        url = _resolve_url("api/qiskit/circuit")
        assert url == "https://quantum.cloud.ibm.com/docs/api/qiskit/circuit"

    def test_resolve_class_path(self):
        """Test resolving a class-level API path."""
        url = _resolve_url("api/qiskit/qiskit.circuit.QuantumCircuit")
        assert "qiskit.circuit.QuantumCircuit" in url

    def test_full_url_allowed_domain(self):
        """Test that full URLs with allowed domain pass through."""
        url = _resolve_url("https://quantum.cloud.ibm.com/docs/guides/transpile")
        assert url == "https://quantum.cloud.ibm.com/docs/guides/transpile"

    def test_full_url_disallowed_domain(self):
        """Test that URLs with disallowed domains raise ValueError."""
        with pytest.raises(ValueError, match="not allowed"):
            _resolve_url("https://evil.com/malicious")

    def test_resolve_addon_path(self):
        """Test resolving an addon path."""
        url = _resolve_url("api/qiskit-addon-sqd")
        assert url == "https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd"

    def test_resolve_empty_string(self):
        """Test resolving an empty string returns base URL."""
        url = _resolve_url("")
        assert url == "https://quantum.cloud.ibm.com/docs/"


class TestGetPageDocs:
    """Test get_page_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_relative_path(self, mock_fetch):
        """Test fetching a page by relative path."""
        mock_fetch.return_value = "<h1>Circuit</h1><p>Documentation</p>"
        result = await get_page_docs("api/qiskit/circuit")
        assert result["status"] == "success"
        assert "documentation" in result
        assert "metadata" in result
        assert "Circuit" in result["documentation"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_full_url(self, mock_fetch):
        """Test fetching a page by full URL."""
        mock_fetch.return_value = "<h1>Guide</h1>"
        result = await get_page_docs("https://quantum.cloud.ibm.com/docs/guides/transpile")
        assert result["status"] == "success"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_fetch_fails(self, mock_fetch):
        """Test get_page when fetch fails."""
        mock_fetch.return_value = None
        result = await get_page_docs("api/qiskit/nonexistent")
        assert result["status"] == "error"
        assert "search_docs_tool" in result["message"]

    async def test_get_page_disallowed_domain(self):
        """Test get_page rejects disallowed domains."""
        result = await get_page_docs("https://evil.com/page")
        assert result["status"] == "error"
        assert "not allowed" in result["message"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_has_metadata(self, mock_fetch):
        """Test that get_page response includes metadata."""
        mock_fetch.return_value = "<p>Content</p>"
        result = await get_page_docs("guides/quick-start")
        assert "metadata" in result
        assert "url" in result["metadata"]
        assert "timestamp" in result["metadata"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_pagination_has_more(self, mock_fetch):
        """Test that get_page_docs returns pagination fields."""
        mock_fetch.return_value = "<p>" + "x" * 50000 + "</p>"
        result = await get_page_docs("api/qiskit/circuit", max_length=1000)
        assert result["status"] == "success"
        assert result["has_more"] is True
        assert result["next_offset"] is not None
        assert result["total_length"] > 1000

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_pagination_with_offset(self, mock_fetch):
        """Test pagination with offset retrieves subsequent content."""
        mock_fetch.return_value = "<p>" + "A" * 100 + "</p>"
        result = await get_page_docs("api/qiskit/circuit", max_length=50, offset=10)
        assert result["status"] == "success"
        assert "has_more" in result

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_page_unlimited_length(self, mock_fetch):
        """Test max_length=0 returns all content without truncation."""
        mock_fetch.return_value = "<p>" + "y" * 50000 + "</p>"
        result = await get_page_docs("api/qiskit/circuit", max_length=0)
        assert result["status"] == "success"
        assert result["has_more"] is False


class TestTruncateContent:
    """Test _truncate_content function."""

    def test_short_content_no_truncation(self):
        """Test that short content is not truncated."""
        result = _truncate_content("hello world", max_length=100)
        assert result["content"] == "hello world"
        assert result["has_more"] is False
        assert result["next_offset"] is None

    def test_long_content_truncated(self):
        """Test that content exceeding max_length is truncated."""
        content = "a" * 200
        result = _truncate_content(content, max_length=100)
        assert len(result["content"]) <= 100
        assert result["has_more"] is True
        assert result["next_offset"] is not None
        assert result["total_length"] == 200

    def test_offset_skips_content(self):
        """Test that offset skips the beginning of content."""
        content = "0123456789"
        result = _truncate_content(content, max_length=100, offset=5)
        assert result["content"] == "56789"
        assert result["has_more"] is False

    def test_unlimited_returns_all(self):
        """Test max_length=0 returns all content."""
        content = "a" * 50000
        result = _truncate_content(content, max_length=0)
        assert result["content"] == content
        assert result["has_more"] is False

    def test_negative_offset_clamped_to_zero(self):
        """Test that negative offset is clamped to 0."""
        result = _truncate_content("hello", max_length=100, offset=-5)
        assert result["content"] == "hello"
        assert result["offset"] == 0

    def test_negative_max_length_treated_as_unlimited(self):
        """Test that negative max_length is treated as unlimited (clamped to 0)."""
        content = "a" * 500
        result = _truncate_content(content, max_length=-10)
        assert result["content"] == content
        assert result["has_more"] is False

    def test_truncation_snaps_to_line_boundary(self):
        """Test that truncation snaps to a nearby newline boundary."""
        lines = "\n".join(["line " + str(i) for i in range(50)])
        result = _truncate_content(lines, max_length=100)
        assert result["content"].endswith("\n")
        assert result["has_more"] is True


class TestStripHtmlTags:
    """Test HTML tag stripping."""

    def test_strip_em_tags(self):
        """Test stripping em tags."""
        assert _strip_html_tags("<em>Transpiler</em> stages") == "Transpiler stages"

    def test_strip_multiple_tags(self):
        """Test stripping multiple tags."""
        assert _strip_html_tags("<em>error</em> <strong>mitigation</strong>") == "error mitigation"

    def test_no_tags(self):
        """Test string without tags."""
        assert _strip_html_tags("plain text") == "plain text"

    def test_empty_string(self):
        """Test empty string."""
        assert _strip_html_tags("") == ""


class TestSearchQiskitDocs:
    """Test search_qiskit_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_with_results(self, mock_fetch):
        """Test search with results."""
        mock_fetch.return_value = [
            {"title": "Circuit", "url": "/docs/api/qiskit/circuit"},
        ]
        result = await search_qiskit_docs("circuit")
        assert result["status"] == "success"
        assert result["query"] == "circuit"
        assert len(result["results"]) == 1
        assert result["total_results"] == 1

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_no_results(self, mock_fetch):
        """Test search with no results."""
        mock_fetch.return_value = []
        result = await search_qiskit_docs("nonexistent")
        assert result["status"] == "success"
        assert result["results"] == []
        assert result["total_results"] == 0

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_fetch_fails(self, mock_fetch):
        """Test search when fetch fails."""
        mock_fetch.return_value = None
        result = await search_qiskit_docs("circuit")
        assert result["status"] == "error"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_strips_html_tags(self, mock_fetch):
        """Test that HTML tags are stripped from search results."""
        mock_fetch.return_value = [
            {
                "title": "<em>Transpiler</em> options",
                "text": "<em>Transpiler</em> passes",
            },
        ]
        result = await search_qiskit_docs("transpiler")
        assert result["results"][0]["title"] == "Transpiler options"
        assert result["results"][0]["text"] == "Transpiler passes"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_uses_scope_param(self, mock_fetch):
        """Test that scope parameter is passed to API."""
        mock_fetch.return_value = []
        await search_qiskit_docs("test", scope="api")
        call_url = mock_fetch.call_args[0][0]
        assert "module=api" in call_url

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_default_scope_is_all(self, mock_fetch):
        """Test that default scope is 'all'."""
        mock_fetch.return_value = []
        await search_qiskit_docs("test")
        call_url = mock_fetch.call_args[0][0]
        assert "module=all" in call_url

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_url_encodes_query(self, mock_fetch):
        """Test that search query is URL-encoded."""
        mock_fetch.return_value = []
        await search_qiskit_docs("error mitigation")
        call_url = mock_fetch.call_args[0][0]
        assert "error%20mitigation" in call_url

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_results_missing_title_text_keys(self, mock_fetch):
        """Test that search results without title/text keys don't error."""
        mock_fetch.return_value = [
            {"url": "/docs/api/qiskit/circuit"},
        ]
        result = await search_qiskit_docs("circuit")
        assert result["status"] == "success"
        assert len(result["results"]) == 1


class TestLookupErrorCode:
    """Test lookup_error_code function."""

    async def test_invalid_code_format_letters(self):
        """Test that non-numeric codes return an error."""
        result = await lookup_error_code("abcd")
        assert result["status"] == "error"
        assert "Invalid error code format" in result["message"]

    async def test_invalid_code_format_short(self):
        """Test that codes with wrong length return an error."""
        result = await lookup_error_code("12")
        assert result["status"] == "error"

    async def test_invalid_code_format_long(self):
        """Test that 5-digit codes return an error."""
        result = await lookup_error_code("12345")
        assert result["status"] == "error"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_fetch_failure(self, mock_fetch):
        """Test lookup when fetch fails."""
        mock_fetch.return_value = None
        result = await lookup_error_code("1002")
        assert result["status"] == "error"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_code_found(self, mock_fetch):
        """Test successful error code lookup."""
        mock_fetch.return_value = (
            "<table><tr><td>1002</td>"
            "<td>Error in the validation process.</td>"
            "<td>Check the job.</td></tr></table>"
        )
        result = await lookup_error_code("1002")
        assert result["status"] == "success"
        assert result["code"] == "1002"
        assert "1002" in result["details"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_code_not_found(self, mock_fetch):
        """Test lookup for a code that does not exist."""
        mock_fetch.return_value = "<table><tr><td>1002</td><td>Some error</td></tr></table>"
        result = await lookup_error_code("9999")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestConvertHtmlToMarkdown:
    """Test convert_html_to_markdown function."""

    def test_basic_html(self):
        """Test conversion of basic HTML."""
        html = "<h1>Title</h1><p>Paragraph text.</p>"
        result = convert_html_to_markdown(html)
        assert "Title" in result
        assert "Paragraph text." in result

    def test_links_preserved(self):
        """Test that links are preserved."""
        html = '<a href="https://example.com">Click here</a>'
        result = convert_html_to_markdown(html)
        assert "https://example.com" in result

    def test_empty_html(self):
        """Test conversion of empty HTML."""
        result = convert_html_to_markdown("")
        assert result.strip() == ""


class TestListHelpers:
    """Test list helper functions."""

    def test_get_list_of_modules(self):
        """Test get_list_of_modules returns correct structure with url_path."""
        result = get_list_of_modules()
        assert result["status"] == "success"
        assert "modules" in result
        assert isinstance(result["modules"], list)
        assert len(result["modules"]) > 0
        # Check structure includes name, description, url_path
        first = result["modules"][0]
        assert "name" in first
        assert "description" in first
        assert "url_path" in first
        assert first["url_path"].startswith("api/qiskit/")

    def test_get_list_of_addons(self):
        """Test get_list_of_addons returns correct structure with url_path."""
        result = get_list_of_addons()
        assert result["status"] == "success"
        assert "addons" in result
        assert len(result["addons"]) > 0
        first = result["addons"][0]
        assert "name" in first
        assert "description" in first
        assert "url_path" in first
        assert "qiskit-addon-" in first["url_path"]

    def test_get_list_of_guides(self):
        """Test get_list_of_guides returns correct structure with url_path."""
        result = get_list_of_guides()
        assert result["status"] == "success"
        assert "guides" in result
        assert len(result["guides"]) > 0
        first = result["guides"][0]
        assert "name" in first
        assert "description" in first
        assert "url_path" in first
        assert first["url_path"].startswith("guides/")

    def test_get_list_of_error_code_categories(self):
        """Test get_list_of_error_code_categories returns correct structure."""
        result = get_list_of_error_code_categories()
        assert result["status"] == "success"
        assert "categories" in result
        assert isinstance(result["categories"], dict)
        assert "registry_url" in result


class TestDocFetcherConstants:
    """Test data_fetcher constants."""

    def test_qiskit_modules_not_empty(self):
        """Test that AVAILABLE_MODULES is not empty."""
        assert len(AVAILABLE_MODULES) > 0

    def test_qiskit_modules_has_circuit(self):
        """Test that AVAILABLE_MODULES contains circuit."""
        assert "circuit" in AVAILABLE_MODULES

    def test_qiskit_modules_are_dict_with_descriptions(self):
        """Test that AVAILABLE_MODULES values are description strings."""
        assert isinstance(AVAILABLE_MODULES, dict)
        for key, value in AVAILABLE_MODULES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_qiskit_addon_modules_not_empty(self):
        """Test that AVAILABLE_ADDONS is not empty."""
        assert len(AVAILABLE_ADDONS) > 0

    def test_qiskit_addons_are_dict_with_descriptions(self):
        """Test that AVAILABLE_ADDONS values are description strings."""
        assert isinstance(AVAILABLE_ADDONS, dict)
        for key, value in AVAILABLE_ADDONS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0

    def test_qiskit_guides_not_empty(self):
        """Test that AVAILABLE_GUIDES is not empty."""
        assert len(AVAILABLE_GUIDES) > 0

    def test_qiskit_guides_are_dict_with_descriptions(self):
        """Test that AVAILABLE_GUIDES values are description strings."""
        assert isinstance(AVAILABLE_GUIDES, dict)
        for key, value in AVAILABLE_GUIDES.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_get_env_float_valid(self):
        """Test _get_env_float with valid value."""
        import os

        original = os.environ.get("TEST_ENV_FLOAT")
        try:
            os.environ["TEST_ENV_FLOAT"] = "5.5"
            result = _get_env_float("TEST_ENV_FLOAT", 10.0)
            assert result == 5.5
        finally:
            if original is not None:
                os.environ["TEST_ENV_FLOAT"] = original
            else:
                os.environ.pop("TEST_ENV_FLOAT", None)

    def test_get_env_float_invalid(self):
        """Test _get_env_float with invalid value returns default."""
        import os

        original = os.environ.get("TEST_ENV_INVALID")
        try:
            os.environ["TEST_ENV_INVALID"] = "not_a_float"
            result = _get_env_float("TEST_ENV_INVALID", 10.0)
            assert result == 10.0
        finally:
            if original is not None:
                os.environ["TEST_ENV_INVALID"] = original
            else:
                os.environ.pop("TEST_ENV_INVALID", None)

    def test_get_env_float_missing(self):
        """Test _get_env_float with missing env var returns default."""
        result = _get_env_float("NONEXISTENT_VAR_12345", 15.0)
        assert result == 15.0

    def test_http_timeout_default(self):
        """Test that HTTP_TIMEOUT has a reasonable default."""
        assert HTTP_TIMEOUT > 0
        assert HTTP_TIMEOUT <= 30.0
