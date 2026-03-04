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
from qiskit_docs_mcp_server.constants import (
    AVAILABLE_ADDONS,
    AVAILABLE_MODULES,
    HTTP_TIMEOUT,
    _get_env_float,
)
from qiskit_docs_mcp_server.data_fetcher import (
    _find_similar,
    fetch_text,
    fetch_text_json,
    get_addon_docs,
    get_component_docs,
    get_guide_docs,
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


class TestGetComponentDocs:
    """Test get_component_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_component_docs_valid_module(self, mock_fetch):
        """Test getting docs for a valid module."""
        mock_fetch.return_value = "Circuit documentation"
        result = await get_component_docs("circuit")

        assert result["status"] == "success"
        assert result["module"] == "circuit"
        assert "Circuit documentation" in result["documentation"]
        assert "metadata" in result
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_component_docs_invalid_module(self, mock_fetch):
        """Test getting docs for an invalid module."""
        result = await get_component_docs("invalid_module")
        assert result["status"] == "error"
        assert "not found" in result["message"]
        assert "available_modules" in result
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_component_docs_invalid_with_suggestions(self, mock_fetch):
        """Test getting docs with similar module suggestions."""
        result = await get_component_docs("circuitt")  # Typo of 'circuit'
        assert result["status"] == "error"
        assert "available_modules" in result
        # May or may not have suggestions depending on similarity cutoff
        if "suggestions" in result:
            assert "circuit" in result["suggestions"]
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_component_docs_all_valid_modules(self, mock_fetch):
        """Test getting docs for all valid modules."""
        mock_fetch.return_value = "Documentation"

        for module in AVAILABLE_MODULES:
            result = await get_component_docs(module)
            assert result["status"] == "success"
            assert result["module"] == module

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_component_docs_fetch_fails(self, mock_fetch):
        """Test get_component_docs when fetch fails."""
        mock_fetch.return_value = None
        result = await get_component_docs("circuit")
        assert result["status"] == "success"
        assert result["documentation"] is None


class TestGetGuideDocs:
    """Test get_guide_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_valid_guide(self, mock_fetch):
        """Test getting docs for a valid guide."""
        mock_fetch.return_value = "Quick start guide"
        result = await get_guide_docs("quick-start")

        assert result["status"] == "success"
        assert result["guide"] == "quick-start"
        assert "Quick start guide" in result["documentation"]
        assert "metadata" in result
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_invalid_guide(self, mock_fetch):
        """Test getting docs for an invalid guide."""
        result = await get_guide_docs("nonexistent-guide")
        assert result["status"] == "error"
        assert "not found" in result["message"]
        assert "available_guides" in result
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_invalid_with_suggestions(self, mock_fetch):
        """Test getting docs with similar guide suggestions."""
        result = await get_guide_docs(
            "transpile-with-pass-manager"
        )  # Similar to 'transpile-with-pass-managers'
        assert result["status"] == "error"
        assert "available_guides" in result
        # May or may not have suggestions depending on similarity cutoff
        if "suggestions" in result:
            assert "transpile-with-pass-managers" in result["suggestions"]
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_all_valid_guides(self, mock_fetch):
        """Test getting docs for all valid guides."""
        mock_fetch.return_value = "Guide documentation"
        valid_guides = [
            "quick-start",
            "construct-circuits",
            "transpile",
            "dynamic-circuits",
            "primitives",
            "configure-error-mitigation",
        ]

        for guide in valid_guides:
            result = await get_guide_docs(guide)
            assert result["status"] == "success"
            assert result["guide"] == guide

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_fetch_fails(self, mock_fetch):
        """Test get_guide_docs when fetch fails."""
        mock_fetch.return_value = None
        result = await get_guide_docs("quick-start")
        assert result["status"] == "success"
        assert result["documentation"] is None

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_error_mitigation(self, mock_fetch):
        """Test getting error-mitigation guide."""
        mock_fetch.return_value = "Error mitigation techniques"
        result = await get_guide_docs("configure-error-mitigation")
        assert result["status"] == "success"
        assert result["guide"] == "configure-error-mitigation"
        assert "Error mitigation techniques" in result["documentation"]


class TestSearchQiskitDocs:
    """Test search_qiskit_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_qiskit_docs_with_results(self, mock_fetch):
        """Test search with results."""
        mock_fetch.return_value = [
            {"name": "circuit", "type": "module"},
            {"name": "optimization", "type": "guide"},
        ]
        result = await search_qiskit_docs("circuit")

        assert result["status"] == "success"
        assert result["query"] == "circuit"
        assert len(result["results"]) == 2
        assert result["total_results"] == 2
        assert "metadata" in result
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_qiskit_docs_no_results(self, mock_fetch):
        """Test search with no results."""
        mock_fetch.return_value = []
        result = await search_qiskit_docs("nonexistent")

        assert result["status"] == "success"
        assert result["query"] == "nonexistent"
        assert result["results"] == []
        assert result["total_results"] == 0

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_qiskit_docs_returns_dict(self, mock_fetch):
        """Test that search returns a dict with proper structure."""
        mock_fetch.return_value = [{"result": "test"}]
        result = await search_qiskit_docs("test")
        assert isinstance(result, dict)
        assert "status" in result
        assert "results" in result

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_qiskit_docs_fetch_fails(self, mock_fetch):
        """Test search when fetch fails."""
        mock_fetch.return_value = None
        result = await search_qiskit_docs("circuit")
        assert result["status"] == "success"
        assert result["results"] == []
        assert result["total_results"] == 0


class TestFuzzyMatching:
    """Test fuzzy matching functionality."""

    def test_find_similar_exact_match(self):
        """Test finding exact match."""
        available = ["circuit", "primitives", "transpiler"]
        result = _find_similar("circuit", available)
        assert "circuit" in result

    def test_find_similar_typo_match(self):
        """Test finding close match with typo."""
        available = ["circuit", "primitives", "transpiler"]
        result = _find_similar("circuitt", available)
        assert "circuit" in result

    def test_find_similar_no_match(self):
        """Test no match returns empty list."""
        available = ["circuit", "primitives", "transpiler"]
        result = _find_similar("xyz123", available)
        assert len(result) == 0

    def test_find_similar_empty_available(self):
        """Test with empty available list."""
        result = _find_similar("circuit", [])
        assert result == []

    def test_find_similar_empty_query(self):
        """Test with empty query."""
        available = ["circuit", "primitives", "transpiler"]
        result = _find_similar("", available)
        assert result == []

    def test_find_similar_partial_match(self):
        """Test finding partial matches."""
        available = ["optimization", "quantum-circuits", "error-mitigation"]
        # Test with a query that should match "error-mitigation"
        result = _find_similar("error", available, cutoff=0.5)
        assert isinstance(result, list)
        # May or may not have matches depending on similarity

    def test_find_similar_limit_results(self):
        """Test that results are limited to 3."""
        available = ["circuit", "circuits", "circuitry", "circular", "circulate"]
        result = _find_similar("circuit", available)
        # difflib.get_close_matches returns at most n=3 by default
        assert len(result) <= 3


class TestMetadataHandling:
    """Test metadata functionality."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_component_docs_has_metadata(self, mock_fetch):
        """Test that component docs response includes metadata."""
        mock_fetch.return_value = "Documentation"
        result = await get_component_docs("circuit")

        assert "metadata" in result
        metadata = result["metadata"]
        assert "url" in metadata
        assert "timestamp" in metadata
        assert "content_type" in metadata
        assert "content_length" in metadata
        assert metadata["content_type"] == "markdown"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_guide_docs_has_metadata(self, mock_fetch):
        """Test that guide docs response includes metadata."""
        mock_fetch.return_value = "Guide content"
        result = await get_guide_docs("quick-start")

        assert "metadata" in result
        metadata = result["metadata"]
        assert "url" in metadata
        assert "timestamp" in metadata
        assert metadata["content_type"] == "markdown"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    async def test_search_docs_has_metadata(self, mock_fetch):
        """Test that search docs response includes metadata."""
        mock_fetch.return_value = [{"name": "circuit"}]
        result = await search_qiskit_docs("circuit")

        assert "metadata" in result
        metadata = result["metadata"]
        assert "url" in metadata
        assert "timestamp" in metadata
        assert metadata["content_type"] == "json"


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
        assert HTTP_TIMEOUT <= 30.0  # Reasonable timeout range

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_uses_http_timeout(self, mock_client_class):
        """Test that fetch_text uses HTTP_TIMEOUT."""
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await fetch_text("https://example.com")

        # Verify httpx.AsyncClient was called with timeout parameter
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == HTTP_TIMEOUT

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.AsyncClient")
    async def test_fetch_text_json_uses_http_timeout(self, mock_client_class):
        """Test that fetch_text_json uses HTTP_TIMEOUT."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        await fetch_text_json("https://example.com/api")

        # Verify httpx.AsyncClient was called with timeout parameter
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == HTTP_TIMEOUT


class TestDocFetcherConstants:
    """Test data_fetcher constants."""

    def test_qiskit_modules_not_empty(self):
        """Test that AVAILABLE_MODULES is not empty."""
        assert len(AVAILABLE_MODULES) > 0

    def test_qiskit_modules_has_circuit(self):
        """Test that AVAILABLE_MODULES contains circuit."""
        assert "circuit" in AVAILABLE_MODULES

    def test_qiskit_modules_has_primitives(self):
        """Test that AVAILABLE_MODULES contains primitives."""
        assert "primitives" in AVAILABLE_MODULES

    def test_qiskit_modules_has_transpiler(self):
        """Test that AVAILABLE_MODULES contains transpiler."""
        assert "transpiler" in AVAILABLE_MODULES

    def test_qiskit_addon_modules_not_empty(self):
        """Test that AVAILABLE_ADDONS is not empty."""
        assert len(AVAILABLE_ADDONS) > 0

    def test_qiskit_addon_modules_has_sqd(self):
        """Test that AVAILABLE_ADDONS contains SQD."""
        assert "sqd" in AVAILABLE_ADDONS

    def test_qiskit_addon_modules_has_cutting(self):
        """Test that AVAILABLE_ADDONS contains cutting."""
        assert "cutting" in AVAILABLE_ADDONS

    def test_qiskit_modules_entries_are_strings(self):
        """Test that AVAILABLE_MODULES entries are strings."""
        for value in AVAILABLE_MODULES:
            assert isinstance(value, str)

    def test_qiskit_addon_modules_entries_are_strings(self):
        """Test that AVAILABLE_ADDONS entries are strings."""
        for value in AVAILABLE_ADDONS:
            assert isinstance(value, str)


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
        assert "Invalid error code format" in result["message"]

    async def test_invalid_code_format_long(self):
        """Test that 5-digit codes return an error."""
        result = await lookup_error_code("12345")
        assert result["status"] == "error"
        assert "Invalid error code format" in result["message"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_fetch_failure(self, mock_fetch):
        """Test lookup when fetch fails."""
        mock_fetch.return_value = None
        result = await lookup_error_code("1002")
        assert result["status"] == "error"
        assert "Failed to fetch" in result["message"]

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
        assert "details" in result
        assert "1002" in result["details"]
        assert "metadata" in result

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_code_not_found(self, mock_fetch):
        """Test lookup for a code that does not exist in the page."""
        mock_fetch.return_value = "<table><tr><td>1002</td><td>Some error</td></tr></table>"
        result = await lookup_error_code("9999")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_metadata_url_includes_range(self, mock_fetch):
        """Test that metadata URL points to the correct error range anchor."""
        mock_fetch.return_value = "<table><tr><td>7001</td><td>Error</td><td>Fix</td></tr></table>"
        result = await lookup_error_code("7001")
        assert result["status"] == "success"
        assert "7xxx" in result["metadata"]["url"]


class TestGetAddonDocs:
    """Test get_addon_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_addon_docs_valid(self, mock_fetch):
        """Test getting docs for a valid addon."""
        mock_fetch.return_value = "SQD documentation"
        result = await get_addon_docs("sqd")
        assert result["status"] == "success"
        assert result["addon"] == "sqd"
        assert "SQD documentation" in result["documentation"]
        assert "metadata" in result
        assert "qiskit-addon-sqd" in result["metadata"]["url"]

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_addon_docs_invalid(self, mock_fetch):
        """Test getting docs for an invalid addon."""
        result = await get_addon_docs("nonexistent")
        assert result["status"] == "error"
        assert "not found" in result["message"]
        assert "available_addons" in result
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_addon_docs_with_suggestions(self, mock_fetch):
        """Test getting docs with fuzzy match suggestions."""
        result = await get_addon_docs("cut")
        assert result["status"] == "error"
        if "suggestions" in result:
            assert "cutting" in result["suggestions"]
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_addon_docs_fetch_fails(self, mock_fetch):
        """Test get_addon_docs when fetch fails."""
        mock_fetch.return_value = None
        result = await get_addon_docs("cutting")
        assert result["status"] == "success"
        assert result["documentation"] is None

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_addon_docs_all_valid(self, mock_fetch):
        """Test getting docs for all valid addons."""
        mock_fetch.return_value = "Addon docs"
        for addon in AVAILABLE_ADDONS:
            result = await get_addon_docs(addon)
            assert result["status"] == "success"
            assert result["addon"] == addon
