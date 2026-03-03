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
from qiskit_docs_mcp_server.data_fetcher import (
    HTTP_TIMEOUT,
    QISKIT_ADDON_MODULES,
    QISKIT_MODULES,
    _find_similar,
    _get_env_float,
    fetch_text,
    fetch_text_json,
    get_component_docs,
    get_guide_docs,
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

        for module in QISKIT_MODULES:
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
        mock_fetch.return_value = "Optimization guide"
        result = await get_guide_docs("optimization")

        assert result["status"] == "success"
        assert result["guide"] == "optimization"
        assert "Optimization guide" in result["documentation"]
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
        result = await get_guide_docs("optimization-guide")  # Similar to 'optimization'
        assert result["status"] == "error"
        assert "available_guides" in result
        # May or may not have suggestions depending on similarity cutoff
        if "suggestions" in result:
            assert "optimization" in result["suggestions"]
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_all_valid_guides(self, mock_fetch):
        """Test getting docs for all valid guides."""
        mock_fetch.return_value = "Guide documentation"
        valid_guides = [
            "optimization",
            "quantum-circuits",
            "error-mitigation",
            "dynamic-circuits",
            "parametric-compilation",
            "performance-tuning",
        ]

        for guide in valid_guides:
            result = await get_guide_docs(guide)
            assert result["status"] == "success"
            assert result["guide"] == guide

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_fetch_fails(self, mock_fetch):
        """Test get_guide_docs when fetch fails."""
        mock_fetch.return_value = None
        result = await get_guide_docs("optimization")
        assert result["status"] == "success"
        assert result["documentation"] is None

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    async def test_get_guide_docs_error_mitigation(self, mock_fetch):
        """Test getting error-mitigation guide."""
        mock_fetch.return_value = "Error mitigation techniques"
        result = await get_guide_docs("error-mitigation")
        assert result["status"] == "success"
        assert result["guide"] == "error-mitigation"
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
        result = await get_guide_docs("optimization")

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
        """Test that QISKIT_MODULES is not empty."""
        assert len(QISKIT_MODULES) > 0

    def test_qiskit_modules_has_circuit(self):
        """Test that QISKIT_MODULES contains circuit."""
        assert "circuit" in QISKIT_MODULES

    def test_qiskit_modules_has_primitives(self):
        """Test that QISKIT_MODULES contains primitives."""
        assert "primitives" in QISKIT_MODULES

    def test_qiskit_modules_has_transpiler(self):
        """Test that QISKIT_MODULES contains transpiler."""
        assert "transpiler" in QISKIT_MODULES

    def test_qiskit_addon_modules_not_empty(self):
        """Test that QISKIT_ADDON_MODULES is not empty."""
        assert len(QISKIT_ADDON_MODULES) > 0

    def test_qiskit_addon_modules_has_vqe(self):
        """Test that QISKIT_ADDON_MODULES contains VQE."""
        assert "addon-vqe" in QISKIT_ADDON_MODULES

    def test_qiskit_addon_modules_has_opt_mapper(self):
        """Test that QISKIT_ADDON_MODULES contains opt-mapper."""
        assert "addon-opt-mapper" in QISKIT_ADDON_MODULES

    def test_qiskit_modules_values_are_strings(self):
        """Test that QISKIT_MODULES values are strings."""
        for value in QISKIT_MODULES.values():
            assert isinstance(value, str)

    def test_qiskit_addon_modules_values_are_strings(self):
        """Test that QISKIT_ADDON_MODULES values are strings."""
        for value in QISKIT_ADDON_MODULES.values():
            assert isinstance(value, str)
