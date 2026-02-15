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

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qiskit_docs_mcp_server.data_fetcher import (
    QISKIT_ADDON_MODULES,
    QISKIT_MODULES,
    fetch_text,
    fetch_text_json,
    get_component_docs,
    get_guide_docs,
    search_qiskit_docs,
)


class TestFetchText:
    """Test fetch_text function."""

    # Clearing the cache before each test to ensure that the tests are independent
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the lru_cache before each test."""
        fetch_text.cache_clear()

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_success(self, mock_client_class):
        """Test successful text fetch."""
        mock_response = MagicMock()
        mock_response.text = "Sample documentation"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text("https://example.com")
        assert result == "Sample documentation"

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_http_error(self, mock_client_class):
        """Test fetch_text with HTTP error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text("https://example.com")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_generic_exception(self, mock_client_class):
        """Test fetch_text with generic exception."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Unexpected error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text("https://example.com")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_timeout(self, mock_client_class):
        """Test fetch_text with timeout."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text("https://example.com")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_caching(self, mock_client_class):
        """Test that fetch_text uses caching."""
        mock_response = MagicMock()
        mock_response.text = "Cached content"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        # First call
        result1 = fetch_text("https://example.com/cached")
        # Second call should use cache
        result2 = fetch_text("https://example.com/cached")

        assert result1 == result2
        # Client should only be called once due to caching
        assert mock_client.get.call_count == 1


class TestFetchTextJson:
    """Test fetch_text_json function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_json_success(self, mock_client_class):
        """Test successful JSON fetch."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"key": "value"}]
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text_json("https://example.com/api")
        assert result == [{"key": "value"}]

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_json_http_error(self, mock_client_class):
        """Test fetch_text_json with HTTP error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text_json("https://example.com/api")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_json_generic_exception(self, mock_client_class):
        """Test fetch_text_json with generic exception."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Unexpected error")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text_json("https://example.com/api")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.httpx.Client")
    def test_fetch_text_json_returns_list(self, mock_client_class):
        """Test that fetch_text_json returns list."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"name": "test"}]
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = fetch_text_json("https://example.com")
        assert isinstance(result, list)


class TestGetComponentDocs:
    """Test get_component_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_component_docs_valid_module(self, mock_fetch):
        """Test getting docs for a valid module."""
        mock_fetch.return_value = "Circuit documentation"
        result = get_component_docs("circuit")

        assert result == "Circuit documentation"
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_component_docs_invalid_module(self, mock_fetch):
        """Test getting docs for an invalid module."""
        result = get_component_docs("invalid_module")
        assert result is None
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_component_docs_all_valid_modules(self, mock_fetch):
        """Test getting docs for all valid modules."""
        mock_fetch.return_value = "Documentation"

        for module in QISKIT_MODULES:
            result = get_component_docs(module)
            assert result == "Documentation"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_component_docs_fetch_fails(self, mock_fetch):
        """Test get_component_docs when fetch fails."""
        mock_fetch.return_value = None
        result = get_component_docs("circuit")
        assert result is None


class TestGetGuideDocs:
    """Test get_guide_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_guide_docs_valid_guide(self, mock_fetch):
        """Test getting docs for a valid guide."""
        mock_fetch.return_value = "Optimization guide"
        result = get_guide_docs("optimization")

        assert result == "Optimization guide"
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_guide_docs_invalid_guide(self, mock_fetch):
        """Test getting docs for an invalid guide."""
        result = get_guide_docs("nonexistent-guide")
        assert result is None
        mock_fetch.assert_not_called()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_guide_docs_all_valid_guides(self, mock_fetch):
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
            result = get_guide_docs(guide)
            assert result == "Guide documentation"

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_guide_docs_fetch_fails(self, mock_fetch):
        """Test get_guide_docs when fetch fails."""
        mock_fetch.return_value = None
        result = get_guide_docs("optimization")
        assert result is None

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text")
    def test_get_guide_docs_error_mitigation(self, mock_fetch):
        """Test getting error-mitigation guide."""
        mock_fetch.return_value = "Error mitigation techniques"
        result = get_guide_docs("error-mitigation")
        assert result == "Error mitigation techniques"


class TestSearchQiskitDocs:
    """Test search_qiskit_docs function."""

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    def test_search_qiskit_docs_with_results(self, mock_fetch):
        """Test search with results."""
        mock_fetch.return_value = [
            {"name": "circuit", "type": "module"},
            {"name": "optimization", "type": "guide"},
        ]
        result = search_qiskit_docs("circuit")

        assert isinstance(result, list)
        assert len(result) == 2
        mock_fetch.assert_called_once()

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    def test_search_qiskit_docs_no_results(self, mock_fetch):
        """Test search with no results."""
        mock_fetch.return_value = []
        result = search_qiskit_docs("nonexistent")

        assert result == []

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    def test_search_qiskit_docs_returns_list(self, mock_fetch):
        """Test that search returns a list."""
        mock_fetch.return_value = [{"result": "test"}]
        result = search_qiskit_docs("test")
        assert isinstance(result, list)

    @patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json")
    def test_search_qiskit_docs_fetch_fails(self, mock_fetch):
        """Test search when fetch fails."""
        mock_fetch.return_value = None
        result = search_qiskit_docs("circuit")
        assert result is None


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
