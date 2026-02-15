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

"""Tests for the qiskit-docs-mcp-server."""

from unittest.mock import patch

import pytest
from qiskit_docs_mcp_server.data_fetcher import QISKIT_MODULES
from qiskit_docs_mcp_server.server import (
    get_component_list,
    get_guide,
    get_sdk_module_docs,
    get_style_list,
    search_docs,
)


@pytest.mark.asyncio
class TestGetGuide:
    """Test get_guide function."""

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_valid_guide(self, mock_get_docs):
        """Test getting a valid guide."""
        mock_get_docs.return_value = "Sample optimization guide"
        result = await get_guide("optimization")

        assert result["guide"] == "optimization"
        assert result["documentation"] == "Sample optimization guide"
        mock_get_docs.assert_called_once_with("optimization")

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_invalid_guide(self, mock_get_docs):
        """Test getting an invalid guide."""
        mock_get_docs.return_value = None
        result = await get_guide("nonexistent-guide")

        assert "status" in result
        assert result["status"] == "error"
        assert "nonexistent-guide" in result["message"]

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_error_mitigation(self, mock_get_docs):
        """Test getting error-mitigation guide."""
        mock_get_docs.return_value = "Error mitigation guide content"
        result = await get_guide("error-mitigation")

        assert result["guide"] == "error-mitigation"

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_returns_dict(self, mock_get_docs):
        """Test that get_guide returns a dictionary."""
        mock_get_docs.return_value = "Some guide"
        result = await get_guide("optimization")

        assert isinstance(result, dict)

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_dynamic_circuits(self, mock_get_docs):
        """Test getting dynamic-circuits guide."""
        mock_get_docs.return_value = "Dynamic circuits guide"
        result = await get_guide("dynamic-circuits")

        assert result["guide"] == "dynamic-circuits"


@pytest.mark.asyncio
class TestSearchDocs:
    """Test search_docs function."""

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_with_results(self, mock_search):
        """Test search_docs with results."""
        mock_search.return_value = [
            {
                "type": "module",
                "name": "circuit",
                "url": "https://docs.quantum.ibm.com/api/qiskit/circuit",
            },
            {
                "type": "guide",
                "name": "optimization",
                "url": "https://docs.quantum.ibm.com/guides/optimization",
            },
        ]
        result = await search_docs("circuit")

        assert len(result["results"]) == 2
        mock_search.assert_called_once_with("circuit", "documentation")

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_no_results(self, mock_search):
        """Test search_docs with no results."""
        mock_search.return_value = []
        result = await search_docs("nonexistent-query")

        assert len(result["results"]) == 0

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_returns_dict(self, mock_search):
        """Test that search_docs returns a dict."""
        mock_search.return_value = {"results": ["test"]}
        result = await search_docs("test")

        assert isinstance(result, dict)

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_optimization_query(self, mock_search):
        """Test search_docs with optimization query."""
        mock_search.return_value = [
            {"type": "guide", "name": "optimization"},
            {"type": "addon", "name": "addon-opt-mapper"},
        ]
        result = await search_docs("optimization")

        assert len(result["results"]) == 2

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_empty_query_result(self, mock_search):
        """Test search_docs when search returns empty list."""
        mock_search.return_value = []
        result = await search_docs("xyz")

        assert len(result["results"]) == 0


@pytest.mark.asyncio
class TestResourceFunctions:
    """Test resource functions."""

    async def test_get_component_list_returns_list(self):
        """Test that get_component_list returns a list."""
        result = await get_component_list()
        assert isinstance(result["modules"], list)

    async def test_get_component_list_matches_modules(self):
        """Test that get_component_list matches QISKIT_MODULES."""
        result = await get_component_list()
        expected = list(QISKIT_MODULES.keys())
        assert result["modules"] == expected

    async def test_get_style_list_returns_list(self):
        """Test that get_style_list returns a list."""
        result = await get_style_list()
        assert isinstance(result["guides"], list)

    async def test_get_style_list_contains_all_guides(self):
        """Test that get_style_list contains all expected guides."""
        result = await get_style_list()
        expected_guides = [
            "optimization",
            "quantum-circuits",
            "error-mitigation",
            "dynamic-circuits",
            "parametric-compilation",
            "performance-tuning",
        ]
        assert result["guides"] == expected_guides


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("qiskit_docs_mcp_server.server.get_component_docs")
    async def test_get_sdk_module_docs_case_sensitive(self, mock_get_docs):
        """Test that module names are case-sensitive."""
        mock_get_docs.return_value = None
        result = await get_sdk_module_docs("Circuit")  # Capital C
        assert "status" in result
        assert result["status"] == "error"

    @patch("qiskit_docs_mcp_server.server.get_guide_docs")
    async def test_get_guide_case_sensitive(self, mock_get_docs):
        """Test that guide names are case-sensitive."""
        mock_get_docs.return_value = None
        result = await get_guide("Optimization")  # Capital O
        assert "status" in result
        assert result["status"] == "error"

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_empty_query(self, mock_search):
        """Test search_docs with empty query string."""
        mock_search.return_value = []
        result = await search_docs("")
        assert isinstance(result["results"], list)

    @patch("qiskit_docs_mcp_server.server.search_qiskit_docs")
    async def test_search_docs_special_characters(self, mock_search):
        """Test search_docs with special characters."""
        mock_search.return_value = []
        result = await search_docs("circuit&transpiler")
        assert isinstance(result["results"], list)
        mock_search.assert_called_once_with("circuit&transpiler", "documentation")
