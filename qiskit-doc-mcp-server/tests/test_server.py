# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the qiskit-doc-mcp server."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "qiskit-doc-mcp"))

from server import (
    list_sdk_modules,
    list_addons,
    list_guides,
    get_sdk_module_docs,
    get_addon_docs,
    get_guide,
    search_docs,
    get_component_list,
    get_pattern_list,
    get_style_list,
)
from data_fetcher import (
    QISKIT_MODULES,
    QISKIT_ADDON_MODULES,
)


class TestListSdkModules:
    """Test list_sdk_modules function."""

    def test_list_sdk_modules_returns_list(self):
        """Test that list_sdk_modules returns a list."""
        result = list_sdk_modules()
        assert isinstance(result, list)

    def test_list_sdk_modules_not_empty(self):
        """Test that list_sdk_modules returns non-empty list."""
        result = list_sdk_modules()
        assert len(result) > 0

    def test_list_sdk_modules_contains_expected_modules(self):
        """Test that list_sdk_modules contains expected modules."""
        result = list_sdk_modules()
        expected_modules = list(QISKIT_MODULES.keys())
        assert result == expected_modules

    def test_list_sdk_modules_contains_circuit(self):
        """Test that list_sdk_modules includes 'circuit' module."""
        result = list_sdk_modules()
        assert "circuit" in result

    def test_list_sdk_modules_contains_primitives(self):
        """Test that list_sdk_modules includes 'primitives' module."""
        result = list_sdk_modules()
        assert "primitives" in result

    def test_list_sdk_modules_contains_transpiler(self):
        """Test that list_sdk_modules includes 'transpiler' module."""
        result = list_sdk_modules()
        assert "transpiler" in result


class TestListAddons:
    """Test list_addons function."""

    def test_list_addons_returns_list(self):
        """Test that list_addons returns a list."""
        result = list_addons()
        assert isinstance(result, list)

    def test_list_addons_not_empty(self):
        """Test that list_addons returns non-empty list."""
        result = list_addons()
        assert len(result) > 0

    def test_list_addons_contains_expected_addons(self):
        """Test that list_addons contains expected addons."""
        result = list_addons()
        expected_addons = list(QISKIT_ADDON_MODULES.keys())
        assert result == expected_addons

    def test_list_addons_contains_opt_mapper(self):
        """Test that list_addons includes 'addon-opt-mapper'."""
        result = list_addons()
        assert "addon-opt-mapper" in result

    def test_list_addons_contains_vqe(self):
        """Test that list_addons includes 'addon-vqe'."""
        result = list_addons()
        assert "addon-vqe" in result


class TestListGuides:
    """Test list_guides function."""

    def test_list_guides_returns_list(self):
        """Test that list_guides returns a list."""
        result = list_guides()
        assert isinstance(result, list)

    def test_list_guides_not_empty(self):
        """Test that list_guides returns non-empty list."""
        result = list_guides()
        assert len(result) > 0

    def test_list_guides_contains_expected_guides(self):
        """Test that list_guides contains expected guides."""
        result = list_guides()
        expected_guides = [
            "optimization",
            "quantum-circuits",
            "error-mitigation",
            "dynamic-circuits",
            "parametric-compilation",
            "performance-tuning",
        ]
        assert result == expected_guides

    def test_list_guides_contains_optimization(self):
        """Test that list_guides includes 'optimization'."""
        result = list_guides()
        assert "optimization" in result

    def test_list_guides_contains_error_mitigation(self):
        """Test that list_guides includes 'error-mitigation'."""
        result = list_guides()
        assert "error-mitigation" in result


class TestGetSdkModuleDocs:
    """Test get_sdk_module_docs function."""

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_valid_module(self, mock_get_docs):
        """Test getting docs for a valid module."""
        mock_get_docs.return_value = "Sample circuit documentation"
        result = get_sdk_module_docs("circuit")

        assert result["module"] == "circuit"
        assert result["documentation"] == "Sample circuit documentation"
        mock_get_docs.assert_called_once_with("circuit")

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_invalid_module(self, mock_get_docs):
        """Test getting docs for an invalid module."""
        mock_get_docs.return_value = None
        result = get_sdk_module_docs("invalid_module")

        assert "error" in result
        assert "invalid_module" in result["error"]
        assert "not found" in result["error"]

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_primitives(self, mock_get_docs):
        """Test getting docs for primitives module."""
        mock_get_docs.return_value = "Sample primitives documentation"
        result = get_sdk_module_docs("primitives")

        assert result["module"] == "primitives"
        assert "primitives" in result["documentation"].lower()

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_returns_dict(self, mock_get_docs):
        """Test that get_sdk_module_docs returns a dictionary."""
        mock_get_docs.return_value = "Some docs"
        result = get_sdk_module_docs("circuit")

        assert isinstance(result, dict)

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_transpiler(self, mock_get_docs):
        """Test getting docs for transpiler module."""
        mock_get_docs.return_value = "Transpiler documentation"
        result = get_sdk_module_docs("transpiler")

        assert result["module"] == "transpiler"
        assert "Transpiler" in result["documentation"]


class TestGetAddonDocs:
    """Test get_addon_docs function."""

    @patch("data_fetcher.get_pattern_docs")
    def test_get_addon_docs_valid_addon(self, mock_get_docs):
        """Test getting docs for a valid addon."""
        mock_get_docs.return_value = "Sample addon documentation"
        result = get_addon_docs("addon-vqe")

        assert result["addon"] == "addon-vqe"
        assert result["documentation"] == "Sample addon documentation"
        mock_get_docs.assert_called_once_with("addon-vqe")

    @patch("data_fetcher.get_pattern_docs")
    def test_get_addon_docs_invalid_addon(self, mock_get_docs):
        """Test getting docs for an invalid addon."""
        mock_get_docs.return_value = None
        result = get_addon_docs("invalid-addon")

        assert "error" in result
        assert "invalid-addon" in result["error"]
        assert "not found" in result["error"]

    @patch("data_fetcher.get_pattern_docs")
    def test_get_addon_docs_opt_mapper(self, mock_get_docs):
        """Test getting docs for opt-mapper addon."""
        mock_get_docs.return_value = "Opt mapper documentation"
        result = get_addon_docs("addon-opt-mapper")

        assert result["addon"] == "addon-opt-mapper"

    @patch("data_fetcher.get_pattern_docs")
    def test_get_addon_docs_returns_dict(self, mock_get_docs):
        """Test that get_addon_docs returns a dictionary."""
        mock_get_docs.return_value = "Some docs"
        result = get_addon_docs("addon-vqe")

        assert isinstance(result, dict)


class TestGetGuide:
    """Test get_guide function."""

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_valid_guide(self, mock_get_docs):
        """Test getting a valid guide."""
        mock_get_docs.return_value = "Sample optimization guide"
        result = get_guide("optimization")

        assert result["guide"] == "optimization"
        assert result["documentation"] == "Sample optimization guide"
        mock_get_docs.assert_called_once_with("optimization")

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_invalid_guide(self, mock_get_docs):
        """Test getting an invalid guide."""
        mock_get_docs.return_value = None
        result = get_guide("nonexistent-guide")

        assert "error" in result
        assert "nonexistent-guide" in result["error"]
        assert "not found" in result["error"]

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_error_mitigation(self, mock_get_docs):
        """Test getting error-mitigation guide."""
        mock_get_docs.return_value = "Error mitigation guide content"
        result = get_guide("error-mitigation")

        assert result["guide"] == "error-mitigation"

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_returns_dict(self, mock_get_docs):
        """Test that get_guide returns a dictionary."""
        mock_get_docs.return_value = "Some guide"
        result = get_guide("optimization")

        assert isinstance(result, dict)

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_dynamic_circuits(self, mock_get_docs):
        """Test getting dynamic-circuits guide."""
        mock_get_docs.return_value = "Dynamic circuits guide"
        result = get_guide("dynamic-circuits")

        assert result["guide"] == "dynamic-circuits"


class TestSearchDocs:
    """Test search_docs function."""

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_with_results(self, mock_search):
        """Test search_docs with results."""
        mock_search.return_value = [
            {"type": "module", "name": "circuit", "url": "https://docs.quantum.ibm.com/api/qiskit/circuit"},
            {"type": "guide", "name": "optimization", "url": "https://docs.quantum.ibm.com/guides/optimization"},
        ]
        result = search_docs("circuit")

        assert isinstance(result, list)
        assert len(result) == 2
        mock_search.assert_called_once_with("circuit")

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_no_results(self, mock_search):
        """Test search_docs with no results."""
        mock_search.return_value = []
        result = search_docs("nonexistent-query")

        assert isinstance(result, list)
        assert len(result) == 1
        assert "info" in result[0]
        assert "No results found" in result[0]["info"]

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_returns_list(self, mock_search):
        """Test that search_docs returns a list."""
        mock_search.return_value = [{"name": "test"}]
        result = search_docs("test")

        assert isinstance(result, list)

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_optimization_query(self, mock_search):
        """Test search_docs with optimization query."""
        mock_search.return_value = [
            {"type": "guide", "name": "optimization"},
            {"type": "addon", "name": "addon-opt-mapper"},
        ]
        result = search_docs("optimization")

        assert len(result) == 2

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_empty_query_result(self, mock_search):
        """Test search_docs when search returns empty list."""
        mock_search.return_value = []
        result = search_docs("xyz")

        assert isinstance(result, list)
        assert result[0]["info"] == "No results found for 'xyz'"


class TestResourceFunctions:
    """Test resource functions."""

    def test_get_component_list_returns_list(self):
        """Test that get_component_list returns a list."""
        result = get_component_list()
        assert isinstance(result, list)

    def test_get_component_list_matches_modules(self):
        """Test that get_component_list matches QISKIT_MODULES."""
        result = get_component_list()
        expected = list(QISKIT_MODULES.keys())
        assert result == expected

    def test_get_pattern_list_returns_list(self):
        """Test that get_pattern_list returns a list."""
        result = get_pattern_list()
        assert isinstance(result, list)

    def test_get_pattern_list_matches_addons(self):
        """Test that get_pattern_list matches QISKIT_ADDON_MODULES."""
        result = get_pattern_list()
        expected = list(QISKIT_ADDON_MODULES.keys())
        assert result == expected

    def test_get_style_list_returns_list(self):
        """Test that get_style_list returns a list."""
        result = get_style_list()
        assert isinstance(result, list)

    def test_get_style_list_contains_all_guides(self):
        """Test that get_style_list contains all expected guides."""
        result = get_style_list()
        expected_guides = [
            "optimization",
            "quantum-circuits",
            "error-mitigation",
            "dynamic-circuits",
            "parametric-compilation",
            "performance-tuning",
        ]
        assert result == expected_guides


class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("data_fetcher.get_component_docs")
    def test_get_sdk_module_docs_case_sensitive(self, mock_get_docs):
        """Test that module names are case-sensitive."""
        mock_get_docs.return_value = None
        result = get_sdk_module_docs("Circuit")  # Capital C
        assert "error" in result

    @patch("data_fetcher.get_pattern_docs")
    def test_get_addon_docs_case_sensitive(self, mock_get_docs):
        """Test that addon names are case-sensitive."""
        mock_get_docs.return_value = None
        result = get_addon_docs("Addon-VQE")  # Capital letters
        assert "error" in result

    @patch("data_fetcher.get_style_docs")
    def test_get_guide_case_sensitive(self, mock_get_docs):
        """Test that guide names are case-sensitive."""
        mock_get_docs.return_value = None
        result = get_guide("Optimization")  # Capital O
        assert "error" in result

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_empty_query(self, mock_search):
        """Test search_docs with empty query string."""
        mock_search.return_value = []
        result = search_docs("")
        assert isinstance(result, list)

    @patch("data_fetcher.search_qiskit_docs")
    def test_search_docs_special_characters(self, mock_search):
        """Test search_docs with special characters."""
        mock_search.return_value = []
        result = search_docs("circuit&transpiler")
        assert isinstance(result, list)
        mock_search.assert_called_once_with("circuit&transpiler")
