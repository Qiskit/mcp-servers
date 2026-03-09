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

"""Pytest configuration and shared fixtures."""

from unittest.mock import patch

import pytest


@pytest.fixture
def mock_fetch_text():
    """Mock fetch_text function."""
    with patch("qiskit_docs_mcp_server.data_fetcher.fetch_text") as mock:
        mock.return_value = "Mock documentation content"
        yield mock


@pytest.fixture
def mock_fetch_text_json():
    """Mock fetch_text_json function."""
    with patch("qiskit_docs_mcp_server.data_fetcher.fetch_text_json") as mock:
        mock.return_value = [
            {"type": "module", "name": "circuit", "url": "https://example.com/circuit"}
        ]
        yield mock


@pytest.fixture
def mock_get_component_docs():
    """Mock get_component_docs function."""
    with patch("qiskit_docs_mcp_server.data_fetcher.get_component_docs") as mock:
        mock.return_value = "Component documentation"
        yield mock


@pytest.fixture
def mock_search_qiskit_docs():
    """Mock search_qiskit_docs function."""
    with patch("qiskit_docs_mcp_server.data_fetcher.search_qiskit_docs") as mock:
        mock.return_value = [
            {
                "type": "module",
                "name": "circuit",
                "url": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit",
            }
        ]
        yield mock


@pytest.fixture
def sample_module_docs():
    """Sample module documentation."""
    return {
        "name": "circuit",
        "description": "Quantum circuit module",
        "url": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit",
        "content": "Detailed circuit documentation...",
    }


@pytest.fixture
def sample_addon_docs():
    """Sample addon documentation."""
    return {
        "name": "sqd",
        "description": "Sample-based Quantum Diagonalization addon",
        "url": "https://qiskit.github.io/qiskit-addon-sqd",
        "content": "SQD implementation details...",
    }


@pytest.fixture
def sample_guide_docs():
    """Sample guide documentation."""
    return {
        "name": "quick-start",
        "description": "Qiskit quick start guide",
        "url": "https://quantum.cloud.ibm.com/docs/en/guides/quick-start",
        "content": "Getting started with Qiskit...",
    }


@pytest.fixture
def sample_search_results():
    """Sample search results."""
    return [
        {
            "type": "sdk_module",
            "name": "circuit",
            "url": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit",
        },
        {
            "type": "addon",
            "name": "sqd",
            "url": "https://qiskit.github.io/qiskit-addon-sqd",
        },
        {
            "type": "guide",
            "name": "quick-start",
            "url": "https://docs.quantum.ibm.com/guides/quick-start",
        },
    ]
