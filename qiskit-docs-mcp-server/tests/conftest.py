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

"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "qiskit-doc-mcp"))


@pytest.fixture
def mock_fetch_text():
    """Mock fetch_text function."""
    with patch("data_fetcher.fetch_text") as mock:
        mock.return_value = "Mock documentation content"
        yield mock


@pytest.fixture
def mock_fetch_text_json():
    """Mock fetch_text_json function."""
    with patch("data_fetcher.fetch_text_json") as mock:
        mock.return_value = [
            {"type": "module", "name": "circuit", "url": "https://example.com/circuit"}
        ]
        yield mock


@pytest.fixture
def mock_get_component_docs():
    """Mock get_component_docs function."""
    with patch("data_fetcher.get_component_docs") as mock:
        mock.return_value = "Component documentation"
        yield mock


@pytest.fixture
def mock_get_pattern_docs():
    """Mock get_pattern_docs function."""
    with patch("data_fetcher.get_pattern_docs") as mock:
        mock.return_value = "Pattern documentation"
        yield mock


@pytest.fixture
def mock_get_style_docs():
    """Mock get_style_docs function."""
    with patch("data_fetcher.get_style_docs") as mock:
        mock.return_value = "Style documentation"
        yield mock


@pytest.fixture
def mock_search_qiskit_docs():
    """Mock search_qiskit_docs function."""
    with patch("data_fetcher.search_qiskit_docs") as mock:
        mock.return_value = [
            {
                "type": "module",
                "name": "circuit",
                "url": "https://docs.quantum.ibm.com/api/qiskit/circuit",
            }
        ]
        yield mock


@pytest.fixture
def sample_module_docs():
    """Sample module documentation."""
    return {
        "name": "circuit",
        "description": "Quantum circuit module",
        "url": "https://docs.quantum.ibm.com/api/qiskit/circuit",
        "content": "Detailed circuit documentation...",
    }


@pytest.fixture
def sample_addon_docs():
    """Sample addon documentation."""
    return {
        "name": "addon-vqe",
        "description": "Variational Quantum Eigensolver addon",
        "url": "https://docs.quantum.ibm.com/guides/vqe",
        "content": "VQE implementation details...",
    }


@pytest.fixture
def sample_guide_docs():
    """Sample guide documentation."""
    return {
        "name": "optimization",
        "description": "Quantum optimization guide",
        "url": "https://docs.quantum.ibm.com/guides/optimization",
        "content": "Optimization techniques and best practices...",
    }


@pytest.fixture
def sample_search_results():
    """Sample search results."""
    return [
        {
            "type": "sdk_module",
            "name": "circuit",
            "url": "https://docs.quantum.ibm.com/api/qiskit/circuit",
        },
        {
            "type": "addon",
            "name": "addon-opt-mapper",
            "url": "https://docs.quantum.ibm.com/guides/qaoa-mapper",
        },
        {
            "type": "guide",
            "name": "optimization",
            "url": "https://docs.quantum.ibm.com/guides/optimization",
        },
    ]
