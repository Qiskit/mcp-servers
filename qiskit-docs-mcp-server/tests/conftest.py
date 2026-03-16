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

import pytest
from qiskit_docs_mcp_server.data_fetcher import clear_cache


@pytest.fixture(autouse=True)
def _clear_caches() -> None:  # type: ignore[misc]
    """Clear caches before each test for isolation."""
    clear_cache()
    yield  # type: ignore[misc]
    clear_cache()
