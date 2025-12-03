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

"""Test configuration and fixtures for Qiskit Code Assistant MCP Server tests."""

import os
from unittest.mock import patch

import httpx
import pytest
import respx


# Note: We cannot import from qiskit_code_assistant_mcp_server at module level
# because it triggers utils.py which calls _get_token_from_system() at import time.
# This would fail in CI where no token is available. Instead, we use hardcoded
# test values here and import lazily inside fixtures where needed.

# Test constants (must match values set in mock_env_vars)
TEST_QCA_API_BASE = "https://test-qca-api.example.com"
TEST_QCA_MODEL_NAME = "test-model"


@pytest.fixture(autouse=True)
async def reset_http_client():
    """Reset the global HTTP client before and after each test.

    This fixture ensures clean state between tests and properly closes
    any async client that was created during the test.
    """
    # Import here to avoid triggering token lookup at module load time
    import qiskit_code_assistant_mcp_server.utils as utils_module

    # Reset before test - force clear without trying to close
    # (might already be closed or in invalid state)
    utils_module._client = None
    utils_module._cached_token = None  # Reset cached token for fresh state
    utils_module._token_checked = False  # Reset token check flag

    yield

    # Reset after test - properly close if still open
    if utils_module._client is not None:
        try:
            if not utils_module._client.is_closed:
                await utils_module._client.aclose()
        except Exception:
            pass  # Ignore errors during cleanup
        finally:
            utils_module._client = None
    utils_module._cached_token = None  # Reset cached token after test
    utils_module._token_checked = False  # Reset token check flag after test


@pytest.fixture
async def http_client_for_tests(mock_env_vars):
    """Create an HTTP client in proper async context for tests.

    This fixture pre-creates the AsyncClient in the async context,
    avoiding sniffio detection issues in Python 3.14+.
    Use this fixture for tests that need to make actual HTTP requests
    (with respx mocking).
    """
    # Import here after mock_env_vars has set up the environment
    import qiskit_code_assistant_mcp_server.utils as utils_module

    # Use test constants directly (matching mock_env_vars values)
    test_timeout = 10.0
    test_x_caller = "qiskit-mcp-server"

    # Get token from env (set by mock_env_vars)
    token = os.environ.get("QISKIT_IBM_TOKEN", "test_token")

    headers = {
        "x-caller": test_x_caller,
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    utils_module._client = httpx.AsyncClient(
        headers=headers,
        timeout=httpx.Timeout(test_timeout),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
    )

    yield utils_module._client

    # Cleanup is handled by reset_http_client autouse fixture


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing.

    This fixture patches both environment variables AND the constants
    in qca.py where they're already imported (since constants are
    read at module import time).
    """
    with patch.dict(
        os.environ,
        {
            "QISKIT_IBM_TOKEN": "test_token_12345",
            "QCA_TOOL_API_BASE": TEST_QCA_API_BASE,
            "QCA_TOOL_MODEL_NAME": TEST_QCA_MODEL_NAME,
            "QCA_REQUEST_TIMEOUT": "10.0",
            "QCA_MCP_DEBUG_LEVEL": "DEBUG",
        },
    ):
        # Also patch the constants in qca.py where they've already been imported
        with patch("qiskit_code_assistant_mcp_server.qca.QCA_TOOL_API_BASE", TEST_QCA_API_BASE):
            with patch(
                "qiskit_code_assistant_mcp_server.qca.QCA_TOOL_MODEL_NAME", TEST_QCA_MODEL_NAME
            ):
                yield


@pytest.fixture
def mock_qiskit_credentials(tmp_path):
    """Mock Qiskit credentials file."""
    qiskit_dir = tmp_path / ".qiskit"
    qiskit_dir.mkdir()

    credentials = {
        "default-ibm-quantum-platform": {
            "token": "test_token_from_file",
            "url": "https://auth.quantum-computing.ibm.com/api",
        }
    }

    import json

    credentials_file = qiskit_dir / "qiskit-ibm.json"
    with open(credentials_file, "w") as f:
        json.dump(credentials, f)

    with patch("pathlib.Path.home", return_value=tmp_path):
        yield credentials_file


@pytest.fixture
def mock_http_responses(mock_env_vars):
    """Mock HTTP responses for QCA API calls.

    Note: This fixture depends on mock_env_vars to ensure constants are patched.
    """
    with respx.mock(assert_all_called=False) as respx_mock:
        # Mock models list endpoint
        respx_mock.get(f"{TEST_QCA_API_BASE}/v1/models").mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "mistral-small-3.2-24b-qiskit",
                            "name": "Mistral small 3.2 24b Qiskit",
                        },
                        {"id": "test-model", "name": "Test Model"},
                    ]
                },
            )
        )

        # Mock model details endpoint
        respx_mock.get(f"{TEST_QCA_API_BASE}/v1/model/mistral-small-3.2-24b-qiskit").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "mistral-small-3.2-24b-qiskit",
                    "name": "Mistral small 3.2 24b Qiskit",
                    "description": "Test model for quantum code assistance",
                },
            )
        )

        # Mock model disclaimer endpoint
        respx_mock.get(
            f"{TEST_QCA_API_BASE}/v1/model/mistral-small-3.2-24b-qiskit/disclaimer"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "disclaimer_123",
                    "model_id": "mistral-small-3.2-24b-qiskit",
                    "text": "This is a test disclaimer",
                },
            )
        )

        # Mock completion endpoint
        respx_mock.post(f"{TEST_QCA_API_BASE}/v1/completions").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "completion_456",
                    "choices": [
                        {
                            "text": "# Create a quantum circuit\nqc = QuantumCircuit(2, 2)",
                            "index": 0,
                        }
                    ],
                },
            )
        )

        # Mock disclaimer acceptance endpoint
        respx_mock.post(
            f"{TEST_QCA_API_BASE}/v1/model/mistral-small-3.2-24b-qiskit/disclaimer"
        ).mock(return_value=httpx.Response(200, json={"success": True}))

        # Mock completion acceptance endpoint
        respx_mock.post(f"{TEST_QCA_API_BASE}/v1/completion/acceptance").mock(
            return_value=httpx.Response(200, json={"result": "accepted"})
        )

        yield respx_mock


@pytest.fixture
def mock_http_error_responses(mock_env_vars):
    """Mock HTTP error responses for testing error handling.

    Note: This fixture depends on mock_env_vars to ensure constants are patched.
    """
    with respx.mock(assert_all_called=False) as respx_mock:
        # Mock 401 Unauthorized
        respx_mock.get(f"{TEST_QCA_API_BASE}/v1/models").mock(
            return_value=httpx.Response(401, json={"detail": "Invalid authentication credentials"})
        )

        # Mock 500 Server Error
        respx_mock.post(f"{TEST_QCA_API_BASE}/v1/completions").mock(
            return_value=httpx.Response(500, json={"detail": "Internal server error"})
        )

        yield respx_mock


@pytest.fixture
def sample_completion_request(mock_env_vars):
    """Sample completion request data."""
    return {
        "prompt": "Create a quantum circuit with 2 qubits",
        "model": TEST_QCA_MODEL_NAME,
    }


@pytest.fixture
def sample_models_response():
    """Sample models API response."""
    return {
        "data": [
            {
                "id": "mistral-small-3.2-24b-qiskit",
                "name": "Mistral Small 3.2 24b Qiskit",
            },
            {"id": "granite-3.3-8b-qiskit", "name": "Granite 8B Qiskit"},
        ]
    }


# Assisted by watsonx Code Assistant
