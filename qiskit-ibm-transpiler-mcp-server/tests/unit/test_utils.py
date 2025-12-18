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
import os
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)
from qiskit_ibm_transpiler_mcp_server.utils import (
    get_backend_service,
    get_token_from_env,
    load_qasm_circuit,
    setup_ibm_quantum_account,
)


# Get the tests directory path relative to this test file
TESTS_DIR = Path(__file__).parent.parent


class TestGetTokenFromEnv:
    """Test get_token_from_env function."""

    def test_get_token_from_env_valid(
        self,
    ):
        """Test getting valid token from environment."""
        with patch.dict(os.environ, {"QISKIT_IBM_TOKEN": "valid_token_123"}):
            token = get_token_from_env()
            assert token == "valid_token_123"

    def test_get_token_from_env_empty(self):
        """Test getting token when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            token = get_token_from_env()
            assert token is None

    def test_get_token_from_env_placeholder(self):
        """Test that placeholder tokens are rejected."""
        with patch.dict(os.environ, {"QISKIT_IBM_TOKEN": "<PASSWORD>"}):
            token = get_token_from_env()
            assert token is None

    def test_get_token_from_env_whitespace(self):
        """Test that whitespace-only tokens return None."""
        with patch.dict(os.environ, {"QISKIT_IBM_TOKEN": "   "}):
            token = get_token_from_env()
            assert token is None


class TestGetBackendService:
    """Test get_backend_service function"""

    @pytest.mark.asyncio
    async def test_get_backend_success(self, mocker, mock_runtime_service):
        """Retrieve a specific backend with success"""
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_sp_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", return_value=mock_runtime_service
        )
        backend_mock_result = create_autospec(IBMBackend)
        backend_mock_result.name = "ibm_brisbane"
        backend_mock_result.num_qubits = 127
        backend_mock_result.simulator = False
        backend_mock_result.status.return_value = MagicMock(
            operational=True, pending_jobs=5, status_msg="active"
        )
        mock_runtime_service.backend.return_value = backend_mock_result

        result = await get_backend_service(backend_name="ibm_brisbane")

        assert result["status"] == "success"
        assert result["backend"].name == "ibm_brisbane"
        assert result["backend"].num_qubits == 127
        assert result["backend"].simulator is False
        qiskit_runtime_sp_get_mock.assert_called_once()
        mock_runtime_service.backend.assert_called_once_with("ibm_brisbane")

    @pytest.mark.asyncio
    async def test_backend_not_found(self, mocker, mock_runtime_service):
        """Failed to retrieve a backend"""
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_sp_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", return_value=mock_runtime_service
        )

        mock_runtime_service.backend.side_effect = ValueError("Backend not found")

        result = await get_backend_service(backend_name="fake_backend")

        assert result["status"] == "error"
        assert "Failed to find backend fake_backend" in result["message"]
        qiskit_runtime_sp_get_mock.assert_called_once()
        mock_runtime_service.backend.assert_called_once_with("fake_backend")


class TestLoadQasmCircuit:
    """Test load_qasm_circuit_function"""

    def test_load_with_success(self):
        """Load a correct, well-formatted, QASM 3.0 string"""
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            correct_qasm = f.read()

        result = load_qasm_circuit(qasm_string=correct_qasm)

        assert result["status"] == "success"
        assert isinstance(result["circuit"], QuantumCircuit)

    def test_load_with_failure(self):
        """Load a wrong, bad-formatted, QASM string"""
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            wrong_qasm = f.read()

        result = load_qasm_circuit(qasm_string=wrong_qasm)

        assert result["status"] == "error"
        assert "QASM string not valid" in result["message"]


class TestSetupIBMAccount:
    """Test setup_ibm_account function"""

    @pytest.mark.asyncio
    async def test_setup_account_success(self, mocker, mock_runtime_service):
        """Test successful account setup."""

        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        qiskit_runtime_sp_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", return_value=mock_runtime_service
        )

        result = await setup_ibm_quantum_account("test_token")

        assert result["status"] == "success"
        assert result["available_backends"] == 2
        qiskit_runtime_sp_get_mock.assert_called_once_with(
            token="test_token", channel="ibm_quantum_platform"
        )

    @pytest.mark.asyncio
    async def test_setup_account_empty_token_with_saved_credentials(
        self, mocker, mock_runtime_service
    ):
        """Test setup with empty token falls back to saved credentials."""
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()
        get_token_from_env_mock = mocker.patch(
            "qiskit_ibm_transpiler_mcp_server.utils.get_token_from_env",
            return_value=None,
        )
        qiskit_runtime_sp_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", return_value=mock_runtime_service
        )

        result = await setup_ibm_quantum_account()

        assert result["status"] == "success"
        # Should initialize with None to use saved credentials
        qiskit_runtime_sp_get_mock.assert_called_once_with(
            token=None, channel="ibm_quantum_platform"
        )
        get_token_from_env_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_account_placeholder_token(self):
        """Test setup with placeholder token is rejected."""
        result = await setup_ibm_quantum_account("<PASSWORD>")

        assert result["status"] == "error"
        assert "appears to be a placeholder value" in result["message"]

    @pytest.mark.asyncio
    async def test_setup_account_invalid_channel(self):
        """Test setup with invalid channel."""
        result = await setup_ibm_quantum_account("test_token", "invalid_channel")

        assert result["status"] == "error"
        assert "Channel must be" in result["message"]

    @pytest.mark.asyncio
    async def test_setup_account_initialization_failure(self, mocker):
        """Test setup when initialization fails."""
        qiskit_runtime_sp = QiskitRuntimeServiceProvider()

        qiskit_runtime_sp_get_mock = mocker.patch.object(
            qiskit_runtime_sp, "get", side_effect=Exception("Authentication failed")
        )

        result = await setup_ibm_quantum_account("test_token")

        assert result["status"] == "error"
        assert "Failed to set up IBM Quantum account" in result["message"]
        qiskit_runtime_sp_get_mock.assert_called_once_with(
            token="test_token", channel="ibm_quantum_platform"
        )
