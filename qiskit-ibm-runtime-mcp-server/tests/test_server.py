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

"""Unit tests for IBM Runtime MCP Server functions."""

import os
from unittest.mock import Mock, patch

import pytest

from qiskit_ibm_runtime_mcp_server.ibm_runtime import (
    cancel_job,
    get_backend_properties,
    get_job_status,
    get_service_status,
    get_token_from_env,
    initialize_service,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    setup_ibm_quantum_account,
)


class TestGetTokenFromEnv:
    """Test get_token_from_env function."""

    def test_get_token_from_env_valid(self):
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


class TestInitializeService:
    """Test service initialization function."""

    def test_initialize_service_existing_account(self, mock_runtime_service):
        """Test initialization with existing account."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service
            mock_qrs.assert_called_once_with(channel="ibm_quantum_platform")

    def test_initialize_service_with_token(self, mock_runtime_service):
        """Test initialization with provided token."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service(token="test_token", channel="ibm_quantum_platform")

            assert service == mock_runtime_service
            mock_qrs.save_account.assert_called_once_with(
                channel="ibm_quantum_platform", token="test_token", overwrite=True
            )

    def test_initialize_service_with_env_token(self, mock_runtime_service, mock_env_vars):
        """Test initialization with environment token."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service

    def test_initialize_service_no_token_available(self):
        """Test initialization failure when no token is available."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs,
            patch.dict(os.environ, {}, clear=True),
        ):
            mock_qrs.side_effect = Exception("No account")

            with pytest.raises(ValueError) as exc_info:
                initialize_service()

            assert "No IBM Quantum token provided" in str(exc_info.value)

    def test_initialize_service_invalid_token(self):
        """Test initialization with invalid token."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.side_effect = Exception("No account")
            mock_qrs.save_account.side_effect = Exception("Invalid token")

            with pytest.raises(ValueError) as exc_info:
                initialize_service(token="invalid_token")

            assert "Invalid token or channel" in str(exc_info.value)

    def test_initialize_service_placeholder_token(self):
        """Test that placeholder tokens are rejected."""
        with pytest.raises(ValueError) as exc_info:
            initialize_service(token="<PASSWORD>")

        assert "appears to be a placeholder value" in str(exc_info.value)

    def test_initialize_service_prioritizes_saved_credentials(self, mock_runtime_service):
        """Test that saved credentials are tried first when no token provided."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service
            # Should NOT call save_account
            mock_qrs.save_account.assert_not_called()


class TestSetupIBMQuantumAccount:
    """Test setup_ibm_quantum_account function."""

    @pytest.mark.asyncio
    async def test_setup_account_success(self, mock_runtime_service):
        """Test successful account setup."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await setup_ibm_quantum_account("test_token")

            assert result["status"] == "success"
            assert result["available_backends"] == 2
            assert result["channel"] == "ibm_quantum_platform"
            mock_init.assert_called_once_with("test_token", "ibm_quantum_platform")

    @pytest.mark.asyncio
    async def test_setup_account_empty_token_with_saved_credentials(self, mock_runtime_service):
        """Test setup with empty token falls back to saved credentials."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.get_token_from_env") as mock_env,
        ):
            mock_env.return_value = None  # No env token
            mock_init.return_value = mock_runtime_service

            result = await setup_ibm_quantum_account("")

            assert result["status"] == "success"
            # Should initialize with None to use saved credentials
            mock_init.assert_called_once_with(None, "ibm_quantum_platform")

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
    async def test_setup_account_initialization_failure(self):
        """Test setup when initialization fails."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.side_effect = Exception("Authentication failed")

            result = await setup_ibm_quantum_account("test_token")

            assert result["status"] == "error"
            assert "Failed to set up account" in result["message"]


class TestListBackends:
    """Test list_backends function."""

    @pytest.mark.asyncio
    async def test_list_backends_success(self, mock_runtime_service):
        """Test successful backends listing."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await list_backends()

            assert result["status"] == "success"
            assert result["total_backends"] == 2
            assert len(result["backends"]) == 2

            backend = result["backends"][0]
            assert "name" in backend
            assert "num_qubits" in backend
            assert "simulator" in backend

    @pytest.mark.asyncio
    async def test_list_backends_no_service(self):
        """Test backends listing when service is None."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", None),
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
        ):
            mock_init.side_effect = Exception("Service initialization failed")

            result = await list_backends()

            assert result["status"] == "error"
            assert "Failed to list backends" in result["message"]


class TestLeastBusyBackend:
    """Test least_busy_backend function."""

    @pytest.mark.asyncio
    async def test_least_busy_backend_success(self, mock_runtime_service):
        """Test successful least busy backend retrieval."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy") as mock_least_busy,
        ):
            mock_init.return_value = mock_runtime_service

            # Create a mock backend for least_busy to return
            mock_backend = Mock()
            mock_backend.name = "ibm_brisbane"
            mock_backend.num_qubits = 127
            mock_backend.status.return_value = Mock(
                operational=True, pending_jobs=2, status_msg="active"
            )
            mock_least_busy.return_value = mock_backend

            result = await least_busy_backend()

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"
            assert result["pending_jobs"] == 2
            assert result["operational"] is True

    @pytest.mark.asyncio
    async def test_least_busy_backend_no_operational(self, mock_runtime_service):
        """Test least busy backend when no operational backends available."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service
            mock_runtime_service.backends.return_value = []  # No operational backends

            result = await least_busy_backend()

            assert result["status"] == "error"
            assert "No operational quantum backends available" in result["message"]


class TestGetBackendProperties:
    """Test get_backend_properties function."""

    @pytest.mark.asyncio
    async def test_get_backend_properties_success(self, mock_runtime_service):
        """Test successful backend properties retrieval."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend configuration
            mock_config = Mock()
            mock_config.coupling_map = [[0, 1], [1, 2]]
            mock_config.basis_gates = ["cx", "id", "rz"]
            mock_config.max_shots = 8192
            mock_config.max_experiments = 300

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "success"
            assert "backend_name" in result
            assert result["backend_name"] == "ibm_brisbane"
            assert result["coupling_map"] == [[0, 1], [1, 2]]
            assert result["basis_gates"] == ["cx", "id", "rz"]

    @pytest.mark.asyncio
    async def test_get_backend_properties_failure(self):
        """Test backend properties retrieval failure."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.side_effect = Exception("Service initialization failed")

            result = await get_backend_properties("nonexistent_backend")

            assert result["status"] == "error"
            assert "Failed to get backend properties" in result["message"]


class TestListMyJobs:
    """Test list_my_jobs function."""

    @pytest.mark.asyncio
    async def test_list_my_jobs_success(self, mock_runtime_service):
        """Test successful jobs listing."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await list_my_jobs(5)

            assert result["status"] == "success"
            assert result["total_jobs"] == 1
            assert len(result["jobs"]) == 1

            job = result["jobs"][0]
            assert job["job_id"] == "job_123"
            assert job["status"] == "DONE"

    @pytest.mark.asyncio
    async def test_list_my_jobs_default_limit(self, mock_runtime_service):
        """Test jobs listing with default limit."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await list_my_jobs()

            assert result["status"] == "success"
            # Check that the service was called with default limit
            mock_runtime_service.jobs.assert_called_with(limit=10)


class TestGetJobStatus:
    """Test get_job_status function."""

    @pytest.mark.asyncio
    async def test_get_job_status_success(self, mock_runtime_service):
        """Test successful job status retrieval."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service):
            result = await get_job_status("job_123")

            assert result["status"] == "success"
            assert result["job_id"] == "job_123"
            assert result["job_status"] == "DONE"

    @pytest.mark.asyncio
    async def test_get_job_status_no_service(self):
        """Test job status retrieval when service is None."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", None):
            result = await get_job_status("job_123")

            assert result["status"] == "error"
            assert "service not initialized" in result["message"]

    @pytest.mark.asyncio
    async def test_get_job_status_job_not_found(self, mock_runtime_service):
        """Test job status retrieval for non-existent job."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service):
            mock_runtime_service.job.side_effect = Exception("Job not found")

            result = await get_job_status("nonexistent_job")

            assert result["status"] == "error"
            assert "Failed to get job status" in result["message"]


class TestCancelJob:
    """Test cancel_job function."""

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, mock_runtime_service):
        """Test successful job cancellation."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service):
            result = await cancel_job("job_123")

            assert result["status"] == "success"
            assert result["job_id"] == "job_123"
            assert "cancellation requested" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_job_no_service(self):
        """Test job cancellation when service is None."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", None):
            result = await cancel_job("job_123")

            assert result["status"] == "error"
            assert "service not initialized" in result["message"]

    @pytest.mark.asyncio
    async def test_cancel_job_failure(self, mock_runtime_service):
        """Test job cancellation failure."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", mock_runtime_service):
            mock_job = mock_runtime_service.job.return_value
            mock_job.cancel.side_effect = Exception("Cannot cancel job")

            result = await cancel_job("job_123")

            assert result["status"] == "error"
            assert "Failed to cancel job" in result["message"]


class TestGetServiceStatus:
    """Test get_service_status function."""

    @pytest.mark.asyncio
    async def test_get_service_status_connected(self, mock_runtime_service):
        """Test service status when connected."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            result = await get_service_status()

            assert "IBM Quantum Service Status" in result
            assert "connected" in result.lower()

    @pytest.mark.asyncio
    async def test_get_service_status_disconnected(self):
        """Test service status when disconnected."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.service", None),
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
        ):
            mock_init.side_effect = Exception("Connection failed")

            result = await get_service_status()

            assert "IBM Quantum Service Status" in result
            assert "error" in result


# Assisted by watsonx Code Assistant
