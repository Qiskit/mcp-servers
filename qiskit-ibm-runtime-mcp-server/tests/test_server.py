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
    get_backend_calibration,
    get_backend_properties,
    get_bell_state_circuit,
    get_ghz_state_circuit,
    get_instance_from_env,
    get_job_status,
    get_quantum_random_circuit,
    get_service_status,
    get_superposition_circuit,
    get_token_from_env,
    initialize_service,
    least_busy_backend,
    list_backends,
    list_my_jobs,
    run_sampler,
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


class TestGetInstanceFromEnv:
    """Test get_instance_from_env function."""

    def test_get_instance_from_env_valid(self):
        """Test getting valid instance from environment."""
        with patch.dict(os.environ, {"QISKIT_IBM_RUNTIME_MCP_INSTANCE": "my-instance-crn"}):
            instance = get_instance_from_env()
            assert instance == "my-instance-crn"

    def test_get_instance_from_env_empty(self):
        """Test getting instance when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            instance = get_instance_from_env()
            assert instance is None

    def test_get_instance_from_env_whitespace(self):
        """Test that whitespace-only instance returns None."""
        with patch.dict(os.environ, {"QISKIT_IBM_RUNTIME_MCP_INSTANCE": "   "}):
            instance = get_instance_from_env()
            assert instance is None

    def test_get_instance_from_env_strips_whitespace(self):
        """Test that instance value is stripped of whitespace."""
        with patch.dict(os.environ, {"QISKIT_IBM_RUNTIME_MCP_INSTANCE": "  my-instance  "}):
            instance = get_instance_from_env()
            assert instance == "my-instance"


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

    def test_initialize_service_with_instance_parameter(self, mock_runtime_service):
        """Test initialization with explicit instance parameter."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service(instance="my-instance-crn")

            assert service == mock_runtime_service
            mock_qrs.assert_called_once_with(
                channel="ibm_quantum_platform", instance="my-instance-crn"
            )

    def test_initialize_service_with_instance_from_env(self, mock_runtime_service):
        """Test initialization with instance from environment variable."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs,
            patch.dict(os.environ, {"QISKIT_IBM_RUNTIME_MCP_INSTANCE": "env-instance-crn"}),
        ):
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service()

            assert service == mock_runtime_service
            mock_qrs.assert_called_once_with(
                channel="ibm_quantum_platform", instance="env-instance-crn"
            )

    def test_initialize_service_explicit_instance_overrides_env(self, mock_runtime_service):
        """Test that explicit instance parameter overrides environment variable."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs,
            patch.dict(os.environ, {"QISKIT_IBM_RUNTIME_MCP_INSTANCE": "env-instance-crn"}),
        ):
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service(instance="explicit-instance-crn")

            assert service == mock_runtime_service
            mock_qrs.assert_called_once_with(
                channel="ibm_quantum_platform", instance="explicit-instance-crn"
            )

    def test_initialize_service_with_token_and_instance(self, mock_runtime_service):
        """Test initialization with both token and instance."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService") as mock_qrs:
            mock_qrs.return_value = mock_runtime_service

            service = initialize_service(token="test_token", instance="my-instance-crn")

            assert service == mock_runtime_service
            mock_qrs.save_account.assert_called_once_with(
                channel="ibm_quantum_platform", token="test_token", overwrite=True
            )
            mock_qrs.assert_called_with(channel="ibm_quantum_platform", instance="my-instance-crn")


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
            assert "No quantum backends available" in result["message"]


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

    @pytest.mark.asyncio
    async def test_get_backend_properties_processor_type_string(self, mock_runtime_service):
        """Test properties includes processor_type when it's a string."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]
            mock_config.basis_gates = ["cx", "id", "rz"]
            mock_config.max_shots = 8192
            mock_config.max_experiments = 300
            mock_config.processor_type = "Heron"
            mock_config.backend_version = "2.0.0"

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "success"
            assert result["processor_type"] == "Heron"
            assert result["backend_version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_get_backend_properties_processor_type_dict(self, mock_runtime_service):
        """Test properties handles processor_type as dict with family and revision."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]
            mock_config.basis_gates = ["cx", "id", "rz"]
            mock_config.max_shots = 8192
            mock_config.max_experiments = 300
            mock_config.processor_type = {"family": "Eagle", "revision": "3"}
            mock_config.backend_version = "1.5.2"

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "success"
            assert result["processor_type"] == "Eagle r3"
            assert result["backend_version"] == "1.5.2"

    @pytest.mark.asyncio
    async def test_get_backend_properties_missing_config_attrs(self, mock_runtime_service):
        """Test properties handles missing config attributes gracefully."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_config = Mock(spec=[])  # Empty spec means no attributes
            mock_config.coupling_map = [[0, 1]]
            mock_config.basis_gates = ["cx", "id", "rz"]
            mock_config.max_shots = 8192
            mock_config.max_experiments = 300
            # Don't set processor_type or backend_version

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_properties("ibm_brisbane")

            assert result["status"] == "success"
            # Should have the keys but with None values
            assert result["processor_type"] is None
            assert result["backend_version"] is None


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


class TestGetBackendCalibration:
    """Test get_backend_calibration function."""

    @pytest.mark.asyncio
    async def test_get_calibration_success(self, mock_runtime_service):
        """Test successful calibration data retrieval."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock backend properties (calibration data)
            mock_properties = Mock()
            mock_properties.t1.return_value = 150.5  # microseconds
            mock_properties.t2.return_value = 80.2  # microseconds
            mock_properties.readout_error.return_value = 0.015
            mock_properties.prob_meas0_prep1.return_value = 0.012
            mock_properties.prob_meas1_prep0.return_value = 0.018
            mock_properties.gate_error.return_value = 0.001
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"

            # Mock backend configuration
            mock_config = Mock()
            mock_config.coupling_map = [[0, 1], [1, 2], [2, 3]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "success"
            assert result["backend_name"] == "ibm_brisbane"
            assert "qubit_calibration" in result
            assert "gate_errors" in result
            assert "last_calibration" in result
            assert len(result["qubit_calibration"]) > 0

            # Check qubit data
            qubit_data = result["qubit_calibration"][0]
            assert "t1_us" in qubit_data
            assert "t2_us" in qubit_data
            assert "readout_error" in qubit_data

    @pytest.mark.asyncio
    async def test_get_calibration_specific_qubits(self, mock_runtime_service):
        """Test calibration data for specific qubits."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_properties = Mock()
            mock_properties.t1.return_value = 100.0
            mock_properties.t2.return_value = 50.0
            mock_properties.readout_error.return_value = 0.02
            mock_properties.prob_meas0_prep1.return_value = None
            mock_properties.prob_meas1_prep0.return_value = None
            mock_properties.gate_error.return_value = 0.001
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane", qubit_indices=[0, 5, 10])

            assert result["status"] == "success"
            # Should have data for requested qubits (filtered by num_qubits)
            assert len(result["qubit_calibration"]) <= 3

    @pytest.mark.asyncio
    async def test_get_calibration_no_properties(self, mock_runtime_service):
        """Test calibration when properties are not available (e.g., simulator)."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = None

            result = await get_backend_calibration("ibmq_qasm_simulator")

            assert result["status"] == "error"
            assert "No calibration data available" in result["message"]

    @pytest.mark.asyncio
    async def test_get_calibration_properties_exception(self, mock_runtime_service):
        """Test calibration when properties() raises an exception."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.side_effect = Exception("Properties not available")

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "error"
            assert "Calibration data not available" in result["message"]

    @pytest.mark.asyncio
    async def test_get_calibration_service_failure(self):
        """Test calibration when service initialization fails."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.side_effect = Exception("Service initialization failed")

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "error"
            assert "Failed to get backend calibration" in result["message"]

    @pytest.mark.asyncio
    async def test_get_calibration_partial_data(self, mock_runtime_service):
        """Test calibration when some data points are missing."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            # Mock properties where some methods raise exceptions
            mock_properties = Mock()
            mock_properties.t1.return_value = 120.0
            mock_properties.t2.side_effect = Exception("T2 not available")
            mock_properties.readout_error.return_value = 0.01
            mock_properties.prob_meas0_prep1.side_effect = Exception("Not available")
            mock_properties.prob_meas1_prep0.side_effect = Exception("Not available")
            mock_properties.gate_error.side_effect = Exception("Not available")
            mock_properties.last_update_date = None
            mock_properties.faulty_qubits.return_value = []
            mock_properties.faulty_gates.return_value = []
            mock_properties.frequency.side_effect = Exception("Not available")

            mock_config = Mock()
            mock_config.coupling_map = []

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "success"
            # Should still return partial data
            qubit_data = result["qubit_calibration"][0]
            assert qubit_data["t1_us"] is not None
            assert qubit_data["t2_us"] is None  # Was exception

    @pytest.mark.asyncio
    async def test_get_calibration_faulty_qubits(self, mock_runtime_service):
        """Test calibration includes faulty_qubits data."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_properties = Mock()
            mock_properties.t1.return_value = 100.0
            mock_properties.t2.return_value = 50.0
            mock_properties.readout_error.return_value = 0.02
            mock_properties.prob_meas0_prep1.return_value = None
            mock_properties.prob_meas1_prep0.return_value = None
            mock_properties.gate_error.return_value = 0.001
            mock_properties.frequency.return_value = 5.0e9  # 5 GHz in Hz
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"
            mock_properties.faulty_qubits.return_value = [3, 7, 15]
            mock_properties.faulty_gates.return_value = []

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "success"
            assert "faulty_qubits" in result
            assert result["faulty_qubits"] == [3, 7, 15]

    @pytest.mark.asyncio
    async def test_get_calibration_faulty_gates(self, mock_runtime_service):
        """Test calibration includes faulty_gates data."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_properties = Mock()
            mock_properties.t1.return_value = 100.0
            mock_properties.t2.return_value = 50.0
            mock_properties.readout_error.return_value = 0.02
            mock_properties.prob_meas0_prep1.return_value = None
            mock_properties.prob_meas1_prep0.return_value = None
            mock_properties.gate_error.return_value = 0.001
            mock_properties.frequency.return_value = 5.0e9
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"
            mock_properties.faulty_qubits.return_value = []

            # Mock faulty gates
            mock_faulty_gate = Mock()
            mock_faulty_gate.gate = "cx"
            mock_faulty_gate.qubits = [5, 6]
            mock_properties.faulty_gates.return_value = [mock_faulty_gate]

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "success"
            assert "faulty_gates" in result
            assert len(result["faulty_gates"]) == 1
            assert result["faulty_gates"][0]["gate"] == "cx"
            assert result["faulty_gates"][0]["qubits"] == [5, 6]

    @pytest.mark.asyncio
    async def test_get_calibration_frequency(self, mock_runtime_service):
        """Test calibration includes qubit frequency in GHz."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_properties = Mock()
            mock_properties.t1.return_value = 100.0
            mock_properties.t2.return_value = 50.0
            mock_properties.readout_error.return_value = 0.02
            mock_properties.prob_meas0_prep1.return_value = None
            mock_properties.prob_meas1_prep0.return_value = None
            mock_properties.gate_error.return_value = 0.001
            mock_properties.frequency.return_value = 5.123456e9  # 5.123456 GHz in Hz
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"
            mock_properties.faulty_qubits.return_value = []
            mock_properties.faulty_gates.return_value = []

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane")

            assert result["status"] == "success"
            qubit_data = result["qubit_calibration"][0]
            assert "frequency_ghz" in qubit_data
            assert qubit_data["frequency_ghz"] == 5.123456  # Converted to GHz

    @pytest.mark.asyncio
    async def test_get_calibration_operational_status(self, mock_runtime_service):
        """Test calibration marks qubits as non-operational if in faulty_qubits list."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.return_value = mock_runtime_service

            mock_properties = Mock()
            mock_properties.t1.return_value = 100.0
            mock_properties.t2.return_value = 50.0
            mock_properties.readout_error.return_value = 0.02
            mock_properties.prob_meas0_prep1.return_value = None
            mock_properties.prob_meas1_prep0.return_value = None
            mock_properties.gate_error.return_value = 0.001
            mock_properties.frequency.return_value = 5.0e9
            mock_properties.last_update_date = "2024-01-15T10:00:00Z"
            # Mark qubit 0 as faulty
            mock_properties.faulty_qubits.return_value = [0]
            mock_properties.faulty_gates.return_value = []

            mock_config = Mock()
            mock_config.coupling_map = [[0, 1]]

            mock_backend = mock_runtime_service.backend.return_value
            mock_backend.properties.return_value = mock_properties
            mock_backend.configuration.return_value = mock_config

            result = await get_backend_calibration("ibm_brisbane", qubit_indices=[0, 1])

            assert result["status"] == "success"
            qubit_data = result["qubit_calibration"]

            # Qubit 0 should be marked as non-operational (in faulty_qubits)
            qubit_0 = next(q for q in qubit_data if q["qubit"] == 0)
            assert qubit_0["operational"] is False

            # Qubit 1 should be operational (not in faulty_qubits)
            qubit_1 = next(q for q in qubit_data if q["qubit"] == 1)
            assert qubit_1["operational"] is True


class TestRunSampler:
    """Test run_sampler function."""

    # Sample valid QASM3 circuit for testing
    SAMPLE_QASM3 = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
"""

    # Sample valid QASM2 circuit for testing (legacy format)
    SAMPLE_QASM2 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

    @pytest.mark.asyncio
    async def test_run_sampler_success(self, mock_runtime_service):
        """Test successful sampler execution with QASM3 and default error mitigation."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service

            # Mock circuit loading
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            # Mock SamplerOptions
            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            # Mock the job returned by sampler.run()
            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_123"

            # Mock the sampler instance
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(self.SAMPLE_QASM3, "ibm_brisbane", 1024)

            assert result["status"] == "success"
            assert result["job_id"] == "sampler_job_123"
            assert result["backend"] == "ibm_brisbane"
            assert result["shots"] == 1024
            # Verify error mitigation in response (defaults)
            assert "error_mitigation" in result
            assert result["error_mitigation"]["dynamical_decoupling"]["enabled"] is True
            assert result["error_mitigation"]["dynamical_decoupling"]["sequence"] == "XY4"
            assert result["error_mitigation"]["twirling"]["gates_enabled"] is True
            assert result["error_mitigation"]["twirling"]["measure_enabled"] is True
            mock_sampler_class.assert_called_once()
            mock_load.assert_called_once_with(self.SAMPLE_QASM3, "auto")

    @pytest.mark.asyncio
    async def test_run_sampler_with_qasm2(self, mock_runtime_service):
        """Test sampler with legacy QASM2 circuit."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service

            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_qasm2"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(self.SAMPLE_QASM2, "ibm_brisbane", 1024, "qasm3")

            assert result["status"] == "success"
            mock_load.assert_called_once_with(self.SAMPLE_QASM2, "qasm3")

    @pytest.mark.asyncio
    async def test_run_sampler_with_qpy_format(self, mock_runtime_service):
        """Test sampler with QPY format."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service

            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_qpy"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            qpy_data = "base64_encoded_qpy_data"
            result = await run_sampler(qpy_data, "ibm_brisbane", 1024, "qpy")

            assert result["status"] == "success"
            mock_load.assert_called_once_with(qpy_data, "qpy")

    @pytest.mark.asyncio
    async def test_run_sampler_least_busy_backend(self, mock_runtime_service):
        """Test sampler uses least busy backend when none specified."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy") as mock_least_busy,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service

            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            # Set up least_busy to return a specific backend
            mock_backend = Mock()
            mock_backend.name = "ibm_least_busy"
            mock_least_busy.return_value = mock_backend

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_456"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(self.SAMPLE_QASM3)

            assert result["status"] == "success"
            assert result["backend"] == "ibm_least_busy"
            mock_least_busy.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sampler_invalid_circuit(self, mock_runtime_service):
        """Test sampler with invalid circuit."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
        ):
            mock_init.return_value = mock_runtime_service
            mock_load.return_value = {
                "status": "error",
                "message": "QASM string not valid. QASM3 error: ...; QASM2 error: ...",
            }

            result = await run_sampler("invalid circuit string", "ibm_brisbane")

            assert result["status"] == "error"
            assert "QASM string not valid" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_backend_not_found(self, mock_runtime_service):
        """Test sampler with non-existent backend."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}
            mock_runtime_service.backend.side_effect = Exception("Backend not found")

            result = await run_sampler(self.SAMPLE_QASM3, "nonexistent_backend")

            assert result["status"] == "error"
            assert "Failed to get backend" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_no_operational_backend(self, mock_runtime_service):
        """Test sampler when no operational backends available."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.least_busy") as mock_least_busy,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}
            mock_least_busy.return_value = None  # No operational backend

            result = await run_sampler(self.SAMPLE_QASM3)

            assert result["status"] == "error"
            assert "No operational backend available" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_invalid_shots(self, mock_runtime_service):
        """Test sampler with invalid shots parameter."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            result = await run_sampler(self.SAMPLE_QASM3, "ibm_brisbane", shots=0)

            assert result["status"] == "error"
            assert "shots must be at least 1" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_service_not_initialized(self):
        """Test sampler when service initialization fails."""
        with patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init:
            mock_init.side_effect = Exception("Service initialization failed")

            result = await run_sampler(self.SAMPLE_QASM3, "ibm_brisbane")

            assert result["status"] == "error"
            assert "Failed to run sampler" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_submission_failure(self, mock_runtime_service):
        """Test sampler when job submission fails."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_sampler = Mock()
            mock_sampler.run.side_effect = Exception("Job submission failed")
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(self.SAMPLE_QASM3, "ibm_brisbane")

            assert result["status"] == "error"
            assert "Failed to run sampler" in result["message"]

    @pytest.mark.asyncio
    async def test_run_sampler_default_shots(self, mock_runtime_service):
        """Test sampler uses default shots when not specified."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_789"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(self.SAMPLE_QASM3, "ibm_brisbane")

            assert result["status"] == "success"
            assert result["shots"] == 4096  # Default value

    @pytest.mark.asyncio
    async def test_run_sampler_error_mitigation_disabled(self, mock_runtime_service):
        """Test sampler with error mitigation disabled."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_no_mitigation"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            # Disable all error mitigation
            result = await run_sampler(
                self.SAMPLE_QASM3,
                "ibm_brisbane",
                1024,
                "auto",
                dynamical_decoupling=False,
                dd_sequence="XX",
                twirling=False,
                measure_twirling=False,
            )

            assert result["status"] == "success"
            assert result["error_mitigation"]["dynamical_decoupling"]["enabled"] is False
            assert result["error_mitigation"]["dynamical_decoupling"]["sequence"] is None
            assert result["error_mitigation"]["twirling"]["gates_enabled"] is False
            assert result["error_mitigation"]["twirling"]["measure_enabled"] is False

    @pytest.mark.asyncio
    async def test_run_sampler_custom_dd_sequence(self, mock_runtime_service):
        """Test sampler with custom dynamical decoupling sequence."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_xx"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            # Use XX sequence instead of default XY4
            result = await run_sampler(
                self.SAMPLE_QASM3,
                "ibm_brisbane",
                1024,
                "auto",
                dynamical_decoupling=True,
                dd_sequence="XX",
            )

            assert result["status"] == "success"
            assert result["error_mitigation"]["dynamical_decoupling"]["enabled"] is True
            assert result["error_mitigation"]["dynamical_decoupling"]["sequence"] == "XX"

    @pytest.mark.asyncio
    async def test_run_sampler_twirling_gates_only(self, mock_runtime_service):
        """Test sampler with only gate twirling enabled (no measure twirling)."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_gates_only"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(
                self.SAMPLE_QASM3,
                "ibm_brisbane",
                1024,
                "auto",
                twirling=True,
                measure_twirling=False,
            )

            assert result["status"] == "success"
            assert result["error_mitigation"]["twirling"]["gates_enabled"] is True
            assert result["error_mitigation"]["twirling"]["measure_enabled"] is False

    @pytest.mark.asyncio
    async def test_run_sampler_xpxm_sequence(self, mock_runtime_service):
        """Test sampler with XpXm dynamical decoupling sequence."""
        with (
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.initialize_service") as mock_init,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerV2") as mock_sampler_class,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.load_circuit") as mock_load,
            patch("qiskit_ibm_runtime_mcp_server.ibm_runtime.SamplerOptions") as mock_options,
        ):
            mock_init.return_value = mock_runtime_service
            mock_circuit = Mock()
            mock_load.return_value = {"status": "success", "circuit": mock_circuit}

            mock_opts_instance = Mock()
            mock_opts_instance.dynamical_decoupling = Mock()
            mock_opts_instance.twirling = Mock()
            mock_options.return_value = mock_opts_instance

            mock_job = Mock()
            mock_job.job_id.return_value = "sampler_job_xpxm"
            mock_sampler = Mock()
            mock_sampler.run.return_value = mock_job
            mock_sampler_class.return_value = mock_sampler

            result = await run_sampler(
                self.SAMPLE_QASM3,
                "ibm_brisbane",
                1024,
                "auto",
                dd_sequence="XpXm",
            )

            assert result["status"] == "success"
            assert result["error_mitigation"]["dynamical_decoupling"]["sequence"] == "XpXm"


class TestExampleCircuits:
    """Test example circuit functions for LLM usability."""

    def test_bell_state_circuit_structure(self):
        """Test Bell state circuit has correct structure."""
        result = get_bell_state_circuit()

        assert "circuit" in result
        assert "name" in result
        assert "description" in result
        assert "expected_results" in result
        assert "num_qubits" in result
        assert "usage" in result

        assert result["name"] == "Bell State"
        assert result["num_qubits"] == 2
        assert "entanglement" in result["description"].lower()

    def test_bell_state_circuit_valid_qasm3(self):
        """Test Bell state circuit is valid QASM3."""
        result = get_bell_state_circuit()
        circuit = result["circuit"]

        assert "OPENQASM 3.0" in circuit
        assert 'include "stdgates.inc"' in circuit
        assert "qubit[2]" in circuit
        assert "bit[2]" in circuit
        assert "h q[0]" in circuit
        assert "cx q[0], q[1]" in circuit
        assert "measure" in circuit

    def test_ghz_state_circuit_default(self):
        """Test GHZ state circuit with default 3 qubits."""
        result = get_ghz_state_circuit()

        assert result["num_qubits"] == 3
        assert "GHZ" in result["name"]
        assert "000" in result["expected_results"]
        assert "111" in result["expected_results"]

    def test_ghz_state_circuit_custom_qubits(self):
        """Test GHZ state circuit with custom qubit count."""
        result = get_ghz_state_circuit(5)

        assert result["num_qubits"] == 5
        assert "5-qubit" in result["name"]
        assert "00000" in result["expected_results"]
        assert "11111" in result["expected_results"]

        circuit = result["circuit"]
        assert "qubit[5]" in circuit
        assert "bit[5]" in circuit
        # Should have 4 CNOT gates for 5 qubits
        assert circuit.count("cx q[") == 4

    def test_ghz_state_circuit_min_qubits(self):
        """Test GHZ state circuit enforces minimum 2 qubits."""
        result = get_ghz_state_circuit(1)
        assert result["num_qubits"] == 2

    def test_ghz_state_circuit_max_qubits(self):
        """Test GHZ state circuit enforces maximum 10 qubits."""
        result = get_ghz_state_circuit(15)
        assert result["num_qubits"] == 10

    def test_quantum_random_circuit_structure(self):
        """Test quantum random circuit has correct structure."""
        result = get_quantum_random_circuit()

        assert result["name"] == "Quantum Random Number Generator"
        assert result["num_qubits"] == 4
        assert "random" in result["description"].lower()
        assert "16" in result["expected_results"]  # 16 possible outcomes

    def test_quantum_random_circuit_valid_qasm3(self):
        """Test quantum random circuit is valid QASM3."""
        result = get_quantum_random_circuit()
        circuit = result["circuit"]

        assert "OPENQASM 3.0" in circuit
        assert "qubit[4]" in circuit
        # Should have 4 Hadamard gates
        assert circuit.count("h q[") == 4
        assert "measure" in circuit

    def test_superposition_circuit_structure(self):
        """Test superposition circuit has correct structure."""
        result = get_superposition_circuit()

        assert result["name"] == "Single Qubit Superposition"
        assert result["num_qubits"] == 1
        assert "simplest" in result["description"].lower()
        assert "50%" in result["expected_results"]

    def test_superposition_circuit_valid_qasm3(self):
        """Test superposition circuit is valid QASM3."""
        result = get_superposition_circuit()
        circuit = result["circuit"]

        assert "OPENQASM 3.0" in circuit
        assert "qubit[1]" in circuit
        assert "bit[1]" in circuit
        assert "h q[0]" in circuit
        assert "measure" in circuit

    def test_all_circuits_have_usage_instructions(self):
        """Test all example circuits include usage instructions."""
        circuits = [
            get_bell_state_circuit(),
            get_ghz_state_circuit(),
            get_quantum_random_circuit(),
            get_superposition_circuit(),
        ]

        for circuit in circuits:
            assert "usage" in circuit
            assert "run_sampler_tool" in circuit["usage"]


# Assisted by watsonx Code Assistant
