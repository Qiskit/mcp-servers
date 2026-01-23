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

"""Test configuration and fixtures for Qiskit IBM Runtime MCP Server tests."""

import os
from unittest.mock import Mock, patch

import pytest
from qiskit_ibm_runtime import QiskitRuntimeService


@pytest.fixture(autouse=True)
def reset_service():
    """Reset the global service instance before each test."""
    import qiskit_ibm_runtime_mcp_server.ibm_runtime

    # Reset service to None before each test
    qiskit_ibm_runtime_mcp_server.ibm_runtime.service = None
    yield
    # Reset service to None after each test
    qiskit_ibm_runtime_mcp_server.ibm_runtime.service = None


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {
            "QISKIT_IBM_TOKEN": "test_token_12345",
            "QISKIT_IBM_CHANNEL": "ibm_quantum_platform",
        },
    ):
        yield


@pytest.fixture
def mock_runtime_service():
    """Mock QiskitRuntimeService for testing."""
    mock_service = Mock(spec=QiskitRuntimeService)
    mock_service._channel = "ibm_quantum_platform"

    # Mock backends
    mock_backend1 = Mock()
    mock_backend1.name = "ibmq_qasm_simulator"
    mock_backend1.num_qubits = 32
    mock_backend1.simulator = True
    mock_backend1.status.return_value = Mock(
        operational=True, pending_jobs=0, status_msg="active"
    )

    mock_backend2 = Mock()
    mock_backend2.name = "ibm_brisbane"
    mock_backend2.num_qubits = 127
    mock_backend2.simulator = False
    mock_backend2.status.return_value = Mock(
        operational=True, pending_jobs=5, status_msg="active"
    )

    mock_service.backends.return_value = [mock_backend1, mock_backend2]
    mock_service.backend.return_value = mock_backend2

    # Mock jobs
    mock_job = Mock()
    mock_job.job_id.return_value = "job_123"
    mock_job.status.return_value = "DONE"
    mock_job.creation_date = "2024-01-01T10:00:00Z"
    mock_job.backend.return_value = mock_backend2
    mock_job.tags = ["test"]
    mock_job.error_message.return_value = None
    mock_job.cancel.return_value = None

    # Mock job result (SamplerV2 format)
    mock_creg_data = Mock()
    mock_creg_data.get_counts.return_value = {"00": 2048, "11": 2048}
    mock_data = Mock()
    mock_data.meas = mock_creg_data
    mock_pub_result = Mock()
    mock_pub_result.data = mock_data
    mock_job.result.return_value = [mock_pub_result]

    # Mock job metrics
    mock_job.metrics.return_value = {"usage": {"quantum_seconds": 1.5}}

    mock_service.jobs.return_value = [mock_job]
    mock_service.job.return_value = mock_job

    return mock_service


@pytest.fixture
def mock_failed_service():
    """Mock a QiskitRuntimeService that fails initialization."""
    with patch(
        "qiskit_ibm_runtime_mcp_server.ibm_runtime.QiskitRuntimeService"
    ) as mock_qrs:
        mock_qrs.side_effect = Exception("Authentication failed")
        mock_qrs.save_account.side_effect = Exception("Invalid token")
        yield mock_qrs


@pytest.fixture
def sample_backend_data():
    """Sample backend data for testing."""
    return {
        "name": "ibm_brisbane",
        "num_qubits": 127,
        "simulator": False,
        "operational": True,
        "pending_jobs": 5,
        "status_msg": "active",
        "basis_gates": ["cx", "id", "rz", "sx", "x"],
        "coupling_map": [[0, 1], [1, 2], [2, 3]],
        "max_shots": 8192,
        "max_experiments": 300,
    }


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "job_id": "job_test_123",
        "status": "DONE",
        "creation_date": "2024-01-01T10:00:00Z",
        "backend": "ibm_brisbane",
        "tags": ["test", "experiment"],
    }


@pytest.fixture
def mock_successful_init():
    """Mock successful service initialization."""

    def init_service_mock(token=None, channel="ibm_quantum_platform"):
        mock_service = Mock(spec=QiskitRuntimeService)
        mock_service._channel = channel
        return mock_service

    return init_service_mock


@pytest.fixture
def mock_estimator_service():
    """Mock que simula el servicio de estimación."""
    return Mock(name="estimator_service")


@pytest.fixture
def mock_backend():
    """Mock que simula un backend de Qiskit."""
    return Mock(name="backend")


# Assisted by watsonx Code Assistant
