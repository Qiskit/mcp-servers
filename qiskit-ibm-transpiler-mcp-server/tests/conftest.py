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
"""Test configuration and fixtures for Qiskit IBM Transpiler MCP Server tests."""

import os
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_transpiler_mcp_server.qiskit_runtime_service_provider import (
    QiskitRuntimeServiceProvider,
)


@pytest.fixture
def mock_circuit_qasm():
    """Mock qasm circuit"""
    return "dummy_circuit_qasm"


@pytest.fixture
def mock_circuit_qpy():
    """Mock QPY circuit (base64-encoded)"""
    return "UFFZX0RBVEFfTU9DSw=="  # Mock base64 QPY data


@pytest.fixture
def mock_backend():
    """Mock backend name"""
    return "fake_backend"


@pytest.fixture
def ai_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_ai_synthesis_success():
    """Successful ai_synthesis_pass_class in run_synthesis"""
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    mock_class.__name__ = "MockAISynthesis"
    return mock_class


@pytest.fixture
def mock_ai_synthesis_failure():
    """Failed ai_synthesis_pass_class in run_synthesis"""

    mock_class = MagicMock(side_effect=Exception("AI Synthesis failed"))
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_class.__name__ = "MockAISynthesis"

    return mock_class


@pytest.fixture
def backend_name():
    """Set real backend name"""
    return os.getenv("TEST_BACKEND_NAME", "ibm_torino")


@pytest.fixture
def get_backend_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def load_qasm_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def pass_manager_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def dumps_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_routing_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_clifford_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_linear_function_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_permutation_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def ai_pauli_networks_synthesis_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_get_circuit_metrics(mocker):
    """Mock _get_circuit_metrics to return consistent metrics for testing"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta._get_circuit_metrics")
    call_count = {"count": 0}

    def metrics_side_effect(circuit):
        """Return different metrics for original vs optimized calls"""
        call_count["count"] += 1
        if call_count["count"] % 2 == 1:  # Odd calls = original circuit
            return {"num_qubits": 2, "depth": 10, "size": 15, "two_qubit_gates": 5}
        else:  # Even calls = optimized circuit
            return {"num_qubits": 2, "depth": 7, "size": 12, "two_qubit_gates": 3}

    mock.side_effect = metrics_side_effect
    return mock


@pytest.fixture
def mock_load_qasm_circuit_success(mocker, mock_get_circuit_metrics):
    """Successful loading of QuantumCircuit object from QASM3.0 string (legacy fixture)"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_circuit")
    mock.return_value = {"status": "success", "circuit": "input_circuit"}
    return mock


@pytest.fixture
def mock_load_qasm_circuit_failure(mocker):
    """Failed loading of QuantumCircuit object from QASM3.0 string (legacy fixture)"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_circuit")
    mock.return_value = {
        "status": "error",
        "message": "Error in loading QuantumCircuit",
    }
    return mock


@pytest.fixture
def mock_load_circuit_success(mocker):
    """Successful loading of QuantumCircuit object (QASM3 or QPY)"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_circuit")
    mock.return_value = {"status": "success", "circuit": "input_circuit"}
    return mock


@pytest.fixture
def mock_load_circuit_failure(mocker):
    """Failed loading of QuantumCircuit object (QASM3 or QPY)"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.load_circuit")
    mock.return_value = {
        "status": "error",
        "message": "Error in loading QuantumCircuit",
    }
    return mock


@pytest.fixture
def mock_dumps_qasm_failure(mocker):
    """Failed dump_circuit method (legacy fixture name for backward compat)"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.dump_circuit",
        side_effect=Exception("Circuit dump failed"),
    )
    return mock


@pytest.fixture
def mock_dumps_qasm_success(mocker):
    """Successful dump_circuit method - returns QPY format"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.dump_circuit")
    mock.return_value = "circuit_qpy"
    return mock


@pytest.fixture
def mock_dump_circuit_success(mocker):
    """Successful dump_circuit method - returns QPY format"""
    mock = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.dump_circuit")
    mock.return_value = "circuit_qpy"
    return mock


@pytest.fixture
def mock_dump_circuit_failure(mocker):
    """Failed dump_circuit method (QASM3 or QPY)"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.dump_circuit",
        side_effect=Exception("Circuit dump failed"),
    )
    return mock


@pytest.fixture
def mock_get_backend_service_success(mocker):
    """Successful get_backend_service procedure"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.get_backend_service",
        new_callable=AsyncMock,
    )
    mock.return_value = {"backend": "mock_backend_object", "status": "success"}
    return mock


@pytest.fixture
def mock_get_backend_service_failure(mocker):
    """Failed get_backend_service procedure"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.get_backend_service",
        new_callable=AsyncMock,
    )
    mock.return_value = {"message": "get_backend failed", "status": "error"}
    return mock


@pytest.fixture
def mock_ai_routing_success(mocker):
    """Successful AIRouting procedure"""
    mock_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.AIRouting")
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_class.__name__ = "AIRouting"
    return mock_class


@pytest.fixture
def mock_ai_routing_failure(mocker):
    """Failed AIRouting procedure"""
    mock_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIRouting",
        side_effect=Exception("AIRouting failed"),
    )
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_class.__name__ = "AIRouting"
    return mock_instance


@pytest.fixture
def mock_pass_manager_success(mocker):
    """Successful PassManager run procedure"""
    mock_pm_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.PassManager")
    mock_pm = MagicMock()
    mock_pm.run.return_value = "optimized_circuit"
    mock_pm_class.return_value = mock_pm
    return mock_pm


@pytest.fixture
def mock_pass_manager_failure(mocker):
    """Failed PassManager run procedure"""
    mock_pm_class = mocker.patch("qiskit_ibm_transpiler_mcp_server.qta.PassManager")
    mock_pm = MagicMock()
    mock_pm.run.side_effect = Exception("PassManager run failed")
    mock_pm_class.return_value = mock_pm
    return mock_pm


@pytest.fixture
def mock_ai_clifford_synthesis_success(mocker):
    """Successful AI Clifford synthesis procedure"""
    mock_clifford_synthesis_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AICliffordSynthesis"
    )
    mock_clifford_synthesis_instance = MagicMock()
    mock_clifford_synthesis_class.return_value = mock_clifford_synthesis_instance
    mock_clifford_synthesis_class.__name__ = "AICliffordSynthesis"
    return mock_clifford_synthesis_class


@pytest.fixture
def mock_ai_clifford_synthesis_failure(mocker):
    """Failed AI Clifford synthesis procedure"""
    mock_clifford_synthesis_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AICliffordSynthesis",
        side_effect=Exception("AI Clifford synthesis failed"),
    )
    mock_clifford_synthesis_instance = MagicMock()
    mock_clifford_synthesis_class.return_value = mock_clifford_synthesis_instance
    mock_clifford_synthesis_class.__name__ = "AICliffordSynthesis"
    return mock_clifford_synthesis_instance


@pytest.fixture
def mock_ai_linear_function_synthesis_success(mocker):
    """Successful AI Linear Function synthesis procedure"""
    mock_linear_function_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AILinearFunctionSynthesis"
    )
    mock_linear_function_instance = MagicMock()
    mock_linear_function_class.return_value = mock_linear_function_instance
    mock_linear_function_class.__name__ = "AILinearFunctionSynthesis"
    return mock_linear_function_class


@pytest.fixture
def mock_ai_linear_function_synthesis_failure(mocker):
    """Failed AI Linear Function synthesis procedure"""
    mock_linear_function_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AILinearFunctionSynthesis",
        side_effect=Exception("AI Linear Function synthesis failed"),
    )
    mock_linear_function_instance = MagicMock()
    mock_linear_function_class.return_value = mock_linear_function_instance
    mock_linear_function_class.__name__ = "AILinearFunctionSynthesis"
    return mock_linear_function_instance


@pytest.fixture
def mock_ai_permutation_synthesis_success(mocker):
    """Successful AI Permutation synthesis procedure"""
    mock_permutation_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPermutationSynthesis"
    )
    mock_permutation_instance = MagicMock()
    mock_permutation_class.return_value = mock_permutation_instance
    mock_permutation_class.__name__ = "AIPermutationSynthesis"
    return mock_permutation_class


@pytest.fixture
def mock_ai_permutation_synthesis_failure(mocker):
    """Failed AI Permutation synthesis procedure"""
    mock_permutation_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPermutationSynthesis",
        side_effect=Exception("Permutation synthesis failed"),
    )
    mock_permutation_instance = MagicMock()
    mock_permutation_class.return_value = mock_permutation_instance
    mock_permutation_class.__name__ = "AIPermutationSynthesis"
    return mock_permutation_instance


@pytest.fixture
def mock_ai_pauli_network_synthesis_success(mocker):
    """Successful AI Pauli Networks synthesis procedure"""
    mock_pauli_network_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPauliNetworkSynthesis"
    )
    mock_pauli_network_instance = MagicMock()
    mock_pauli_network_class.return_value = mock_pauli_network_instance
    mock_pauli_network_class.__name__ = "AIPauliNetworkSynthesis"
    return mock_pauli_network_class


@pytest.fixture
def mock_ai_pauli_network_synthesis_failure(mocker):
    """Failed AI Pauli Networks synthesis procedure"""
    mock_pauli_network_class = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.AIPauliNetworkSynthesis",
        side_effect=Exception("Pauli Networks synthesis failed"),
    )
    mock_pauli_network_instance = MagicMock()
    mock_pauli_network_class.return_value = mock_pauli_network_instance
    mock_pauli_network_class.__name__ = "AIPauliNetworkSynthesis"
    return mock_pauli_network_instance


@pytest.fixture
def mock_runtime_service():
    """Mock QiskitRuntimeService for testing."""
    mock_service = create_autospec(spec=QiskitRuntimeService)
    mock_service._channel = "ibm_quantum_platform"

    # Mock backends
    mock_backend1 = MagicMock()
    mock_backend1.name = "ibmq_qasm_simulator"
    mock_backend1.num_qubits = 32
    mock_backend1.simulator = True
    mock_backend1.status.return_value = MagicMock(
        operational=True, pending_jobs=0, status_msg="active"
    )

    mock_backend2 = MagicMock()
    mock_backend2.name = "ibm_brisbane"
    mock_backend2.num_qubits = 127
    mock_backend2.simulator = False
    mock_backend2.status.return_value = MagicMock(
        operational=True, pending_jobs=5, status_msg="active"
    )

    mock_service.backends.return_value = [mock_backend1, mock_backend2]
    mock_service.backend.return_value = mock_backend2

    # Mock jobs
    mock_job = MagicMock()
    mock_job.job_id.return_value = "job_123"
    mock_job.status.return_value = "DONE"
    mock_job.creation_date = "2024-01-01T10:00:00Z"
    mock_job.backend.return_value = mock_backend2
    mock_job.tags = ["test"]
    mock_job.error_message.return_value = None
    mock_job.cancel.return_value = None

    mock_service.jobs.return_value = [mock_job]
    mock_service.job.return_value = mock_job

    return mock_service


@pytest.fixture
def mock_env_vars(mocker):
    """Mock environment variables for testing."""
    env_mock = mocker.patch.dict(
        os.environ,
        {
            "QISKIT_IBM_TOKEN": "test_token_12345",
            "QISKIT_IBM_CHANNEL": "ibm_quantum_platform",
        },
    )
    yield env_mock


@pytest.fixture(autouse=True)
def reset_singleton():
    """Automatically reset singleton instance after each test"""
    QiskitRuntimeServiceProvider._instance = None
    yield
    QiskitRuntimeServiceProvider._instance = None


@pytest.fixture
def generate_ai_pass_manager_fixture(request):
    """Retrieve the specific fixture given the input request. Useful in parametrized tests"""
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_generate_ai_pass_manager_success(mocker):
    """Successful generate_ai_pass_manager procedure"""
    mock_pass_manager = MagicMock()
    mock_pass_manager.run.return_value = "transpiled_circuit"
    mock_func = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.generate_ai_pass_manager",
        return_value=mock_pass_manager,
    )
    return mock_func


@pytest.fixture
def mock_generate_ai_pass_manager_failure(mocker):
    """Failed generate_ai_pass_manager procedure"""
    mock_func = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.generate_ai_pass_manager",
        side_effect=Exception("Hybrid AI transpilation failed"),
    )
    return mock_func


@pytest.fixture
def mock_get_backend_service_with_coupling_map(mocker):
    """Successful get_backend_service with coupling_map attribute"""
    mock = mocker.patch(
        "qiskit_ibm_transpiler_mcp_server.qta.get_backend_service",
        new_callable=AsyncMock,
    )
    mock_backend = MagicMock()
    mock_backend.coupling_map = "mock_coupling_map"
    mock.return_value = {"backend": mock_backend, "status": "success"}
    return mock
