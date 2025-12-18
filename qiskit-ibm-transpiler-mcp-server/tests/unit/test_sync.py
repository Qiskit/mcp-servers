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
import pytest
from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
)
from qiskit_ibm_transpiler_mcp_server.utils import setup_ibm_quantum_account


class TestWithSyncDecorator:
    """Test that async functions have .sync attribute."""

    def test_ai_routing_has_sync(self):
        """Test ai_routing has .sync attribute."""
        assert hasattr(ai_routing, "sync")
        assert callable(ai_routing.sync)

    def test_ai_clifford_synthesis_has_sync(self):
        """Test ai_clifford_synthesis has .sync attribute."""
        assert hasattr(ai_clifford_synthesis, "sync")
        assert callable(ai_clifford_synthesis.sync)

    def test_ai_linear_function_synthesis_has_sync(self):
        """Test ai_linear_function_synthesis has .sync attribute."""
        assert hasattr(ai_linear_function_synthesis, "sync")
        assert callable(ai_linear_function_synthesis.sync)

    def test_ai_permutation_synthesis_has_sync(self):
        """Test ai_permutation_synthesis has .sync attribute."""
        assert hasattr(ai_permutation_synthesis, "sync")
        assert callable(ai_permutation_synthesis.sync)

    def test_pauli_network_has_sync(self):
        """Test pauli_network_synthesis has .sync attribute."""
        assert hasattr(ai_pauli_network_synthesis, "sync")
        assert callable(ai_pauli_network_synthesis.sync)

    def test_setup_ibm_account_has_sync(self):
        """Test setup_ibm_quantum_account has .sync attribute."""
        assert hasattr(setup_ibm_quantum_account, "sync")
        assert callable(setup_ibm_quantum_account.sync)


class TestAIRoutingSync:
    """Test AIRouting sync tool."""

    def test_ai_routing_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI routing sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "circuit_qpy": "circuit_qpy",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_mcp_server.utils._run_async",
            return_value=mock_response,
        )
        result = ai_routing.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_routing_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_routing_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_routing_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_routing_failure",
                "AIRouting failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_routing_fixture",
        ],
    )
    def test_ai_routing_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        ai_routing_fixture,
        dumps_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI routing sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_routing.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAICliffordSync:
    """Test AI Clifford synthesis sync tool."""

    def test_ai_clifford_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Clifford synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "circuit_qpy": "circuit_qpy",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_mcp_server.utils._run_async",
            return_value=mock_response,
        )
        result = ai_clifford_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_clifford_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_clifford_synthesis_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_clifford_synthesis_failure",
                "AI Clifford synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_clifford_synthesis_fixture",
        ],
    )
    def test_ai_clifford_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_clifford_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Clifford synthesis sync tool with existing backend, quantum circuit and PassManager.
        """
        result = ai_clifford_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAILinearFunctionSync:
    """Test AI Linear Function synthesis sync tool."""

    def test_ai_linear_function_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Linear Function synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "circuit_qpy": "circuit_qpy",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_mcp_server.utils._run_async",
            return_value=mock_response,
        )
        result = ai_linear_function_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_linear_function_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_linear_function_synthesis_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_linear_function_synthesis_failure",
                "AI Linear Function synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_linear_function_synthesis_fixture",
        ],
    )
    def test_ai_linear_function_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_linear_function_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Linear Function synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_linear_function_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPermutationSync:
    """Test AI Permutation synthesis sync tool."""

    def test_ai_permutation_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Permutation synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "circuit_qpy": "circuit_qpy",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_mcp_server.utils._run_async",
            return_value=mock_response,
        )
        result = ai_permutation_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_permutation_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_permutation_synthesis_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_permutation_synthesis_failure",
                "Permutation synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_permutation_synthesis_fixture",
        ],
    )
    def test_ai_permutation_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_permutation_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Permutation synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_permutation_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPauliNetworkSync:
    """Test AI Pauli Network synthesis sync tool."""

    def test_ai_pauli_network_sync_success(self, mocker, mock_circuit_qasm, mock_backend):
        """
        Successful test AI Pauli Network synthesis sync tool with mocked backend, QASM quantum circuit and PassManager
        """
        mock_response = {
            "status": "success",
            "circuit_qpy": "circuit_qpy",
        }
        mocker_run_sync = mocker.patch(
            "qiskit_mcp_server.utils._run_async",
            return_value=mock_response,
        )
        result = ai_pauli_network_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        mocker_run_sync.assert_called_once()

    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_pauli_networks_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_pauli_network_synthesis_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_pauli_network_synthesis_failure",
                "Pauli Networks synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_pauli_networks_synthesis_fixture",
        ],
    )
    def test_ai_pauli_networks_synthesis_sync_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        dumps_fixture,
        ai_pauli_networks_synthesis_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test AI Pauli Network synthesis sync tool with existing backend, quantum circuit and PassManager
        """
        result = ai_pauli_network_synthesis.sync(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestSetupIBMQuantumAccountSync:
    """Test setup_ibm_quantum_account_sync function."""

    def test_setup_account_sync_success(self, mocker):
        """Test successful account setup with sync wrapper."""
        mock_response = {
            "status": "success",
            "message": "IBM Quantum account set up successfully",
            "channel": "ibm_quantum_platform",
            "available_backends": 10,
        }
        run_async_mock = mocker.patch("qiskit_mcp_server.utils._run_async")
        run_async_mock.return_value = mock_response
        result = setup_ibm_quantum_account.sync("test_token")
        assert result["status"] == "success"
        assert result["available_backends"] == 10

    def test_setup_account_sync_empty_token_uses_saved_credentials(self, mocker):
        """Test that empty token falls back to saved credentials."""
        mock_response = {
            "status": "success",
            "message": "IBM Quantum account set up successfully",
            "channel": "ibm_quantum_platform",
            "available_backends": 5,
        }

        run_async_mock = mocker.patch("qiskit_mcp_server.utils._run_async")
        run_async_mock.return_value = mock_response
        result = setup_ibm_quantum_account.sync("")

        assert result["status"] == "success"
        assert result["available_backends"] == 5
