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
"""Unit tests for IBM Qiskit Transpiler MCP Server functions."""

import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_transpiler_mcp_server.qta import (
    _get_circuit_metrics,
    _run_synthesis_pass,
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
    hybrid_ai_transpile,
)


class TestGetCircuitMetrics:
    """Test _get_circuit_metrics helper function."""

    def test_simple_circuit_metrics(self):
        """Test metrics extraction from a simple circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        metrics = _get_circuit_metrics(qc)

        assert metrics["num_qubits"] == 2
        assert metrics["depth"] == 2
        assert metrics["size"] == 2
        assert metrics["two_qubit_gates"] == 1

    def test_no_two_qubit_gates(self):
        """Test metrics for circuit with no two-qubit gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)

        metrics = _get_circuit_metrics(qc)

        assert metrics["num_qubits"] == 3
        assert metrics["depth"] == 1
        assert metrics["size"] == 3
        assert metrics["two_qubit_gates"] == 0

    def test_multiple_two_qubit_gates(self):
        """Test metrics for circuit with multiple two-qubit gates."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)

        metrics = _get_circuit_metrics(qc)

        assert metrics["num_qubits"] == 3
        assert metrics["two_qubit_gates"] == 3

    def test_empty_circuit(self):
        """Test metrics for empty circuit."""
        qc = QuantumCircuit(2)

        metrics = _get_circuit_metrics(qc)

        assert metrics["num_qubits"] == 2
        assert metrics["depth"] == 0
        assert metrics["size"] == 0
        assert metrics["two_qubit_gates"] == 0


class TestRunSynthesis:
    """Test run_synthesis function"""

    @pytest.mark.asyncio
    async def test_run_synthesis(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_synthesis_success,
    ):
        """
        Successful test run_synthesis tool with existing backend, quantum circuit and PassManager
        """
        ai_synthesis_pass_kwargs = {
            "optimization_level": 1,
            "layout_mode": "optimize",
            "optimization_preferences": None,
            "local_mode": True,
        }
        result = await _run_synthesis_pass(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            synthesis_pass_class=mock_ai_synthesis_success,
            pass_kwargs=ai_synthesis_pass_kwargs,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        # Check metrics are present
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        # Check original metrics
        assert result["original_circuit"]["num_qubits"] == 2
        assert result["original_circuit"]["depth"] == 10
        assert result["original_circuit"]["two_qubit_gates"] == 5
        # Check optimized metrics
        assert result["optimized_circuit"]["depth"] == 7
        assert result["optimized_circuit"]["two_qubit_gates"] == 3
        # Check improvements
        assert result["improvements"]["depth_reduction"] == 3
        assert result["improvements"]["two_qubit_gate_reduction"] == 2
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            "dummy_circuit_qasm", circuit_format="qasm3"
        )
        mock_ai_synthesis_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            optimization_level=1,
            layout_mode="optimize",
            optimization_preferences=None,
            local_mode=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, pass_manager_fixture, dumps_fixture, ai_synthesis_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_synthesis_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_failure",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_synthesis_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_failure",
                "mock_dumps_qasm_success",
                "mock_ai_synthesis_success",
                "PassManager run failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_failure",
                "mock_ai_synthesis_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_success",
                "mock_load_qasm_circuit_success",
                "mock_pass_manager_success",
                "mock_dumps_qasm_success",
                "mock_ai_synthesis_failure",
                "AI Synthesis failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "pass_manager_fixture",
            "dumps_fixture",
            "ai_synthesis_fixture",
        ],
    )
    async def test_ai_synthesis_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        pass_manager_fixture,
        ai_synthesis_fixture,
        dumps_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
    ):
        """
        Failed test run_synthesis function with existing backend, quantum circuit and PassManager
        """
        ai_synthesis_pass_kwargs = {
            "optimization_level": 1,
            "layout_mode": "optimize",
            "optimization_preferences": None,
            "local_mode": True,
        }
        result = await _run_synthesis_pass(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            synthesis_pass_class=ai_synthesis_fixture,
            pass_kwargs=ai_synthesis_pass_kwargs,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIRouting:
    """Test AIRouting tool."""

    @pytest.mark.asyncio
    async def test_ai_routing_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_routing_success,
    ):
        """
        Successful test AI routing tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_routing(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            "dummy_circuit_qasm", circuit_format="qasm3"
        )
        mock_ai_routing_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            optimization_level=1,
            layout_mode="optimize",
            optimization_preferences=None,
            local_mode=True,
        )

    @pytest.mark.asyncio
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
    async def test_ai_routing_failures_parametrized(
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
        Failed test AI routing tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_routing(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAICliffordSynthesis:
    """Test AI Clifford synthesis tool"""

    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_clifford_synthesis_success,
    ):
        """
        Successful test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager.
        """
        result = await ai_clifford_synthesis(circuit=mock_circuit_qasm, backend_name=mock_backend)

        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            "dummy_circuit_qasm", circuit_format="qasm3"
        )
        mock_ai_clifford_synthesis_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            replace_only_if_better=True,
            local_mode=True,
        )

    @pytest.mark.asyncio
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
    async def test_ai_clifford_synthesis_failures_parametrized(
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
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager.
        """
        result = await ai_clifford_synthesis(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAILinearFunctionSynthesis:
    """Test AI Linear Function synthesis tool"""

    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_linear_function_synthesis_success,
    ):
        """
        Successful test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_linear_function_synthesis(
            circuit=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            mock_circuit_qasm, circuit_format="qasm3"
        )
        mock_ai_linear_function_synthesis_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            replace_only_if_better=True,
            local_mode=True,
        )

    @pytest.mark.asyncio
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
    async def test_ai_linear_function_synthesis_failures_parametrized(
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
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_linear_function_synthesis(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPermutationSynthesis:
    """Test AI Permutation synthesis tool"""

    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_permutation_synthesis_success,
    ):
        """
        Successful test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager-
        """
        result = await ai_permutation_synthesis(
            circuit=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            mock_circuit_qasm, circuit_format="qasm3"
        )
        mock_ai_permutation_synthesis_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            replace_only_if_better=True,
            local_mode=True,
        )

    @pytest.mark.asyncio
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
    async def test_ai_permutation_synthesis_failures_parametrized(
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
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_permutation_synthesis(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestAIPauliNetworkSynthesis:
    """Test AI Pauli Network synthesis"""

    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_success,
        mock_pass_manager_success,
        mock_ai_pauli_network_synthesis_success,
    ):
        """
        Successful test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_pauli_network_synthesis(
            circuit=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(
            mock_circuit_qasm, circuit_format="qasm3"
        )
        mock_ai_pauli_network_synthesis_success.assert_called_once_with(
            backend=mock_get_backend_service_success.return_value["backend"],
            replace_only_if_better=True,
            local_mode=True,
        )

    @pytest.mark.asyncio
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
    async def test_ai_pauli_networks_synthesis_failures_parametrized(
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
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager
        """
        result = await ai_pauli_network_synthesis(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]


class TestHybridAITranspile:
    """Test hybrid AI transpilation tool"""

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_success(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_with_coupling_map,
        mock_generate_ai_pass_manager_success,
        mock_get_circuit_metrics,
    ):
        """
        Successful test hybrid AI transpilation with existing backend and quantum circuit.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )

        assert result["status"] == "success"
        assert result["circuit_qpy"] == "circuit_qpy"
        assert "original_circuit" in result
        assert "optimized_circuit" in result
        assert "improvements" in result
        mock_get_backend_service_with_coupling_map.assert_awaited_once_with(
            backend_name=mock_backend
        )
        mock_load_qasm_circuit_success.assert_called_once_with(
            mock_circuit_qasm, circuit_format="qasm3"
        )
        mock_generate_ai_pass_manager_success.assert_called_once_with(
            coupling_map="mock_coupling_map",
            ai_optimization_level=3,
            optimization_level=3,
            ai_layout_mode="optimize",
        )

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_with_custom_params(
        self,
        mock_circuit_qasm,
        mock_backend,
        mock_load_qasm_circuit_success,
        mock_dumps_qasm_success,
        mock_get_backend_service_with_coupling_map,
        mock_generate_ai_pass_manager_success,
        mock_get_circuit_metrics,
    ):
        """
        Test hybrid AI transpilation with custom optimization levels and layout mode.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            ai_optimization_level=1,
            optimization_level=2,
            ai_layout_mode="improve",
        )

        assert result["status"] == "success"
        mock_generate_ai_pass_manager_success.assert_called_once_with(
            coupling_map="mock_coupling_map",
            ai_optimization_level=1,
            optimization_level=2,
            ai_layout_mode="improve",
        )

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_empty_backend(self, mock_circuit_qasm):
        """
        Test hybrid AI transpilation with empty backend returns error.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name="",
        )
        assert result["status"] == "error"
        assert result["message"] == "backend_name is required and cannot be empty"

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_invalid_ai_optimization_level(
        self, mock_circuit_qasm, mock_backend
    ):
        """
        Test hybrid AI transpilation with invalid ai_optimization_level returns error.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            ai_optimization_level=5,
        )
        assert result["status"] == "error"
        assert "ai_optimization_level must be 1, 2, or 3" in result["message"]

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_invalid_optimization_level(
        self, mock_circuit_qasm, mock_backend
    ):
        """
        Test hybrid AI transpilation with invalid optimization_level returns error.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            optimization_level=0,
        )
        assert result["status"] == "error"
        assert "optimization_level must be 1, 2, or 3" in result["message"]

    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_invalid_layout_mode(self, mock_circuit_qasm, mock_backend):
        """
        Test hybrid AI transpilation with invalid ai_layout_mode returns error.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
            ai_layout_mode="invalid",
        )
        assert result["status"] == "error"
        assert "ai_layout_mode must be one of" in result["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "get_backend_fixture, load_qasm_fixture, dumps_fixture, generate_ai_pass_manager_fixture, expected_message",
        [
            (
                "mock_get_backend_service_failure",
                "mock_load_qasm_circuit_success",
                "mock_dumps_qasm_success",
                "mock_generate_ai_pass_manager_success",
                "get_backend failed",
            ),
            (
                "mock_get_backend_service_with_coupling_map",
                "mock_load_qasm_circuit_failure",
                "mock_dumps_qasm_success",
                "mock_generate_ai_pass_manager_success",
                "Error in loading QuantumCircuit",
            ),
            (
                "mock_get_backend_service_with_coupling_map",
                "mock_load_qasm_circuit_success",
                "mock_dumps_qasm_failure",
                "mock_generate_ai_pass_manager_success",
                "Circuit dump failed",
            ),
            (
                "mock_get_backend_service_with_coupling_map",
                "mock_load_qasm_circuit_success",
                "mock_dumps_qasm_success",
                "mock_generate_ai_pass_manager_failure",
                "Hybrid AI transpilation failed",
            ),
        ],
        indirect=[
            "get_backend_fixture",
            "load_qasm_fixture",
            "dumps_fixture",
            "generate_ai_pass_manager_fixture",
        ],
    )
    async def test_hybrid_ai_transpile_failures_parametrized(
        self,
        get_backend_fixture,
        load_qasm_fixture,
        dumps_fixture,
        generate_ai_pass_manager_fixture,
        expected_message,
        mock_circuit_qasm,
        mock_backend,
        mock_get_circuit_metrics,
    ):
        """
        Test hybrid AI transpilation failures with various error scenarios.
        """
        result = await hybrid_ai_transpile(
            circuit=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]
