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
from qiskit_ibm_transpiler_mcp_server.qta import (
    _run_synthesis_pass,
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
)


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
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
            synthesis_pass_class=mock_ai_synthesis_success,
            pass_kwargs=ai_synthesis_pass_kwargs,
        )
        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with("dummy_circuit_qasm")
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
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
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with("dummy_circuit_qasm")
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
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
        result = await ai_clifford_synthesis(
            circuit_qasm=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with("dummy_circuit_qasm")
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
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
            circuit_qasm=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(mock_circuit_qasm)
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
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
            circuit_qasm=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(mock_circuit_qasm)
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
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
            circuit_qasm=mock_circuit_qasm, backend_name=mock_backend
        )

        assert result == {
            "status": "success",
            "optimized_circuit_qasm": "optimized_circuit",
        }
        mock_get_backend_service_success.assert_awaited_once_with(backend_name=mock_backend)
        mock_load_qasm_circuit_success.assert_called_once_with(mock_circuit_qasm)
        mock_pass_manager_success.run.assert_called_once_with("input_circuit")
        mock_dumps_qasm_success.assert_called_once_with(mock_pass_manager_success.run.return_value)
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
                "Error in loading QuantumCircuit from QASM3.0",
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
                "QASM dumps failed",
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
            circuit_qasm=mock_circuit_qasm,
            backend_name=mock_backend,
        )
        assert result["status"] == "error"
        assert expected_message in result["message"]
