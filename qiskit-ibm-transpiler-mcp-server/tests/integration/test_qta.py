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
"""Integration tests for IBM Qiskit Transpiler MCP Server functions."""

from pathlib import Path

import pytest
from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_pauli_network_synthesis,
    ai_permutation_synthesis,
    ai_routing,
    hybrid_ai_transpile,
)

from tests.utils.helpers import validate_synthesis_result


# Get the path to the tests directory
TESTS_DIR = Path(__file__).parent.parent


class TestAIRouting:
    """Test AIRouting tool."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_success(self, backend_name):
        """
        Successful test AI routing tool with existing backend, quantum circuit and PassManager
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "success"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_failure_backend_name(
        self,
    ):
        """
        Failed test AI routing tool with existing backend, quantum circuit and PassManager. Here we simulate wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_empty_backend(self):
        """
        Failed test AI routing tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_routing(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI routing tool with existing backend, quantum circuit and PassManager. Here we simulate wrong input QASM string.
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_routing(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAICliffordSynthesis:
    """Test AI Clifford synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_success(self, backend_name):
        """
        Successful test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_empty_backend(self):
        """
        Failed test AI Clifford synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_clifford_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAILinearFunctionSynthesis:
    """Test AI Linear Function synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_success(self, backend_name):
        """
        Successful test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_empty_backend(self):
        """
        Failed test AI Linear Function synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_linear_function_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAIPermutationSynthesis:
    """Test AI Permutation synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_success(self, backend_name):
        """
        Successful test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_empty_backend(self):
        """
        Failed test AI Permutation synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_permutation_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


class TestAIPauliNetworkSynthesis:
    """Test AI Pauli Network synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_success(self, backend_name):
        """
        Successful test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        validate_synthesis_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_empty_backend(self):
        """
        Failed test AI Pauli Network synthesis tool with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await ai_pauli_network_synthesis(circuit=qasm_str, backend_name=backend_name)
        assert result["status"] == "error"


def validate_hybrid_transpile_result(result: dict) -> None:
    """Helper to validate hybrid transpilation result.

    Unlike synthesis passes, hybrid transpilation maps the circuit to the full backend,
    so the number of qubits may change (e.g., from 2 logical qubits to 133 physical qubits).
    """
    assert result["status"] == "success"
    assert isinstance(result["circuit_qpy"], str)
    assert len(result["circuit_qpy"]) > 0  # Non-empty QPY

    # Validate original_circuit metrics
    orig = result["original_circuit"]
    assert isinstance(orig, dict)
    assert "num_qubits" in orig and isinstance(orig["num_qubits"], int)
    assert "depth" in orig and isinstance(orig["depth"], int)
    assert "size" in orig and isinstance(orig["size"], int)
    assert "two_qubit_gates" in orig and isinstance(orig["two_qubit_gates"], int)
    assert orig["num_qubits"] > 0

    # Validate optimized_circuit metrics
    opt = result["optimized_circuit"]
    assert isinstance(opt, dict)
    assert "num_qubits" in opt and isinstance(opt["num_qubits"], int)
    assert "depth" in opt and isinstance(opt["depth"], int)
    assert "size" in opt and isinstance(opt["size"], int)
    assert "two_qubit_gates" in opt and isinstance(opt["two_qubit_gates"], int)
    # Note: num_qubits may change because hybrid transpilation maps to physical backend

    # Validate improvements
    imp = result["improvements"]
    assert isinstance(imp, dict)
    assert "depth_reduction" in imp and isinstance(imp["depth_reduction"], int)
    assert "two_qubit_gate_reduction" in imp and isinstance(imp["two_qubit_gate_reduction"], int)
    # Verify improvement calculation is correct
    assert imp["depth_reduction"] == orig["depth"] - opt["depth"]
    assert imp["two_qubit_gate_reduction"] == orig["two_qubit_gates"] - opt["two_qubit_gates"]


class TestHybridAITranspile:
    """Test hybrid AI transpilation tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="generate_ai_pass_manager may fail with Layout error on some backends/circuits",
        strict=False,
    )
    async def test_hybrid_ai_transpile_success(self, backend_name):
        """
        Successful test hybrid AI transpilation with existing backend and quantum circuit.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await hybrid_ai_transpile(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        validate_hybrid_transpile_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="generate_ai_pass_manager may fail with Layout error on some backends/circuits",
        strict=False,
    )
    async def test_hybrid_ai_transpile_with_custom_params(self, backend_name):
        """
        Test hybrid AI transpilation with custom optimization levels and layout mode.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await hybrid_ai_transpile(
            circuit=qasm_str,
            backend_name=backend_name,
            ai_optimization_level=1,
            optimization_level=1,
            ai_layout_mode="optimize",
        )
        validate_hybrid_transpile_result(result)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_failure_backend_name(self):
        """
        Failed test hybrid AI transpilation with wrong backend name.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = await hybrid_ai_transpile(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_empty_backend(self):
        """
        Failed test hybrid AI transpilation with empty backend.
        """
        with open(TESTS_DIR / "qasm" / "correct_qasm_1") as f:
            qasm_str = f.read()

        result = await hybrid_ai_transpile(circuit=qasm_str, backend_name="")
        assert result["status"] == "error"
        assert result["message"] == "backend_name is required and cannot be empty"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_hybrid_ai_transpile_failure_wrong_qasm_str(self, backend_name):
        """
        Failed test hybrid AI transpilation with wrong QASM string.
        """
        with open(TESTS_DIR / "qasm" / "wrong_qasm_1") as f:
            qasm_str = f.read()

        result = await hybrid_ai_transpile(
            circuit=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
