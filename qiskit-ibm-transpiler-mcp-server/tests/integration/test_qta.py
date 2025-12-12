"""Integration tests for IBM Qiskit Transpiler MCP Server functions."""

from qiskit_ibm_transpiler_mcp_server.qta import (
    ai_routing,
    ai_clifford_synthesis,
    ai_linear_function_synthesis,
    ai_permutation_synthesis,
    ai_pauli_network_synthesis,
)

from tests.utils.helpers import calculate_2q_count_and_depth_improvement
import pytest


class TestAIRouting:
    """Test AIRouting tool."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_success(
        self,
    ):
        """
        Successful test AI routing tool with existing backend, quantum circuit and PassManager
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = await ai_routing(
            circuit_qasm=qasm_str,
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
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = await ai_routing(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_routing_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI routing tool with existing backend, quantum circuit and PassManager. Here we simulate wrong input QASM string.
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = await ai_routing(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAICliffordSynthesis:
    """Test AI Clifford synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_success(
        self,
    ):
        """
        Successful test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_clifford_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "success"

        improvements = calculate_2q_count_and_depth_improvement(
            circuit1_qasm=qasm_str, circuit2_qasm=result["optimized_circuit_qasm"]
        )
        assert improvements["improvement_2q_gates"] >= 0, (
            f"Optimization decreased 2q gates: Δ={improvements['improvement_2q_gates']}%"
        )
        assert improvements["improvement_2q_depth"] >= 0, (
            f"Optimization decreased 2q depth: Δ={improvements['improvement_2q_depth']}%"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_clifford_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_clifford_synthesis_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI Clifford synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_clifford_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"


class TestAILinearFunctionSynthesis:
    """Test AI Linear Function synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_success(
        self,
    ):
        """
        Successful test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_linear_function_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "success"

        improvements = calculate_2q_count_and_depth_improvement(
            circuit1_qasm=qasm_str, circuit2_qasm=result["optimized_circuit_qasm"]
        )
        assert improvements["improvement_2q_gates"] >= 0, (
            f"Optimization decreased 2q gates: Δ={improvements['improvement_2q_gates']}%"
        )
        assert improvements["improvement_2q_depth"] >= 0, (
            f"Optimization decreased 2q depth: Δ={improvements['improvement_2q_depth']}%"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_linear_function_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_linear_function_synthesis_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI Linear Function synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_linear_function_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"


class TestAIPermutationSynthesis:
    """Test AI Permutation synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_success(
        self,
    ):
        """
        Successful test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_permutation_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "success"
        improvements = calculate_2q_count_and_depth_improvement(
            circuit1_qasm=qasm_str, circuit2_qasm=result["optimized_circuit_qasm"]
        )
        assert improvements["improvement_2q_gates"] >= 0, (
            f"Optimization decreased 2q gates: Δ={improvements['improvement_2q_gates']}%"
        )
        assert improvements["improvement_2q_depth"] >= 0, (
            f"Optimization decreased 2q depth: Δ={improvements['improvement_2q_depth']}%"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_permutation_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_permutation_synthesis_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI Permutation synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_permutation_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"


class TestAIPauliNetworkSynthesis:
    """Test AI Pauli Network synthesis tool"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_success(
        self,
    ):
        """
        Successful test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager.
        """

        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_pauli_network_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "success"
        improvements = calculate_2q_count_and_depth_improvement(
            circuit1_qasm=qasm_str, circuit2_qasm=result["optimized_circuit_qasm"]
        )
        assert improvements["improvement_2q_gates"] >= 0, (
            f"Optimization decreased 2q gates: Δ={improvements['improvement_2q_gates']}%"
        )
        assert improvements["improvement_2q_depth"] >= 0, (
            f"Optimization decreased 2q depth: Δ={improvements['improvement_2q_depth']}%"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_backend_name(
        self,
    ):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong backend name
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"
        result = await ai_pauli_network_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_ai_pauli_network_synthesis_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI Pauli Network synthesis tool with existing backend, quantum circuit and PassManager. Wrong QASM str
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = await ai_pauli_network_synthesis(
            circuit_qasm=qasm_str, backend_name=backend_name
        )
        assert result["status"] == "error"
