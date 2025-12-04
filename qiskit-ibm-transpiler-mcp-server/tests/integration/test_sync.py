from qiskit_ibm_transpiler_mcp_server.sync import (
    ai_routing_sync,
    ai_clifford_synthesis_sync,
    ai_linear_function_synthesis_sync,
    ai_pauli_network_synthesis_sync,
    ai_permutation_synthesis_sync,
)
from tests.utils.helpers import calculate_2q_count_and_depth_improvement
import pytest


class TestAIRoutingSync:
    """Test AI Routing sync method"""

    @pytest.mark.integration
    def test_ai_routing_sync_success(self):
        """
        Successful test AI routing sync tool.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"
        result = ai_routing_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "success"

    @pytest.mark.integration
    def test_ai_routing_sync_failure_backend_name(
        self,
    ):
        """
        Failed test AI routing sync tool. Here we simulate wrong backend name.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_routing_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_routing_sync_failure_wrong_qasm_str(
        self,
    ):
        """
        Failed test AI routing sync tool. Here we simulate wrong input QASM string.
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_routing_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAICliffordSync:
    """Test AI Clifford synthesis sync method"""

    @pytest.mark.integration
    def test_ai_clifford_sync_success(self):
        """
        Successful test AI Clifford synthesis sync tool.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_clifford_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
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
    def test_ai_clifford_sync_failure_backend_name(self):
        """
        Failed test AI Clifford synthesis sync tool. Here we simulate wrong backend name.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_clifford_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_clifford_sync_failure_wrong_qasm(self):
        """
        Failed test AI Clifford synthesis sync tool. Here we simulate wrong qasm string
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_clifford_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAILinearFunctionSync:
    """Test AI Linear Function synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_linear_function_sync_success(self):
        """
        Successful test AI Linear Function synthesis sync tool.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_linear_function_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
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
    def test_ai_linear_function_sync_failure_backend_name(self):
        """
        Failed test AI Linear Function synthesis sync tool. Here we simulate wrong backend name.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_linear_function_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_linear_function_sync_failure_wrong_qasm(self):
        """
        Failed test AI Linear Function synthesis sync tool. Here we simulate wrong qasm string
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_linear_function_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAIPermutationSync:
    """Test AI Permutation synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_permutation_sync_success(self):
        """
        Successful test AI Permutation synthesis sync tool.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_permutation_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
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
    def test_ai_permutation_sync_failure_backend_name(self):
        """
        Failed test AI Permutation synthesis sync tool. Here we simulate wrong backend name.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_permutation_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_permutation_sync_failure_wrong_qasm(self):
        """
        Failed test AI Permutation synthesis sync tool. Here we simulate wrong qasm string
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_permutation_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"


class TestAIPauliNetworkSync:
    """Test AI Pauli Network synthesis sync tool"""

    @pytest.mark.integration
    def test_ai_pauli_network_sync_success(self):
        """
        Successful test AI Pauli Network synthesis sync tool.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_pauli_network_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
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
    def test_ai_pauli_network_sync_failure_backend_name(self):
        """
        Failed test AI Pauli Network synthesis sync tool. Here we simulate wrong backend name.
        """
        with open("tests/qasm/correct_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_fake"

        result = ai_pauli_network_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
        assert "Failed to find backend ibm_fake" in result["message"]

    @pytest.mark.integration
    def test_ai_pauli_network_sync_failure_wrong_qasm(self):
        """
        Failed test AI Pauli Network synthesis sync tool. Here we simulate wrong qasm string
        """
        with open("tests/qasm/wrong_qasm_1", "r") as f:
            qasm_str = f.read()
        backend_name = "ibm_torino"

        result = ai_pauli_network_synthesis_sync(
            circuit_qasm=qasm_str,
            backend_name=backend_name,
        )
        assert result["status"] == "error"
