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

"""Tests for the Qiskit MCP Server transpilation functions."""

import pytest
from qiskit_mcp_server.transpiler import (
    analyze_circuit,
    compare_optimization_levels,
    get_available_basis_gates,
    get_available_topologies,
    get_transpiler_info,
    transpile_circuit,
)


class TestTranspileCircuit:
    """Tests for the transpile_circuit function."""

    @pytest.mark.asyncio
    async def test_transpile_simple_circuit(self, simple_circuit_qasm: str) -> None:
        """Test transpiling a simple circuit."""
        result = await transpile_circuit(simple_circuit_qasm)

        assert result["status"] == "success"
        assert "original_circuit" in result
        assert "transpiled_circuit" in result
        assert "improvements" in result
        assert result["original_circuit"]["num_qubits"] == 2

    @pytest.mark.asyncio
    async def test_transpile_with_optimization_level_0(self, simple_circuit_qasm: str) -> None:
        """Test transpiling with optimization level 0."""
        result = await transpile_circuit(simple_circuit_qasm, optimization_level=0)

        assert result["status"] == "success"
        assert result["optimization_level"] == 0

    @pytest.mark.asyncio
    async def test_transpile_with_optimization_level_3(self, complex_circuit_qasm: str) -> None:
        """Test transpiling with optimization level 3."""
        result = await transpile_circuit(complex_circuit_qasm, optimization_level=3)

        assert result["status"] == "success"
        assert result["optimization_level"] == 3

    @pytest.mark.asyncio
    async def test_transpile_with_invalid_optimization_level(
        self, simple_circuit_qasm: str
    ) -> None:
        """Test that invalid optimization level returns error."""
        result = await transpile_circuit(simple_circuit_qasm, optimization_level=5)

        assert result["status"] == "error"
        assert "optimization_level" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_transpile_with_basis_gates_preset(self, simple_circuit_qasm: str) -> None:
        """Test transpiling with a preset basis gate set."""
        result = await transpile_circuit(simple_circuit_qasm, basis_gates="ibm_eagle")

        assert result["status"] == "success"
        assert result["basis_gates"] == ["id", "rz", "sx", "x", "ecr", "reset"]

    @pytest.mark.asyncio
    async def test_transpile_with_custom_basis_gates(
        self, simple_circuit_qasm: str, sample_basis_gates: list[str]
    ) -> None:
        """Test transpiling with custom basis gates."""
        result = await transpile_circuit(simple_circuit_qasm, basis_gates=sample_basis_gates)

        assert result["status"] == "success"
        assert result["basis_gates"] == sample_basis_gates

    @pytest.mark.asyncio
    async def test_transpile_with_invalid_basis_gates_preset(
        self, simple_circuit_qasm: str
    ) -> None:
        """Test that invalid basis gate preset returns error."""
        result = await transpile_circuit(simple_circuit_qasm, basis_gates="invalid_preset")

        assert result["status"] == "error"
        assert "basis gate" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_transpile_with_linear_topology(self, simple_circuit_qasm: str) -> None:
        """Test transpiling with linear topology."""
        result = await transpile_circuit(simple_circuit_qasm, coupling_map="linear")

        assert result["status"] == "success"
        assert result["coupling_map_type"] == "linear"

    @pytest.mark.asyncio
    async def test_transpile_with_ring_topology(self, ghz_state_qasm: str) -> None:
        """Test transpiling with ring topology."""
        result = await transpile_circuit(ghz_state_qasm, coupling_map="ring")

        assert result["status"] == "success"
        assert result["coupling_map_type"] == "ring"

    @pytest.mark.asyncio
    async def test_transpile_with_custom_coupling_map(
        self, simple_circuit_qasm: str, sample_coupling_map: list[list[int]]
    ) -> None:
        """Test transpiling with custom coupling map."""
        result = await transpile_circuit(simple_circuit_qasm, coupling_map=sample_coupling_map)

        assert result["status"] == "success"
        assert result["coupling_map_type"] == "custom"

    @pytest.mark.asyncio
    async def test_transpile_with_invalid_topology(self, simple_circuit_qasm: str) -> None:
        """Test that invalid topology returns error."""
        result = await transpile_circuit(simple_circuit_qasm, coupling_map="invalid_topology")

        assert result["status"] == "error"
        assert "topology" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_transpile_with_seed(self, complex_circuit_qasm: str) -> None:
        """Test that seed provides reproducible results."""
        result1 = await transpile_circuit(complex_circuit_qasm, seed_transpiler=42)
        result2 = await transpile_circuit(complex_circuit_qasm, seed_transpiler=42)

        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result1["transpiled_circuit"]["depth"] == result2["transpiled_circuit"]["depth"]

    @pytest.mark.asyncio
    async def test_transpile_invalid_qasm(self, invalid_qasm: str) -> None:
        """Test that invalid QASM returns error."""
        result = await transpile_circuit(invalid_qasm)

        assert result["status"] == "error"
        assert "parse" in result["message"].lower() or "valid" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_transpiled_circuit_has_qpy_output(self, simple_circuit_qasm: str) -> None:
        """Test that transpiled circuit includes QPY output."""
        result = await transpile_circuit(simple_circuit_qasm)

        assert result["status"] == "success"
        # QPY format should be present as 'circuit_qpy'
        assert result["transpiled_circuit"]["circuit_qpy"] is not None
        assert result["original_circuit"]["circuit_qpy"] is not None


class TestAnalyzeCircuit:
    """Tests for the analyze_circuit function."""

    @pytest.mark.asyncio
    async def test_analyze_simple_circuit(self, simple_circuit_qasm: str) -> None:
        """Test analyzing a simple circuit."""
        result = await analyze_circuit(simple_circuit_qasm)

        assert result["status"] == "success"
        assert "circuit_info" in result
        assert "gate_categories" in result
        assert result["circuit_info"]["num_qubits"] == 2

    @pytest.mark.asyncio
    async def test_analyze_circuit_gate_categorization(self, complex_circuit_qasm: str) -> None:
        """Test that gates are properly categorized."""
        result = await analyze_circuit(complex_circuit_qasm)

        assert result["status"] == "success"
        categories = result["gate_categories"]
        assert "single_qubit_gates" in categories
        assert "two_qubit_gates" in categories
        assert "multi_qubit_gates" in categories

    @pytest.mark.asyncio
    async def test_analyze_invalid_qasm(self, invalid_qasm: str) -> None:
        """Test analyzing invalid QASM."""
        result = await analyze_circuit(invalid_qasm)

        assert result["status"] == "error"


class TestCompareOptimizationLevels:
    """Tests for the compare_optimization_levels function."""

    @pytest.mark.asyncio
    async def test_compare_levels_simple_circuit(self, simple_circuit_qasm: str) -> None:
        """Test comparing optimization levels for a simple circuit."""
        result = await compare_optimization_levels(simple_circuit_qasm)

        assert result["status"] == "success"
        assert "original_circuit" in result
        assert "optimization_results" in result
        assert "recommendation" in result

        # Check all levels are present
        for level in range(4):
            assert f"level_{level}" in result["optimization_results"]

    @pytest.mark.asyncio
    async def test_compare_levels_invalid_qasm(self, invalid_qasm: str) -> None:
        """Test comparing levels with invalid QASM."""
        result = await compare_optimization_levels(invalid_qasm)

        assert result["status"] == "error"


class TestGetAvailableBasisGates:
    """Tests for the get_available_basis_gates function."""

    @pytest.mark.asyncio
    async def test_get_basis_gates(self) -> None:
        """Test getting available basis gate sets."""
        result = await get_available_basis_gates()

        assert result["status"] == "success"
        assert "basis_gate_sets" in result
        assert "ibm_eagle" in result["basis_gate_sets"]
        assert "ibm_heron" in result["basis_gate_sets"]
        assert "gates" in result["basis_gate_sets"]["ibm_eagle"]


class TestGetAvailableTopologies:
    """Tests for the get_available_topologies function."""

    @pytest.mark.asyncio
    async def test_get_topologies(self) -> None:
        """Test getting available topologies."""
        result = await get_available_topologies()

        assert result["status"] == "success"
        assert "topologies" in result
        assert "linear" in result["topologies"]
        assert "ring" in result["topologies"]
        assert "grid" in result["topologies"]
        assert "heavy_hex" in result["topologies"]


class TestGetTranspilerInfo:
    """Tests for the get_transpiler_info function."""

    @pytest.mark.asyncio
    async def test_get_transpiler_info(self) -> None:
        """Test getting transpiler information."""
        result = await get_transpiler_info()

        assert result["status"] == "success"
        assert "transpiler_info" in result
        assert "stages" in result["transpiler_info"]
        assert "optimization_levels" in result["transpiler_info"]
        assert "usage_tips" in result

    @pytest.mark.asyncio
    async def test_transpiler_info_includes_limits(self) -> None:
        """Test that transpiler info includes circuit size limits."""
        result = await get_transpiler_info()

        assert result["status"] == "success"
        assert "limits" in result["transpiler_info"]
        assert "max_qubits" in result["transpiler_info"]["limits"]
        assert "max_gates" in result["transpiler_info"]["limits"]


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_initial_layout_wrong_length(self, simple_circuit_qasm: str) -> None:
        """Test that wrong initial_layout length returns error."""
        # simple_circuit_qasm has 2 qubits, provide 3 layout values
        result = await transpile_circuit(simple_circuit_qasm, initial_layout=[0, 1, 2])

        assert result["status"] == "error"
        assert "initial_layout" in result["message"]
        assert "2 qubits" in result["message"]

    @pytest.mark.asyncio
    async def test_initial_layout_duplicate_qubits(self, simple_circuit_qasm: str) -> None:
        """Test that duplicate qubit indices in initial_layout returns error."""
        result = await transpile_circuit(simple_circuit_qasm, initial_layout=[0, 0])

        assert result["status"] == "error"
        assert "duplicate" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_initial_layout_negative_qubit(self, simple_circuit_qasm: str) -> None:
        """Test that negative qubit index in initial_layout returns error."""
        result = await transpile_circuit(simple_circuit_qasm, initial_layout=[-1, 1])

        assert result["status"] == "error"
        assert "negative" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_valid_initial_layout(self, simple_circuit_qasm: str) -> None:
        """Test that valid initial_layout works correctly."""
        result = await transpile_circuit(simple_circuit_qasm, initial_layout=[0, 1])

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_circuit_exceeds_max_qubits(self) -> None:
        """Test that circuits exceeding max qubits return error."""
        # Create a circuit with 101 qubits (exceeds MAX_QUBITS=100)
        large_circuit_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[101];
creg c[101];
h q[0];
"""
        result = await transpile_circuit(large_circuit_qasm)

        assert result["status"] == "error"
        assert "101 qubits" in result["message"]
        assert "100" in result["message"]

    @pytest.mark.asyncio
    async def test_optimization_level_3_includes_note(self, simple_circuit_qasm: str) -> None:
        """Test that optimization level 3 results include a performance note."""
        result = await transpile_circuit(simple_circuit_qasm, optimization_level=3)

        assert result["status"] == "success"
        assert "note" in result
        assert "level 3" in result["note"].lower() or "slower" in result["note"].lower()


class TestSyncInterface:
    """Tests for the synchronous interface (.sync attribute)."""

    def test_transpile_circuit_sync(self, simple_circuit_qasm: str) -> None:
        """Test synchronous transpile_circuit call."""
        result = transpile_circuit.sync(simple_circuit_qasm)

        assert result["status"] == "success"
        assert "transpiled_circuit" in result

    def test_analyze_circuit_sync(self, simple_circuit_qasm: str) -> None:
        """Test synchronous analyze_circuit call."""
        result = analyze_circuit.sync(simple_circuit_qasm)

        assert result["status"] == "success"
        assert "circuit_info" in result

    def test_get_basis_gates_sync(self) -> None:
        """Test synchronous get_available_basis_gates call."""
        result = get_available_basis_gates.sync()

        assert result["status"] == "success"
        assert "basis_gate_sets" in result


class TestCircuitFormat:
    """Tests for circuit format support (QASM3, QPY, QASM2 fallback)."""

    @pytest.mark.asyncio
    async def test_transpile_with_qpy_input(self, simple_circuit_qpy: str) -> None:
        """Test transpiling a QPY-encoded circuit returns QPY output."""
        result = await transpile_circuit(simple_circuit_qpy, circuit_format="qpy")

        assert result["status"] == "success"
        assert result["original_circuit"]["num_qubits"] == 2
        # QPY format returned as 'circuit_qpy'
        assert result["transpiled_circuit"]["circuit_qpy"] is not None

    @pytest.mark.asyncio
    async def test_analyze_with_qpy_input(self, simple_circuit_qpy: str) -> None:
        """Test analyzing a QPY-encoded circuit returns QPY output."""
        result = await analyze_circuit(simple_circuit_qpy, circuit_format="qpy")

        assert result["status"] == "success"
        assert result["circuit_info"]["num_qubits"] == 2
        assert result["circuit_info"]["circuit_qpy"] is not None

    @pytest.mark.asyncio
    async def test_compare_levels_with_qpy_input(self, simple_circuit_qpy: str) -> None:
        """Test comparing optimization levels with QPY-encoded circuit."""
        result = await compare_optimization_levels(simple_circuit_qpy, circuit_format="qpy")

        assert result["status"] == "success"
        for level in range(4):
            assert f"level_{level}" in result["optimization_results"]
        # Original circuit has QPY format
        assert result["original_circuit"]["circuit_qpy"] is not None

    @pytest.mark.asyncio
    async def test_qasm2_fallback(self, simple_circuit_qasm: str) -> None:
        """Test that QASM2 input is accepted when circuit_format is qasm3."""
        # simple_circuit_qasm is in QASM2 format
        result = await transpile_circuit(simple_circuit_qasm, circuit_format="qasm3")

        assert result["status"] == "success"
        assert result["original_circuit"]["num_qubits"] == 2

    def test_transpile_qpy_sync(self, simple_circuit_qpy: str) -> None:
        """Test synchronous transpile_circuit call with QPY input."""
        result = transpile_circuit.sync(simple_circuit_qpy, circuit_format="qpy")

        assert result["status"] == "success"
        # QPY format returned as 'circuit_qpy'
        assert result["transpiled_circuit"]["circuit_qpy"] is not None

    def test_analyze_qpy_sync(self, simple_circuit_qpy: str) -> None:
        """Test synchronous analyze_circuit call with QPY input."""
        result = analyze_circuit.sync(simple_circuit_qpy, circuit_format="qpy")

        assert result["status"] == "success"
        assert result["circuit_info"]["num_qubits"] == 2
