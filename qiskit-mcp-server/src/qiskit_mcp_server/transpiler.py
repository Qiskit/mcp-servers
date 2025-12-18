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

"""Core transpilation functions for the Qiskit MCP server.

This module provides transpilation functionality using Qiskit's pass managers,
enabling AI assistants to transpile quantum circuits with various optimization
levels and custom configurations.
"""

import logging
import math
import os
from typing import Any

from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_mcp_server.circuit_serialization import (
    CircuitFormat,
    dump_circuit,
    load_circuit,
)
from qiskit_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)

# Circuit size limits to prevent resource exhaustion
# Can be overridden via environment variables
MAX_QUBITS = int(os.environ.get("QISKIT_MCP_MAX_QUBITS", "100"))
MAX_GATES = int(os.environ.get("QISKIT_MCP_MAX_GATES", "10000"))

# IBM Quantum basis gate sets
# Eagle r3 processors use ECR as the native 2-qubit gate
# Heron processors use CZ as the native 2-qubit gate
BASIS_GATE_SETS: dict[str, list[str]] = {
    "ibm_eagle": ["id", "rz", "sx", "x", "ecr", "reset"],  # Eagle r3 (127 qubits)
    "ibm_heron": ["id", "rz", "sx", "x", "cz", "reset"],  # Heron r1/r2 (133-156 qubits)
    "ibm_legacy": ["id", "rz", "sx", "x", "cx", "reset"],  # Older IBM systems
}

# Common coupling map topologies
COUPLING_MAP_TOPOLOGIES: dict[str, str] = {
    "linear": "Linear chain topology (qubit i connected to i+1)",
    "ring": "Ring topology (linear with wraparound)",
    "grid": "2D grid topology (roughly square)",
    "heavy_hex": "IBM heavy-hex topology (used by Eagle/Heron processors)",
    "full": "All-to-all connectivity (no routing needed)",
}


def _create_topology(topology: str, num_qubits: int) -> CouplingMap | None:
    """Create a CouplingMap for the given topology name.

    Uses Qiskit's built-in CouplingMap factory methods where available.
    """
    if topology == "linear":
        return CouplingMap.from_line(num_qubits)
    if topology == "ring":
        return CouplingMap.from_ring(num_qubits)
    if topology == "full":
        return None  # None means all-to-all in Qiskit
    if topology == "grid":
        # Create a roughly square grid
        rows = math.ceil(math.sqrt(num_qubits))
        cols = math.ceil(num_qubits / rows)
        return CouplingMap.from_grid(rows, cols)
    if topology == "heavy_hex":
        # Heavy-hex with distance d has approximately 5*d^2 - 2*d - 1 qubits
        d = 1
        while 5 * d * d - 2 * d - 1 < num_qubits and d < 20:
            d += 1
        logger.debug(f"Using heavy_hex with distance={d} for {num_qubits} qubits")
        return CouplingMap.from_heavy_hex(d)
    return None  # Unknown topology


def _parse_coupling_map(
    coupling_map: list[list[int]] | str | None, num_qubits: int
) -> CouplingMap | None:
    """Parse coupling map from various input formats.

    Args:
        coupling_map: Either a list of edges, a topology name, or None
        num_qubits: Number of qubits (used for topology generation)

    Returns:
        CouplingMap object or None for all-to-all connectivity

    Raises:
        ValueError: If topology name is unknown
    """
    if coupling_map is None:
        return None

    if isinstance(coupling_map, str):
        topology = coupling_map.lower()
        logger.debug(f"Generating {topology} topology for {num_qubits} qubits")

        if topology not in COUPLING_MAP_TOPOLOGIES:
            logger.warning(f"Unknown topology requested: {topology}")
            raise ValueError(
                f"Unknown topology: {topology}. Available: {list(COUPLING_MAP_TOPOLOGIES.keys())}"
            )

        return _create_topology(topology, num_qubits)

    # Assume it's a list of edges
    return CouplingMap(coupling_map)


def _circuit_to_dict(circuit: QuantumCircuit) -> dict[str, Any]:
    """Convert a QuantumCircuit to a dictionary representation.

    Exports the circuit in QPY format (source of truth for precision when chaining).

    Args:
        circuit: The quantum circuit to convert

    Returns:
        Dictionary with circuit information including QPY serialization
    """
    # Get operation counts
    op_counts: dict[str, int] = {}
    for instruction in circuit.data:
        op_name = instruction.operation.name
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

    # Calculate depth
    depth = circuit.depth()

    # Serialize circuit in QPY format (source of truth)
    qpy_str = dump_circuit(circuit, circuit_format="qpy")

    return {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "depth": depth,
        "size": circuit.size(),
        "width": circuit.width(),
        "operation_counts": op_counts,
        "total_operations": sum(op_counts.values()),
        "circuit_qpy": qpy_str,  # QPY format (use for chaining tools/servers)
    }


def _parse_circuit(circuit_input: str, circuit_format: CircuitFormat = "qasm3") -> QuantumCircuit:
    """Parse a circuit from QASM3, QPY, or QASM2 string.

    Attempts to load the circuit using the specified format first, then falls back
    to QASM2 if QASM3 parsing fails (for backwards compatibility).

    Args:
        circuit_input: Circuit as QASM3 string, base64-encoded QPY, or QASM2 string
        circuit_format: Expected format ("qasm3" or "qpy"). Defaults to "qasm3".

    Returns:
        QuantumCircuit object

    Raises:
        ValueError: If the circuit cannot be parsed
    """
    # Try to load using the shared circuit serialization library (QPY or QASM3)
    result = load_circuit(circuit_input, circuit_format=circuit_format)
    if result["status"] == "success":
        return result["circuit"]

    # If QASM3 failed, try QASM2 as fallback (for backwards compatibility)
    if circuit_format == "qasm3":
        logger.debug(f"QASM3 parsing failed: {result.get('message')}, trying QASM2")
        try:
            return qasm2.loads(circuit_input)
        except Exception as e:
            logger.debug(f"QASM2 parsing also failed: {e}")

    # If we get here, all parsing attempts failed
    raise ValueError(
        f"Could not parse circuit as {circuit_format.upper()}. "
        f"Error: {result.get('message', 'Unknown error')}"
    )


def _validate_circuit_size(circuit: QuantumCircuit) -> str | None:
    """Validate circuit is within size limits.

    Returns:
        Error message if invalid, None if valid
    """
    if circuit.num_qubits > MAX_QUBITS:
        return (
            f"Circuit has {circuit.num_qubits} qubits, exceeding maximum of {MAX_QUBITS}. "
            "Large circuits may cause performance issues."
        )
    if circuit.size() > MAX_GATES:
        return (
            f"Circuit has {circuit.size()} gates, exceeding maximum of {MAX_GATES}. "
            "Large circuits may cause performance issues."
        )
    return None


def _validate_initial_layout(initial_layout: list[int] | None, num_qubits: int) -> str | None:
    """Validate initial_layout parameter.

    Returns:
        Error message if invalid, None if valid
    """
    if initial_layout is None:
        return None

    if len(initial_layout) != num_qubits:
        return (
            f"initial_layout has {len(initial_layout)} entries but circuit has "
            f"{num_qubits} qubits. They must match."
        )

    # Check for duplicates
    if len(set(initial_layout)) != len(initial_layout):
        return "initial_layout contains duplicate qubit indices."

    # Check for negative values
    if any(q < 0 for q in initial_layout):
        return "initial_layout contains negative qubit indices."

    return None


def _validate_optimization_level(optimization_level: int) -> str | None:
    """Validate optimization level parameter.

    Returns:
        Error message if invalid, None if valid
    """
    if optimization_level not in [0, 1, 2, 3]:
        return f"Invalid optimization_level: {optimization_level}. Must be 0, 1, 2, or 3."
    return None


def _resolve_basis_gates(
    basis_gates: list[str] | str | None,
) -> tuple[list[str] | None, str | None]:
    """Resolve basis gates from preset name or pass through list.

    Returns:
        Tuple of (resolved_basis_gates, error_message)
    """
    if basis_gates is None:
        return None, None

    if isinstance(basis_gates, str):
        if basis_gates in BASIS_GATE_SETS:
            return BASIS_GATE_SETS[basis_gates], None
        return None, (
            f"Unknown basis gate set: {basis_gates}. Available: {list(BASIS_GATE_SETS.keys())}"
        )

    return basis_gates, None


@with_sync
async def transpile_circuit(
    circuit: str,
    optimization_level: int = 2,
    basis_gates: list[str] | str | None = None,
    coupling_map: list[list[int]] | str | None = None,
    initial_layout: list[int] | None = None,
    seed_transpiler: int | None = None,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """Transpile a quantum circuit using Qiskit's preset pass managers.

    This function takes a quantum circuit and transpiles it to match
    target hardware constraints while optimizing for depth and gate count.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
            For QASM2, set circuit_format="qasm3" (it will auto-detect and parse QASM2).
        optimization_level: Optimization level (0-3):
            - 0: No optimization, just maps to basis gates
            - 1: Light optimization (default mapping, simple optimizations)
            - 2: Medium optimization (noise-adaptive layout, more passes) [default]
            - 3: Heavy optimization (best results, longest compilation time)
        basis_gates: Target basis gates. Can be:
            - A list of gate names (e.g., ["cx", "id", "rz", "sx", "x"])
            - A preset name (e.g., "ibm_default", "ibm_heron", "ion_trap")
            - None for no basis gate restriction
        coupling_map: Qubit connectivity. Can be:
            - A list of [control, target] pairs (e.g., [[0, 1], [1, 2]])
            - A topology name ("linear", "ring", "grid", "full")
            - None for all-to-all connectivity
        initial_layout: Optional initial qubit layout as list of physical qubit indices
        seed_transpiler: Random seed for reproducibility
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
            When "qasm3" is specified, QASM2 is also accepted as a fallback.

    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - original_circuit: Information about the input circuit (includes circuit in same format)
        - transpiled_circuit: Information about the transpiled circuit
        - optimization_level: The optimization level used
        - basis_gates: The basis gates used
        - circuit_format: The format used for circuit serialization
        - improvements: Metrics showing optimization improvements
    """
    try:
        # Parse input circuit
        try:
            parsed_circuit = _parse_circuit(circuit, circuit_format=circuit_format)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        # Validate all inputs - collect first error found
        validation_error = (
            _validate_circuit_size(parsed_circuit)
            or _validate_optimization_level(optimization_level)
            or _validate_initial_layout(initial_layout, parsed_circuit.num_qubits)
        )
        if validation_error:
            return {"status": "error", "message": validation_error}

        # Resolve basis gates
        resolved_basis_gates, basis_error = _resolve_basis_gates(basis_gates)
        if basis_error:
            return {"status": "error", "message": basis_error}

        # Parse coupling map
        try:
            resolved_coupling_map = _parse_coupling_map(coupling_map, parsed_circuit.num_qubits)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        # Get original circuit info
        original_info = _circuit_to_dict(parsed_circuit)

        # Log warning for level 3 with large circuits
        if optimization_level == 3 and (
            parsed_circuit.num_qubits > 20 or parsed_circuit.size() > 500
        ):
            logger.warning(
                f"Using optimization level 3 on circuit with {parsed_circuit.num_qubits} qubits "
                f"and {parsed_circuit.size()} gates. This may take a long time."
            )

        # Generate preset pass manager
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            basis_gates=resolved_basis_gates,
            coupling_map=resolved_coupling_map,
            initial_layout=initial_layout,
            seed_transpiler=seed_transpiler,
        )

        # Run transpilation
        transpiled = pm.run(parsed_circuit)

        # Get transpiled circuit info
        transpiled_info = _circuit_to_dict(transpiled)

        # Calculate improvements
        depth_reduction = original_info["depth"] - transpiled_info["depth"]
        depth_reduction_pct = (
            (depth_reduction / original_info["depth"] * 100) if original_info["depth"] > 0 else 0
        )
        size_reduction = original_info["size"] - transpiled_info["size"]
        size_reduction_pct = (
            (size_reduction / original_info["size"] * 100) if original_info["size"] > 0 else 0
        )

        result: dict[str, Any] = {
            "status": "success",
            "original_circuit": original_info,
            "transpiled_circuit": transpiled_info,
            "optimization_level": optimization_level,
            "basis_gates": resolved_basis_gates,
            "coupling_map_type": coupling_map if isinstance(coupling_map, str) else "custom",
            "improvements": {
                "depth_reduction": depth_reduction,
                "depth_reduction_percent": round(depth_reduction_pct, 2),
                "size_reduction": size_reduction,
                "size_reduction_percent": round(size_reduction_pct, 2),
            },
        }

        # Add warning for level 3
        if optimization_level == 3:
            result["note"] = (
                "Optimization level 3 provides best results but is slower. "
                "Consider level 2 for faster transpilation with good quality."
            )

        return result

    except Exception as e:
        logger.error(f"Transpilation failed: {e}")
        return {"status": "error", "message": f"Transpilation failed: {e!s}"}


@with_sync
async def analyze_circuit(circuit: str, circuit_format: CircuitFormat = "qasm3") -> dict[str, Any]:
    """Analyze a quantum circuit without transpiling it.

    Provides detailed information about the circuit structure, gate counts,
    and other metrics useful for understanding circuit complexity.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".

    Returns:
        Dictionary containing circuit analysis including:
        - num_qubits: Number of quantum bits
        - num_clbits: Number of classical bits
        - depth: Circuit depth
        - size: Total number of operations
        - operation_counts: Count of each gate type
        - two_qubit_gates: Count of two-qubit gates (important for noise)
        - single_qubit_gates: Count of single-qubit gates
    """
    try:
        try:
            parsed_circuit = _parse_circuit(circuit, circuit_format=circuit_format)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        info = _circuit_to_dict(parsed_circuit)

        # Categorize gates
        two_qubit_gates = 0
        single_qubit_gates = 0
        multi_qubit_gates = 0

        for instruction in parsed_circuit.data:
            num_qubits = len(instruction.qubits)
            if num_qubits == 1:
                single_qubit_gates += 1
            elif num_qubits == 2:
                two_qubit_gates += 1
            else:
                multi_qubit_gates += 1

        return {
            "status": "success",
            "circuit_info": info,
            "gate_categories": {
                "single_qubit_gates": single_qubit_gates,
                "two_qubit_gates": two_qubit_gates,
                "multi_qubit_gates": multi_qubit_gates,
            },
            "notes": [
                "Two-qubit gates are typically the noisiest operations",
                "Circuit depth affects decoherence - lower is better",
                "Consider transpiling with optimization_level=2 or 3 for hardware execution",
            ],
        }

    except Exception as e:
        logger.error(f"Circuit analysis failed: {e}")
        return {"status": "error", "message": f"Circuit analysis failed: {e!s}"}


def _transpile_at_level(
    circuit: QuantumCircuit, level: int, original_info: dict[str, Any]
) -> dict[str, Any]:
    """Transpile circuit at a specific optimization level."""
    try:
        pm = generate_preset_pass_manager(optimization_level=level)
        transpiled = pm.run(circuit)
        transpiled_info = _circuit_to_dict(transpiled)

        return {
            "depth": transpiled_info["depth"],
            "size": transpiled_info["size"],
            "operation_counts": transpiled_info["operation_counts"],
            "depth_vs_original": original_info["depth"] - transpiled_info["depth"],
            "size_vs_original": original_info["size"] - transpiled_info["size"],
        }
    except Exception as exc:
        logger.warning(f"Transpilation at level {level} failed: {exc}")
        return {"error": str(exc)}


def _find_best_level(optimization_results: dict[str, Any]) -> int:
    """Find the best optimization level based on circuit depth."""
    valid_levels = [
        level
        for level in range(4)
        if f"level_{level}" in optimization_results
        and "depth" in optimization_results[f"level_{level}"]
    ]
    if not valid_levels:
        return 2  # Default to level 2

    return min(
        valid_levels,
        key=lambda level: optimization_results[f"level_{level}"]["depth"],
    )


@with_sync
async def compare_optimization_levels(
    circuit: str, circuit_format: CircuitFormat = "qasm3"
) -> dict[str, Any]:
    """Compare transpilation results across all optimization levels.

    Useful for understanding the trade-off between compilation time
    and circuit quality for a specific circuit.

    Args:
        circuit: Quantum circuit as QASM3 string, base64-encoded QPY, or QASM2 string.
        circuit_format: Format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".

    Returns:
        Dictionary comparing results from optimization levels 0-3
    """
    try:
        try:
            parsed_circuit = _parse_circuit(circuit, circuit_format=circuit_format)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        # Validate circuit size (more lenient for comparison)
        if parsed_circuit.num_qubits > MAX_QUBITS:
            return {
                "status": "error",
                "message": f"Circuit has {parsed_circuit.num_qubits} qubits, exceeding maximum of {MAX_QUBITS}.",
            }

        original_info = _circuit_to_dict(parsed_circuit)
        results: dict[str, Any] = {
            "status": "success",
            "original_circuit": original_info,
            "optimization_results": {},
        }

        for level in range(4):
            results["optimization_results"][f"level_{level}"] = _transpile_at_level(
                parsed_circuit, level, original_info
            )

        # Add recommendation
        best_level = _find_best_level(results["optimization_results"])
        results["recommendation"] = {
            "best_for_depth": f"level_{best_level}",
            "note": "Level 2 is recommended for most use cases (balanced compilation time and quality)",
        }

        return results

    except Exception as e:
        logger.error(f"Optimization comparison failed: {e}")
        return {"status": "error", "message": f"Optimization comparison failed: {e!s}"}


@with_sync
async def get_available_basis_gates() -> dict[str, Any]:
    """Get available preset basis gate sets.

    Returns information about predefined basis gate sets that can be
    used with the transpile_circuit function.

    Returns:
        Dictionary with available basis gate presets and their gates
    """
    return {
        "status": "success",
        "basis_gate_sets": {
            name: {
                "gates": gates,
                "description": _get_basis_gate_description(name),
            }
            for name, gates in BASIS_GATE_SETS.items()
        },
        "note": "You can also provide a custom list of gate names",
    }


def _get_basis_gate_description(name: str) -> str:
    """Get description for a basis gate set."""
    descriptions = {
        "ibm_eagle": "IBM Eagle r3 processors (127 qubits, uses ECR)",
        "ibm_heron": "IBM Heron processors (133-156 qubits, uses CZ)",
        "ibm_legacy": "Older IBM systems (uses CX)",
    }
    return descriptions.get(name, "Custom basis gate set")


@with_sync
async def get_available_topologies() -> dict[str, Any]:
    """Get available coupling map topologies.

    Returns information about predefined qubit connectivity topologies
    that can be used with the transpile_circuit function.

    Returns:
        Dictionary with available topology names and descriptions
    """
    return {
        "status": "success",
        "topologies": COUPLING_MAP_TOPOLOGIES,
        "note": "You can also provide a custom coupling map as a list of [control, target] pairs",
    }


@with_sync
async def get_transpiler_info() -> dict[str, Any]:
    """Get information about the Qiskit transpiler and available options.

    Returns:
        Dictionary with transpiler information and usage guidance
    """
    return {
        "status": "success",
        "transpiler_info": {
            "description": "The Qiskit transpiler converts quantum circuits to match "
            "hardware constraints while optimizing for performance.",
            "stages": [
                {"name": "init", "description": "Unroll and decompose multi-qubit gates"},
                {"name": "layout", "description": "Map virtual qubits to physical qubits"},
                {"name": "routing", "description": "Insert SWAP gates for connectivity"},
                {"name": "translation", "description": "Convert to target basis gates"},
                {"name": "optimization", "description": "Reduce gate count and depth"},
                {"name": "scheduling", "description": "Add timing/delay instructions"},
            ],
            "optimization_levels": {
                "0": "No optimization - only decomposition to basis gates (fastest)",
                "1": "Light optimization with default layout",
                "2": "Medium optimization with noise-aware layout (recommended)",
                "3": "Heavy optimization for best results (can be slow for large circuits)",
            },
            "limits": {
                "max_qubits": MAX_QUBITS,
                "max_gates": MAX_GATES,
            },
        },
        "usage_tips": [
            "Use optimization_level=2 for most cases (good balance of speed and quality)",
            "Use optimization_level=3 only when circuit quality is critical and circuit is small",
            "Use optimization_level=0 or 1 for quick iterations during development",
            "Specify basis_gates to match your target hardware",
            "Specify coupling_map for hardware-specific routing",
        ],
    }
