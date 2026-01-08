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
import logging
from typing import Any, Literal

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_ibm_transpiler import generate_ai_pass_manager
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit_ibm_transpiler.ai.synthesis import (
    AICliffordSynthesis,
    AILinearFunctionSynthesis,
    AIPauliNetworkSynthesis,
    AIPermutationSynthesis,
)

from qiskit_ibm_transpiler_mcp_server.utils import (
    CircuitFormat,
    dump_circuit,
    get_backend_service,
    load_circuit,
    with_sync,
)


logger = logging.getLogger(__name__)

# Allowed values for optimization parameters
ALLOWED_OPTIMIZATION_LEVELS = (1, 2, 3)
ALLOWED_LAYOUT_MODES = ("keep", "improve", "optimize")


def _validate_optimization_params(
    optimization_level: int | None = None,
    layout_mode: str | None = None,
    param_prefix: str = "",
) -> str | None:
    """Validate optimization level and layout mode parameters.

    Args:
        optimization_level: The optimization level to validate (1, 2, or 3).
        layout_mode: The layout mode to validate ("keep", "improve", or "optimize").
        param_prefix: Optional prefix for parameter name in error messages (e.g., "ai_").

    Returns:
        Error message string if validation fails, None if all validations pass.
    """
    if optimization_level is not None and optimization_level not in ALLOWED_OPTIMIZATION_LEVELS:
        param_name = f"{param_prefix}optimization_level"
        return f"{param_name} must be 1, 2, or 3, got {optimization_level}"
    if layout_mode is not None and layout_mode not in ALLOWED_LAYOUT_MODES:
        param_name = f"{param_prefix}layout_mode"
        return f"{param_name} must be one of {ALLOWED_LAYOUT_MODES}, got '{layout_mode}'"
    return None


def _get_circuit_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Extract metrics from a quantum circuit for reporting.

    Args:
        circuit: The quantum circuit to analyze.

    Returns:
        Dictionary with circuit metrics useful for LLM reporting.
    """
    # Count two-qubit gates (important for quantum error rates)
    two_qubit_gate_count = 0
    for instruction in circuit.data:
        if len(instruction.qubits) == 2:
            two_qubit_gate_count += 1

    return {
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "size": circuit.size(),
        "two_qubit_gates": two_qubit_gate_count,
    }


async def _run_synthesis_pass(
    circuit: str,
    backend_name: str,
    synthesis_pass_class: type,
    pass_kwargs: dict[str, Any],
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Helper function to run specific synthesis routine

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        synthesis_pass_class: the specific AI synthesis procedure to be executed
        pass_kwargs: args for the AI synthesis class (e.g., `optimization_preferences`, `layout_mode`, `local_mode`, ...)
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".

    Returns:
        Dictionary with QPY format of the optimized circuit (base64-encoded).
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info(f"{synthesis_pass_class.__name__} pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_synthesis_pass = PassManager(
            [synthesis_pass_class(backend=backend_service, **pass_kwargs)]
        )
        loaded_quantum_circuit = load_circuit(circuit, circuit_format=circuit_format)
        if loaded_quantum_circuit["status"] == "success":
            original_circuit = loaded_quantum_circuit["circuit"]
            original_metrics = _get_circuit_metrics(original_circuit)

            ai_optimized_circuit = ai_synthesis_pass.run(original_circuit)
            optimized_metrics = _get_circuit_metrics(ai_optimized_circuit)

            # Return QPY format (source of truth for precision and chaining)
            qpy_str = dump_circuit(ai_optimized_circuit, circuit_format="qpy")

            # Calculate improvements (positive = reduction/improvement)
            depth_reduction = original_metrics["depth"] - optimized_metrics["depth"]
            two_qubit_reduction = (
                original_metrics["two_qubit_gates"] - optimized_metrics["two_qubit_gates"]
            )

            return {
                "status": "success",
                "circuit_qpy": qpy_str,
                "original_circuit": original_metrics,
                "optimized_circuit": optimized_metrics,
                "improvements": {
                    "depth_reduction": depth_reduction,
                    "two_qubit_gate_reduction": two_qubit_reduction,
                },
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"{synthesis_pass_class.__name__} failed: {e}")
        return {"status": "error", "message": f"{e}"}


@with_sync
async def ai_routing(
    circuit: str,
    backend_name: str,
    optimization_level: Literal[1, 2, 3] = 1,
    layout_mode: Literal["keep", "improve", "optimize"] = "optimize",
    optimization_preferences: Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Route input quantum circuit. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied.
    It should be used as an initial step before any other AI synthesis routine.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        optimization_level: The potential optimization level to apply during the transpilation process. Valid values are [1,2,3], where 1 is the least optimization (and fastest), and 3 the most optimization (and most time-intensive)
        layout_mode: specifies how to handle the layout selection. It can assume the following values:
            - keep: This respects the layout set by the previous transpiler passes. Typically used when the circuit must be run on specific qubits of the device. It often produces worse results because it has less room for optimization.
            - improve: It is useful when you have a good initial guess for the layout
            - optimize: This is the default mode. It works best for general circuits where you might not have good layout guesses. This mode ignores previous layout selections.
        optimization_preferences: indicates what you want to reduce through optimization: number of cnot gates (n_cnots), number of gates (n_gates), number of cnots layers (cnot_layers), number of layers (layers), and/or noise (noise)
        local_mode: determines where the AIRouting pass runs. If False, AIRouting runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
    """
    # Validate parameters
    validation_error = _validate_optimization_params(
        optimization_level=optimization_level, layout_mode=layout_mode
    )
    if validation_error:
        return {"status": "error", "message": validation_error}

    ai_routing_pass_kwargs = {
        "optimization_level": optimization_level,
        "layout_mode": layout_mode,
        "optimization_preferences": optimization_preferences,
        "local_mode": local_mode,
    }
    ai_routing_result = await _run_synthesis_pass(
        circuit=circuit,
        backend_name=backend_name,
        synthesis_pass_class=AIRouting,
        pass_kwargs=ai_routing_pass_kwargs,
        circuit_format=circuit_format,
    )
    return ai_routing_result


@with_sync
async def ai_clifford_synthesis(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Synthesis for Clifford circuits (blocks of H, S, and CX gates). Currently, up to nine qubit blocks.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Clifford synthesis runs. If False, AI Clifford synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
    """
    ai_clifford_synthesis_pass_kwargs = {
        "replace_only_if_better": replace_only_if_better,
        "local_mode": local_mode,
    }
    ai_clifford_synthesis_result = await _run_synthesis_pass(
        circuit=circuit,
        backend_name=backend_name,
        synthesis_pass_class=AICliffordSynthesis,
        pass_kwargs=ai_clifford_synthesis_pass_kwargs,
        circuit_format=circuit_format,
    )
    return ai_clifford_synthesis_result


@with_sync
async def ai_linear_function_synthesis(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Synthesis for Linear Function circuits (blocks of CX and SWAP gates). Currently, up to nine qubit blocks.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the Linear Function synthesis pass runs. If False, Linear Function synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
    """
    ai_linear_function_synthesis_pass_kwargs = {
        "replace_only_if_better": replace_only_if_better,
        "local_mode": local_mode,
    }
    ai_linear_function_synthesis_result = await _run_synthesis_pass(
        circuit=circuit,
        backend_name=backend_name,
        synthesis_pass_class=AILinearFunctionSynthesis,
        pass_kwargs=ai_linear_function_synthesis_pass_kwargs,
        circuit_format=circuit_format,
    )
    return ai_linear_function_synthesis_result


@with_sync
async def ai_permutation_synthesis(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Synthesis for Permutation circuits (blocks of SWAP gates). Currently available for 65, 33, and 27 qubit blocks.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Permutation synthesis pass runs. If False, AI Permutation synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
    """
    ai_permutation_synthesis_pass_kwargs = {
        "replace_only_if_better": replace_only_if_better,
        "local_mode": local_mode,
    }
    ai_permutation_synthesis_result = await _run_synthesis_pass(
        circuit=circuit,
        backend_name=backend_name,
        synthesis_pass_class=AIPermutationSynthesis,
        pass_kwargs=ai_permutation_synthesis_pass_kwargs,
        circuit_format=circuit_format,
    )
    return ai_permutation_synthesis_result


@with_sync
async def ai_pauli_network_synthesis(
    circuit: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently, up to six qubit blocks.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Pauli Network synthesis pass runs. If False, AI Pauli Network synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".
    """
    ai_pauli_network_synthesis_pass_kwargs = {
        "replace_only_if_better": replace_only_if_better,
        "local_mode": local_mode,
    }
    ai_pauli_network_synthesis_result = await _run_synthesis_pass(
        circuit=circuit,
        backend_name=backend_name,
        synthesis_pass_class=AIPauliNetworkSynthesis,
        pass_kwargs=ai_pauli_network_synthesis_pass_kwargs,
        circuit_format=circuit_format,
    )
    return ai_pauli_network_synthesis_result


@with_sync
async def hybrid_ai_transpile(
    circuit: str,
    backend_name: str,
    ai_optimization_level: Literal[1, 2, 3] = 3,
    optimization_level: Literal[1, 2, 3] = 3,
    ai_layout_mode: Literal["keep", "improve", "optimize"] = "optimize",
    circuit_format: CircuitFormat = "qasm3",
) -> dict[str, Any]:
    """
    Transpile a quantum circuit using a hybrid pass manager that combines Qiskit's heuristic
    optimization with AI-powered transpiler passes.

    This function creates a unified transpilation pipeline that leverages both classical
    heuristic approaches and AI-based optimization for routing and synthesis.

    Args:
        circuit: quantum circuit as QASM 3.0 string or base64-encoded QPY.
        backend_name: Qiskit Runtime Service backend name (e.g., 'ibm_torino', 'ibm_fez')
        ai_optimization_level: Optimization level (1-3) for AI components. Higher values
            yield better results but require more computational resources.
        optimization_level: Optimization level (1-3) for heuristic components in the PassManager.
        ai_layout_mode: Specifies how the AI routing component handles layout selection:
            - 'keep': Respects the layout set by previous transpiler passes
            - 'improve': Uses prior layouts as starting points for optimization
            - 'optimize': Default; ignores previous layout selections for general circuits
        circuit_format: format of the input circuit ("qasm3" or "qpy"). Defaults to "qasm3".

    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - circuit_qpy: Base64-encoded QPY format (for chaining with other tools)
        - original_circuit: Metrics dict (num_qubits, depth, size, two_qubit_gates)
        - optimized_circuit: Metrics dict for the optimized circuit
        - improvements: Dict with depth_reduction and two_qubit_gate_reduction
    """
    # Validate parameters
    validation_error = None
    if not backend_name or not backend_name.strip():
        validation_error = "backend_name is required and cannot be empty"
    else:
        # Validate AI optimization level, then heuristic optimization level, then AI layout mode
        validation_error = (
            _validate_optimization_params(
                optimization_level=ai_optimization_level, param_prefix="ai_"
            )
            or _validate_optimization_params(optimization_level=optimization_level)
            or _validate_optimization_params(layout_mode=ai_layout_mode, param_prefix="ai_")
        )
    if validation_error:
        return {"status": "error", "message": validation_error}

    try:
        logger.info("Hybrid AI transpilation pass")

        # Get backend to extract coupling map
        backend_service_result = await get_backend_service(backend_name=backend_name)
        if backend_service_result["status"] != "success":
            return {"status": "error", "message": backend_service_result["message"]}
        backend_service = backend_service_result["backend"]

        # Load input circuit
        loaded_quantum_circuit = load_circuit(circuit, circuit_format=circuit_format)
        if loaded_quantum_circuit["status"] != "success":
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
        original_circuit = loaded_quantum_circuit["circuit"]

        # Create hybrid AI pass manager
        ai_pass_manager = generate_ai_pass_manager(
            coupling_map=backend_service.coupling_map,
            ai_optimization_level=ai_optimization_level,
            optimization_level=optimization_level,
            ai_layout_mode=ai_layout_mode,
        )

        # Run the hybrid transpilation
        original_metrics = _get_circuit_metrics(original_circuit)
        transpiled_circuit = ai_pass_manager.run(original_circuit)
        optimized_metrics = _get_circuit_metrics(transpiled_circuit)

        # Return QPY format (source of truth for precision and chaining)
        qpy_str = dump_circuit(transpiled_circuit, circuit_format="qpy")

        return {
            "status": "success",
            "circuit_qpy": qpy_str,
            "original_circuit": original_metrics,
            "optimized_circuit": optimized_metrics,
            "improvements": {
                "depth_reduction": original_metrics["depth"] - optimized_metrics["depth"],
                "two_qubit_gate_reduction": (
                    original_metrics["two_qubit_gates"] - optimized_metrics["two_qubit_gates"]
                ),
            },
        }
    except Exception as e:
        logger.error(f"Hybrid AI transpilation failed: {e}")
        return {"status": "error", "message": f"{e}"}
