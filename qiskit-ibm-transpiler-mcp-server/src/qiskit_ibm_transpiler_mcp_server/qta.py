from typing import Any, Literal

from qiskit_ibm_transpiler_mcp_server.utils import (
    load_qasm_circuit,
    get_backend_service,
    with_sync,
)

from qiskit.transpiler import PassManager  # type: ignore[import-untyped]
from qiskit.qasm3 import dumps  # type: ignore[import-untyped]
from qiskit_ibm_transpiler.ai.routing import AIRouting  # type: ignore[import-untyped]
from qiskit_ibm_transpiler.ai.synthesis import AICliffordSynthesis  # type: ignore[import-untyped]
from qiskit_ibm_transpiler.ai.synthesis import AILinearFunctionSynthesis
from qiskit_ibm_transpiler.ai.synthesis import AIPermutationSynthesis
from qiskit_ibm_transpiler.ai.synthesis import AIPauliNetworkSynthesis

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@with_sync
async def ai_routing(
    circuit_qasm: str,
    backend_name: str,
    optimization_level: int = 1,
    layout_mode: str = "optimize",
    optimization_preferences: Literal[
        "n_cnots", "n_gates", "cnot_layers", "layers", "noise"
    ]
    | list[Literal["n_cnots", "n_gates", "cnot_layers", "layers", "noise"]]
    | None = None,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Route input quantum circuit. It inserts SWAP operations on a circuit to make two-qubits operations compatible with a given coupling map that restricts the pair of qubits on which operations can be applied.
    It should be used as an initial step before any other AI synthesis routine.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        optimization_level: The potential optimization level to apply during the transpilation process. Valid values are [1,2,3], where 1 is the least optimization (and fastest), and 3 the most optimization (and most time-intensive)
        layout_mode: specifies how to handle the layout selection. It can assume the following values:
            - keep: This respects the layout set by the previous transpiler passes. Typically used when the circuit must be run on specific qubits of the device. It often produces worse results because it has less room for optimization.
            - improve: It is useful when you have a good initial guess for the layout
            - optimize: This is the default mode. It works best for general circuits where you might not have good layout guesses. This mode ignores previous layout selections.
        optimization_preferences: indicates what you want to reduce through optimization: number of cnot gates (n_cnots), number of gates (n_gates), number of cnots layers (cnot_layers), number of layers (layers), and/or noise (noise)
        local_mode: determines where the AIRouting pass runs. If False, AIRouting runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info("AI Routing pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_routing_pass = PassManager(
            [
                AIRouting(
                    backend=backend_service,
                    optimization_level=optimization_level,
                    layout_mode=layout_mode,
                    optimization_preferences=optimization_preferences,
                    local_mode=local_mode,
                ),
            ]
        )
        loaded_quantum_circuit = load_qasm_circuit(circuit_qasm)
        if loaded_quantum_circuit["status"] == "success":
            ai_optimized_circuit = ai_routing_pass.run(
                loaded_quantum_circuit["circuit"]
            )
            return {
                "status": "success",
                "optimized_circuit_qasm": dumps(ai_optimized_circuit),
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"AI Routing failed: {e}")
        return {"status": "error", "message": f"{e}"}


@with_sync
async def ai_clifford_synthesis(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Clifford circuits (blocks of H, S, and CX gates). Currently, up to nine qubit blocks.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Clifford synthesis runs. If False, AI Clifford synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info("AI Clifford synthesis pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_optimize_cliffords = PassManager(
            [
                AICliffordSynthesis(
                    backend=backend_service,
                    replace_only_if_better=replace_only_if_better,
                    local_mode=local_mode,
                ),
            ]
        )

        loaded_quantum_circuit = load_qasm_circuit(circuit_qasm)
        if loaded_quantum_circuit["status"] == "success":
            ai_optimized_circuit = ai_optimize_cliffords.run(
                loaded_quantum_circuit["circuit"]
            )
            return {
                "status": "success",
                "optimized_circuit_qasm": dumps(ai_optimized_circuit),
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"AI Clifford synthesis pass failed: {e}")
        return {"status": "error", "message": f"{e}"}


@with_sync
async def ai_linear_function_synthesis(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Linear Function circuits (blocks of CX and SWAP gates). Currently, up to nine qubit blocks.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the Linear Function synthesis pass runs. If False, Linear Function synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info("AI Linear Function synthesis pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_optimize_linear_functions = PassManager(
            [
                AILinearFunctionSynthesis(
                    backend=backend_service,
                    replace_only_if_better=replace_only_if_better,
                    local_mode=local_mode,
                ),
            ]
        )
        loaded_quantum_circuit = load_qasm_circuit(circuit_qasm)
        if loaded_quantum_circuit["status"] == "success":
            ai_optimized_circuit = ai_optimize_linear_functions.run(
                loaded_quantum_circuit["circuit"]
            )
            return {
                "status": "success",
                "optimized_circuit_qasm": dumps(ai_optimized_circuit),
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"AI Linear Function synthesis pass failed: {e}")
        return {"status": "error", "message": f"{e}"}


@with_sync
async def ai_permutation_synthesis(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Permutation circuits (blocks of SWAP gates). Currently available for 65, 33, and 27 qubit blocks.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Permutation synthesis pass runs. If False, AI Permutation synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info("AI Permutation synthesis pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_optimize_permutations = PassManager(
            [
                AIPermutationSynthesis(
                    backend=backend_service,
                    replace_only_if_better=replace_only_if_better,
                    local_mode=local_mode,
                ),
            ]
        )
        loaded_quantum_circuit = load_qasm_circuit(circuit_qasm)
        if loaded_quantum_circuit["status"] == "success":
            ai_optimized_circuit = ai_optimize_permutations.run(
                loaded_quantum_circuit["circuit"]
            )
            return {
                "status": "success",
                "optimized_circuit_qasm": dumps(ai_optimized_circuit),
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"AI Permutations synthesis pass failed: {e}")
        return {"status": "error", "message": f"{e}"}


@with_sync
async def ai_pauli_network_synthesis(
    circuit_qasm: str,
    backend_name: str,
    replace_only_if_better: bool = True,
    local_mode: bool = True,
) -> dict[str, Any]:
    """
    Synthesis for Pauli Network circuits (blocks of H, S, SX, CX, RX, RY and RZ gates). Currently, up to six qubit blocks.

    Args:
        circuit_qasm: quantum circuit as QASM string to be synthesized.
        backend_name: Qiskit Runtime Service backend name on which to map the input circuit synthesis
        replace_only_if_better: By default, the synthesis will replace the original sub-circuit only if the synthesized sub-circuit improves the original (currently only checking CNOT count), but this can be forced to always replace the circuit by setting replace_only_if_better=False
        local_mode: determines where the AI Pauli Network synthesis pass runs. If False, AI Pauli Network synthesis runs remotely through the Qiskit Transpiler Service. If True, the package tries to run the pass in your local environment with a fallback to cloud mode if the required dependencies are not found
    """
    if not backend_name or not backend_name.strip():
        return {
            "status": "error",
            "message": "backend is required and cannot be empty",
        }
    try:
        logger.info("AI Pauli Network synthesis pass")
        backend_service_coroutine = await get_backend_service(backend_name=backend_name)
        if backend_service_coroutine["status"] == "success":
            backend_service = backend_service_coroutine["backend"]
        else:
            return {"status": "error", "message": backend_service_coroutine["message"]}
        ai_optimize_pauli_network = PassManager(
            [
                AIPauliNetworkSynthesis(
                    backend=backend_service,
                    replace_only_if_better=replace_only_if_better,
                    local_mode=local_mode,
                ),
            ]
        )
        loaded_quantum_circuit = load_qasm_circuit(circuit_qasm)
        if loaded_quantum_circuit["status"] == "success":
            ai_optimized_circuit = ai_optimize_pauli_network.run(
                loaded_quantum_circuit["circuit"]
            )
            return {
                "status": "success",
                "optimized_circuit_qasm": dumps(ai_optimized_circuit),
            }
        else:
            return {"status": "error", "message": loaded_quantum_circuit["message"]}
    except Exception as e:
        logger.error(f"AI Pauli Network synthesis pass failed: {e}")
        return {"status": "error", "message": f"{e}"}
