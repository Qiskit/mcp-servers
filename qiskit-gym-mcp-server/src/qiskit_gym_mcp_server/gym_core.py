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

"""Core gym environment creation functions for qiskit-gym MCP server.

This module provides functions to create and manage RL environments:
- PermutationGym: Learn optimal qubit routing with SWAP gates
- LinearFunctionGym: Learn optimal CNOT synthesis for linear functions
- CliffordGym: Learn optimal Clifford circuit synthesis
"""

import logging
from typing import Any

from qiskit.transpiler import CouplingMap

from qiskit_gym_mcp_server.constants import QISKIT_GYM_MAX_QUBITS
from qiskit_gym_mcp_server.coupling_maps import (
    create_coupling_map_from_edges,
    create_coupling_map_from_preset,
)
from qiskit_gym_mcp_server.state import GymStateProvider
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)


def _validate_qubit_count(num_qubits: int) -> dict[str, Any] | None:
    """Validate qubit count is within limits. Returns error dict or None."""
    if num_qubits > QISKIT_GYM_MAX_QUBITS:
        return {
            "status": "error",
            "message": f"Number of qubits ({num_qubits}) exceeds maximum ({QISKIT_GYM_MAX_QUBITS}). "
            f"Set QISKIT_GYM_MAX_QUBITS environment variable to increase limit.",
        }
    if num_qubits < 2:
        return {
            "status": "error",
            "message": "At least 2 qubits are required for an environment.",
        }
    return None


def _resolve_coupling_map(
    coupling_map: list[list[int]] | None,
    preset: str | None,
) -> tuple[CouplingMap, list[list[int]], int]:
    """Resolve coupling map from preset or custom edges.

    Returns:
        Tuple of (CouplingMap, edges, num_qubits)

    Raises:
        ValueError: If neither or both are provided, or invalid preset
    """
    if coupling_map is not None and preset is not None:
        raise ValueError("Provide either 'coupling_map' or 'preset', not both")

    if coupling_map is None and preset is None:
        raise ValueError("Must provide either 'coupling_map' or 'preset' parameter")

    if preset is not None:
        return create_coupling_map_from_preset(preset)
    else:
        # coupling_map is guaranteed to be not None here due to earlier validation
        assert coupling_map is not None
        return create_coupling_map_from_edges(coupling_map, bidirectional=True)


# ============================================================================
# Permutation Environment
# ============================================================================


@with_sync
async def create_permutation_environment(
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    """Create a PermutationGym environment for learning qubit routing.

    The PermutationGym environment learns to implement arbitrary qubit
    permutations using minimal SWAP gates on the given coupling map.

    Args:
        coupling_map: Custom coupling map as list of [qubit1, qubit2] edges
        preset: Hardware preset name (e.g., "ibm_nighthawk", "grid_3x3")

    Returns:
        Dict with env_id and environment info on success
    """
    try:
        # Import qiskit_gym here to handle import errors gracefully
        from qiskit_gym.envs import PermutationGym

        # Resolve coupling map
        cmap, edges, num_qubits = _resolve_coupling_map(coupling_map, preset)

        # Validate qubit count
        error = _validate_qubit_count(num_qubits)
        if error:
            return error

        # Create environment
        env = PermutationGym.from_coupling_map(cmap)

        # Get environment info (cast to int for JSON serialization)
        action_space_size = int(env.action_space.n) if hasattr(env.action_space, "n") else 0

        # Register in state
        state = GymStateProvider()
        config = {
            "preset": preset,
            "custom_edges": coupling_map is not None,
        }
        env_id = state.register_environment(
            gym_instance=env,
            env_type="permutation",
            config=config,
            coupling_map_edges=edges,
            num_qubits=num_qubits,
        )

        return {
            "status": "success",
            "env_id": env_id,
            "env_type": "permutation",
            "num_qubits": num_qubits,
            "num_edges": len(edges),
            "action_space_size": action_space_size,
            "description": "RL environment for learning optimal SWAP-based qubit routing",
            "usage": f"Use start_training with env_id='{env_id}' to train a model",
        }

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed. Install with: pip install qiskit-gym",
        }
    except Exception as e:
        logger.error(f"Failed to create permutation environment: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Linear Function Environment
# ============================================================================


@with_sync
async def create_linear_function_environment(
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
    basis_gates: list[str] | None = None,
) -> dict[str, Any]:
    """Create a LinearFunctionGym environment for learning CNOT synthesis.

    The LinearFunctionGym environment learns to decompose linear Boolean
    functions into efficient quantum circuits using CNOT gates.

    Args:
        coupling_map: Custom coupling map as list of [qubit1, qubit2] edges
        preset: Hardware preset name (e.g., "ibm_nighthawk", "grid_3x3")
        basis_gates: Optional list of basis gates (default uses CNOT)

    Returns:
        Dict with env_id and environment info on success
    """
    try:
        from qiskit_gym.envs import LinearFunctionGym

        # Resolve coupling map
        cmap, edges, num_qubits = _resolve_coupling_map(coupling_map, preset)

        # Validate qubit count
        error = _validate_qubit_count(num_qubits)
        if error:
            return error

        # Create environment
        if basis_gates:
            env = LinearFunctionGym.from_coupling_map(cmap, basis_gates=basis_gates)
        else:
            env = LinearFunctionGym.from_coupling_map(cmap)

        # Get environment info (cast to int for JSON serialization)
        action_space_size = int(env.action_space.n) if hasattr(env.action_space, "n") else 0

        # Register in state
        state = GymStateProvider()
        config = {
            "preset": preset,
            "custom_edges": coupling_map is not None,
            "basis_gates": basis_gates,
        }
        env_id = state.register_environment(
            gym_instance=env,
            env_type="linear_function",
            config=config,
            coupling_map_edges=edges,
            num_qubits=num_qubits,
        )

        return {
            "status": "success",
            "env_id": env_id,
            "env_type": "linear_function",
            "num_qubits": num_qubits,
            "num_edges": len(edges),
            "action_space_size": action_space_size,
            "basis_gates": basis_gates or ["cx"],
            "description": "RL environment for learning optimal CNOT synthesis of linear functions",
            "usage": f"Use start_training with env_id='{env_id}' to train a model",
        }

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed. Install with: pip install qiskit-gym",
        }
    except Exception as e:
        logger.error(f"Failed to create linear function environment: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Clifford Environment
# ============================================================================


def _parse_gateset(
    gateset: list[str | dict[str, Any]] | None,
    num_qubits: int,
    coupling_map_edges: list[list[int]],
) -> list[tuple[str, list[int]]]:
    """Parse gateset specification into qiskit-gym format.

    Args:
        gateset: User-provided gateset specification
        num_qubits: Number of qubits
        coupling_map_edges: Coupling map edges for 2-qubit gates

    Returns:
        List of (gate_name, qubit_list) tuples
    """
    if gateset is None:
        # Default gateset: H on all qubits, CX on all edges
        gates = []
        for q in range(num_qubits):
            gates.append(("H", [q]))
            gates.append(("S", [q]))
        # Add CX for each edge (use unique edges)
        seen = set()
        for e in coupling_map_edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in seen:
                seen.add(key)
                gates.append(("CX", [e[0], e[1]]))
        return gates

    # Parse user-provided gateset
    # CliffordGym allowed gates: H, S, Sdg, SX, SXdg, CX, CZ, SWAP
    gates = []
    for item in gateset:
        if isinstance(item, str):
            # Single gate name - apply to all qubits or edges
            if item.upper() in ["H", "S", "SDG", "X", "Y", "Z", "SX", "SXDG", "RZ"]:
                for q in range(num_qubits):
                    gates.append((item.upper(), [q]))
            elif item.upper() in ["CX", "CZ", "CNOT", "SWAP"]:
                seen = set()
                for e in coupling_map_edges:
                    key = (min(e[0], e[1]), max(e[0], e[1]))
                    if key not in seen:
                        seen.add(key)
                        gates.append((item.upper(), [e[0], e[1]]))
        elif isinstance(item, dict):
            # Explicit gate specification
            gate_name = item.get("gate", item.get("name", ""))
            qubits = item.get("qubits", [])
            if gate_name and qubits:
                gates.append((gate_name.upper(), list(qubits)))

    return gates


@with_sync
async def create_clifford_environment(
    num_qubits: int,
    coupling_map: list[list[int]] | None = None,
    preset: str | None = None,
    gateset: list[str | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a CliffordGym environment for learning Clifford circuit synthesis.

    The CliffordGym environment learns to synthesize optimal implementations
    of Clifford group elements using a customizable gate set.

    Args:
        num_qubits: Number of qubits for the environment
        coupling_map: Custom coupling map as list of [qubit1, qubit2] edges
        preset: Hardware preset name (e.g., "ibm_nighthawk", "grid_3x3")
        gateset: Optional custom gate set. Can be:
            - List of gate names: ["H", "S", "CX"] (applied to all qubits/edges)
            - List of dicts: [{"gate": "H", "qubits": [0]}, {"gate": "CX", "qubits": [0, 1]}]

    Returns:
        Dict with env_id and environment info on success
    """
    try:
        from qiskit_gym.envs import CliffordGym

        # Validate qubit count
        error = _validate_qubit_count(num_qubits)
        if error:
            return error

        # Resolve coupling map (optional for CliffordGym)
        if coupling_map is not None or preset is not None:
            _, edges, cmap_qubits = _resolve_coupling_map(coupling_map, preset)
            if cmap_qubits != num_qubits:
                return {
                    "status": "error",
                    "message": f"Coupling map has {cmap_qubits} qubits but num_qubits={num_qubits}",
                }
        else:
            # Create fully connected edges for gateset if no coupling map
            edges = []
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    edges.append([i, j])
                    edges.append([j, i])

        # Parse gateset
        parsed_gateset = _parse_gateset(gateset, num_qubits, edges)

        # Create environment
        env = CliffordGym(num_qubits=num_qubits, gateset=parsed_gateset)

        # Get environment info (cast to int for JSON serialization)
        action_space_size = int(env.action_space.n) if hasattr(env.action_space, "n") else 0

        # Register in state
        state = GymStateProvider()
        config = {
            "preset": preset,
            "custom_edges": coupling_map is not None,
            "gateset": gateset,
            "parsed_gateset_size": len(parsed_gateset),
        }
        env_id = state.register_environment(
            gym_instance=env,
            env_type="clifford",
            config=config,
            coupling_map_edges=edges,
            num_qubits=num_qubits,
        )

        # Summarize gateset
        gate_names = list(set(g[0] for g in parsed_gateset))

        return {
            "status": "success",
            "env_id": env_id,
            "env_type": "clifford",
            "num_qubits": num_qubits,
            "num_edges": len(edges) // 2,  # Count unique edges
            "action_space_size": action_space_size,
            "gateset_size": len(parsed_gateset),
            "gate_types": gate_names,
            "description": "RL environment for learning optimal Clifford circuit synthesis",
            "usage": f"Use start_training with env_id='{env_id}' to train a model",
        }

    except ImportError as e:
        logger.error(f"qiskit-gym not installed: {e}")
        return {
            "status": "error",
            "message": "qiskit-gym package not installed. Install with: pip install qiskit-gym",
        }
    except Exception as e:
        logger.error(f"Failed to create Clifford environment: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Environment Management Functions
# ============================================================================


@with_sync
async def list_environments() -> dict[str, Any]:
    """List all active RL environments.

    Returns:
        Dict with list of environments and their info
    """
    state = GymStateProvider()
    environments = state.list_environments()

    return {
        "status": "success",
        "environments": environments,
        "total": len(environments),
    }


@with_sync
async def get_environment_info(env_id: str) -> dict[str, Any]:
    """Get detailed information about a specific environment.

    Args:
        env_id: Environment ID

    Returns:
        Dict with environment details
    """
    state = GymStateProvider()
    env = state.get_environment(env_id)

    if env is None:
        return {
            "status": "error",
            "message": f"Environment '{env_id}' not found. Use list_environments to see available environments.",
        }

    # Get additional info from gym instance (cast to int for JSON serialization)
    action_space_size = 0
    observation_shape = None
    try:
        if hasattr(env.gym_instance, "action_space"):
            action_space_size = int(getattr(env.gym_instance.action_space, "n", 0))
        if hasattr(env.gym_instance, "observation_space"):
            shape = getattr(env.gym_instance.observation_space, "shape", None)
            if shape is not None:
                observation_shape = tuple(int(x) for x in shape)
    except Exception:
        pass

    return {
        "status": "success",
        "env_id": env.env_id,
        "env_type": env.env_type,
        "num_qubits": env.num_qubits,
        "num_edges": len(env.coupling_map_edges),
        "action_space_size": action_space_size,
        "observation_shape": observation_shape,
        "config": env.config,
    }


@with_sync
async def delete_environment(env_id: str) -> dict[str, Any]:
    """Delete an environment by ID.

    Args:
        env_id: Environment ID to delete

    Returns:
        Dict with deletion status
    """
    state = GymStateProvider()
    deleted = state.delete_environment(env_id)

    if deleted:
        return {
            "status": "success",
            "message": f"Environment '{env_id}' deleted successfully",
        }
    else:
        return {
            "status": "error",
            "message": f"Environment '{env_id}' not found",
        }
