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

"""Coupling map management and hardware presets for qiskit-gym MCP server.

This module provides:
- Hardware presets for IBM Quantum backends (Heron, Nighthawk)
- Grid and linear topology generators
- Subtopology extraction for training on hardware subgraphs
- Custom coupling map creation and validation
"""

import itertools
import logging
from typing import Any

from qiskit.transpiler import CouplingMap

from qiskit_gym_mcp_server.constants import QISKIT_GYM_MAX_QUBITS
from qiskit_gym_mcp_server.utils import with_sync


logger = logging.getLogger(__name__)

# ============================================================================
# Hardware Presets
# ============================================================================

HARDWARE_PRESETS: dict[str, dict[str, Any]] = {
    # IBM Heron - Heavy-hex topology
    "ibm_heron_r1": {
        "description": "IBM Heron r1 processor (133 qubits, heavy-hex topology)",
        "num_qubits": 133,
        "topology": "heavy_hex",
        "basis_gates": ["cz", "id", "rz", "sx", "x"],
        "bidirectional": True,
    },
    "ibm_heron_r2": {
        "description": "IBM Heron r2 processor (156 qubits, heavy-hex topology)",
        "num_qubits": 156,
        "topology": "heavy_hex",
        "basis_gates": ["cz", "id", "rz", "sx", "x"],
        "bidirectional": True,
    },
    # IBM Nighthawk - 10x12 rectangular grid (120 qubits)
    # 600% circuit depth improvement vs Heron for Ising circuits
    "ibm_nighthawk": {
        "description": "IBM Nighthawk processor (120 qubits, 10x12 rectangular grid)",
        "num_qubits": 120,
        "topology": "grid",
        "rows": 10,
        "cols": 12,
        "basis_gates": ["cz", "id", "rz", "sx", "x"],
        "bidirectional": True,
    },
    # Generic grids
    "grid_2x2": {
        "description": "2x2 grid topology (4 qubits)",
        "num_qubits": 4,
        "topology": "grid",
        "rows": 2,
        "cols": 2,
        "bidirectional": True,
    },
    "grid_2x3": {
        "description": "2x3 grid topology (6 qubits)",
        "num_qubits": 6,
        "topology": "grid",
        "rows": 2,
        "cols": 3,
        "bidirectional": True,
    },
    "grid_3x3": {
        "description": "3x3 grid topology (9 qubits)",
        "num_qubits": 9,
        "topology": "grid",
        "rows": 3,
        "cols": 3,
        "bidirectional": True,
    },
    "grid_4x4": {
        "description": "4x4 grid topology (16 qubits)",
        "num_qubits": 16,
        "topology": "grid",
        "rows": 4,
        "cols": 4,
        "bidirectional": True,
    },
    "grid_5x5": {
        "description": "5x5 grid topology (25 qubits)",
        "num_qubits": 25,
        "topology": "grid",
        "rows": 5,
        "cols": 5,
        "bidirectional": True,
    },
    "grid_10x10": {
        "description": "10x10 grid topology (100 qubits)",
        "num_qubits": 100,
        "topology": "grid",
        "rows": 10,
        "cols": 10,
        "bidirectional": True,
    },
    # Linear chains
    "linear_3": {
        "description": "Linear chain (3 qubits)",
        "num_qubits": 3,
        "topology": "line",
        "bidirectional": True,
    },
    "linear_4": {
        "description": "Linear chain (4 qubits)",
        "num_qubits": 4,
        "topology": "line",
        "bidirectional": True,
    },
    "linear_5": {
        "description": "Linear chain (5 qubits)",
        "num_qubits": 5,
        "topology": "line",
        "bidirectional": True,
    },
    "linear_10": {
        "description": "Linear chain (10 qubits)",
        "num_qubits": 10,
        "topology": "line",
        "bidirectional": True,
    },
}


# ============================================================================
# Coupling Map Generation
# ============================================================================


def _generate_grid_edges(rows: int, cols: int, bidirectional: bool = True) -> list[list[int]]:
    """Generate coupling map edges for a 2D grid topology."""
    edges: list[list[int]] = []
    for r in range(rows):
        for c in range(cols):
            qubit = r * cols + c
            # Right neighbor
            if c < cols - 1:
                edges.append([qubit, qubit + 1])
                if bidirectional:
                    edges.append([qubit + 1, qubit])
            # Down neighbor
            if r < rows - 1:
                edges.append([qubit, qubit + cols])
                if bidirectional:
                    edges.append([qubit + cols, qubit])
    return edges


def _generate_line_edges(num_qubits: int, bidirectional: bool = True) -> list[list[int]]:
    """Generate coupling map edges for a linear chain topology."""
    edges: list[list[int]] = []
    for i in range(num_qubits - 1):
        edges.append([i, i + 1])
        if bidirectional:
            edges.append([i + 1, i])
    return edges


def _generate_heavy_hex_edges(num_qubits: int, bidirectional: bool = True) -> list[list[int]]:
    """Generate coupling map edges for heavy-hex topology.

    Heavy-hex is a hexagonal lattice with additional qubits at edge midpoints.
    This is a simplified approximation - for exact IBM topologies, use the
    actual backend coupling maps.
    """
    # For simplicity, we approximate heavy-hex as a rectangular grid with
    # some edges removed to create the heavy-hex pattern.
    # This is sufficient for RL training purposes.

    # Calculate grid dimensions that give approximately the right qubit count
    # Heavy-hex has roughly 60% of full grid connectivity
    import math

    side = int(math.sqrt(num_qubits * 1.67))  # Approximate
    rows = side
    cols = (num_qubits + rows - 1) // rows

    # Start with grid edges
    edges = _generate_grid_edges(rows, cols, bidirectional=False)

    # Filter edges to only include qubits within num_qubits range
    # (grid approximation may create more nodes than num_qubits)
    edges = [e for e in edges if e[0] < num_qubits and e[1] < num_qubits]

    # Make bidirectional if requested
    if bidirectional:
        reverse_edges = [[e[1], e[0]] for e in edges]
        edges.extend(reverse_edges)

    return edges


def generate_coupling_map_edges(
    topology: str,
    num_qubits: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    bidirectional: bool = True,
) -> list[list[int]]:
    """Generate coupling map edges for a given topology.

    Args:
        topology: Topology type ("grid", "line", "heavy_hex")
        num_qubits: Number of qubits (required for line and heavy_hex)
        rows: Number of rows (required for grid)
        cols: Number of columns (required for grid)
        bidirectional: Whether edges should be bidirectional

    Returns:
        List of [source, target] edge pairs
    """
    if topology == "grid":
        if rows is None or cols is None:
            raise ValueError("Grid topology requires rows and cols parameters")
        return _generate_grid_edges(rows, cols, bidirectional)
    elif topology == "line":
        if num_qubits is None:
            raise ValueError("Line topology requires num_qubits parameter")
        return _generate_line_edges(num_qubits, bidirectional)
    elif topology == "heavy_hex":
        if num_qubits is None:
            raise ValueError("Heavy-hex topology requires num_qubits parameter")
        return _generate_heavy_hex_edges(num_qubits, bidirectional)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def create_coupling_map_from_preset(preset: str) -> tuple[CouplingMap, list[list[int]], int]:
    """Create a CouplingMap from a hardware preset.

    Args:
        preset: Name of the hardware preset

    Returns:
        Tuple of (CouplingMap, edges, num_qubits)

    Raises:
        ValueError: If preset is not found
    """
    if preset not in HARDWARE_PRESETS:
        available = ", ".join(sorted(HARDWARE_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    config = HARDWARE_PRESETS[preset]
    topology = config["topology"]
    num_qubits = config["num_qubits"]
    bidirectional = config.get("bidirectional", True)

    if topology == "grid":
        edges = _generate_grid_edges(config["rows"], config["cols"], bidirectional)
    elif topology == "line":
        edges = _generate_line_edges(num_qubits, bidirectional)
    elif topology == "heavy_hex":
        edges = _generate_heavy_hex_edges(num_qubits, bidirectional)
    else:
        raise ValueError(f"Unknown topology in preset: {topology}")

    cmap = CouplingMap(couplinglist=edges)
    # Ensure native Python types for JSON serialization
    edges = [[int(e[0]), int(e[1])] for e in edges]
    return cmap, edges, int(num_qubits)


def create_coupling_map_from_edges(
    edges: list[list[int]],
    bidirectional: bool = True,
) -> tuple[CouplingMap, list[list[int]], int]:
    """Create a CouplingMap from custom edges.

    Args:
        edges: List of [source, target] pairs
        bidirectional: Whether to add reverse edges

    Returns:
        Tuple of (CouplingMap, edges, num_qubits)
    """
    all_edges = list(edges)
    if bidirectional:
        existing = set(tuple(e) for e in edges)
        for e in edges:
            reverse = (e[1], e[0])
            if reverse not in existing:
                all_edges.append([e[1], e[0]])

    # Determine number of qubits from edges
    all_qubits = set()
    for e in all_edges:
        all_qubits.add(e[0])
        all_qubits.add(e[1])
    num_qubits = int(max(all_qubits) + 1) if all_qubits else 0

    cmap = CouplingMap(couplinglist=all_edges)
    # Ensure native Python types for JSON serialization
    all_edges = [[int(e[0]), int(e[1])] for e in all_edges]
    return cmap, all_edges, num_qubits


# ============================================================================
# Subtopology Extraction
# ============================================================================


def _get_connected_subgraphs(
    edges: list[list[int]],
    num_qubits: int,
    subgraph_size: int,
    max_subgraphs: int = 100,
) -> list[list[list[int]]]:
    """Find connected subgraphs of a specific size.

    Uses BFS/DFS to find connected induced subgraphs.

    Args:
        edges: Coupling map edges
        num_qubits: Total number of qubits
        subgraph_size: Number of qubits in subgraph
        max_subgraphs: Maximum number of subgraphs to return

    Returns:
        List of edge lists for each connected subgraph
    """
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(num_qubits)}
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    found_subgraphs: list[frozenset[int]] = []
    subgraph_edges: list[list[list[int]]] = []

    def is_connected(nodes: set[int]) -> bool:
        """Check if a set of nodes forms a connected subgraph."""
        if len(nodes) <= 1:
            return True
        start = next(iter(nodes))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor in nodes and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(nodes)

    def get_induced_edges(nodes: set[int]) -> list[list[int]]:
        """Get edges induced by a set of nodes."""
        node_list = sorted(nodes)
        node_map = {old: new for new, old in enumerate(node_list)}
        induced = []
        for e in edges:
            if e[0] in nodes and e[1] in nodes:
                induced.append([node_map[e[0]], node_map[e[1]]])
        return induced

    # For small subgraph sizes, enumerate combinations
    if subgraph_size <= 6:
        for combo in itertools.combinations(range(num_qubits), subgraph_size):
            if len(found_subgraphs) >= max_subgraphs:
                break
            nodes = set(combo)
            if is_connected(nodes):
                frozen = frozenset(nodes)
                # Check for isomorphic duplicates (simplified - just check exact match)
                if frozen not in found_subgraphs:
                    found_subgraphs.append(frozen)
                    subgraph_edges.append(get_induced_edges(nodes))
    else:
        # For larger sizes, use random sampling with BFS growth
        import random

        attempts = 0
        max_attempts = max_subgraphs * 10

        while len(found_subgraphs) < max_subgraphs and attempts < max_attempts:
            attempts += 1
            # Start from random node and grow via BFS
            start = random.randint(0, num_qubits - 1)
            nodes = {start}
            frontier = list(adj[start])
            random.shuffle(frontier)

            while len(nodes) < subgraph_size and frontier:
                next_node = frontier.pop(0)
                if next_node not in nodes:
                    nodes.add(next_node)
                    for neighbor in adj[next_node]:
                        if neighbor not in nodes and neighbor not in frontier:
                            frontier.append(neighbor)
                    random.shuffle(frontier)

            if len(nodes) == subgraph_size:
                frozen = frozenset(nodes)
                if frozen not in found_subgraphs:
                    found_subgraphs.append(frozen)
                    subgraph_edges.append(get_induced_edges(nodes))

    return subgraph_edges


def extract_unique_subtopologies(
    preset_or_edges: str | list[list[int]],
    num_qubits_target: int,
    max_subtopologies: int = 50,
) -> list[dict[str, Any]]:
    """Extract unique connected subtopologies of N qubits from a coupling map.

    This function finds connected subgraphs of the specified size, useful for
    training RL models on subtopologies of larger hardware.

    Args:
        preset_or_edges: Hardware preset name or list of edges
        num_qubits_target: Number of qubits for subtopologies
        max_subtopologies: Maximum number of subtopologies to return

    Returns:
        List of subtopology dicts with edges and metadata
    """
    # Get edges from preset or use directly
    if isinstance(preset_or_edges, str):
        _, edges, total_qubits = create_coupling_map_from_preset(preset_or_edges)
        source = preset_or_edges
    else:
        edges = preset_or_edges
        all_qubits = set()
        for e in edges:
            all_qubits.update(e)
        total_qubits = max(all_qubits) + 1 if all_qubits else 0
        source = "custom"

    if num_qubits_target > total_qubits:
        raise ValueError(f"Requested {num_qubits_target} qubits but source only has {total_qubits}")

    if num_qubits_target > QISKIT_GYM_MAX_QUBITS:
        raise ValueError(
            f"Requested {num_qubits_target} qubits exceeds maximum {QISKIT_GYM_MAX_QUBITS}"
        )

    # Find connected subgraphs
    subgraph_edges = _get_connected_subgraphs(
        edges, total_qubits, num_qubits_target, max_subtopologies
    )

    # Build result
    subtopologies = []
    for i, sub_edges in enumerate(subgraph_edges):
        # Classify the subtopology shape
        shape = _classify_topology_shape(sub_edges, num_qubits_target)
        subtopologies.append(
            {
                "subtopology_id": f"sub_{source}_{num_qubits_target}q_{i:03d}",
                "source": source,
                "num_qubits": num_qubits_target,
                "num_edges": len(sub_edges),
                "shape": shape,
                "edges": sub_edges,
            }
        )

    return subtopologies


def _classify_topology_shape(edges: list[list[int]], num_qubits: int) -> str:
    """Classify the shape of a small topology."""
    num_edges = len(edges) // 2  # Bidirectional edges counted once

    # Build degree distribution
    degrees: dict[int, int] = dict.fromkeys(range(num_qubits), 0)
    seen = set()
    for e in edges:
        key = (min(e[0], e[1]), max(e[0], e[1]))
        if key not in seen:
            seen.add(key)
            degrees[e[0]] += 1
            degrees[e[1]] += 1

    degree_list = sorted(degrees.values())
    max_degree = max(degree_list)

    # Classify based on structure
    if num_edges == num_qubits - 1:
        if max_degree == 2:
            return "line"
        elif max_degree == num_qubits - 1:
            return "star"
        else:
            return "tree"
    elif num_edges == num_qubits:
        if max_degree == 2:
            return "ring"
    elif num_edges >= num_qubits:
        # Check for grid-like structure
        if all(d <= 4 for d in degree_list):
            # Could be grid or partial grid
            if num_qubits in [4, 6, 9, 12, 16]:
                return "grid"
    return "complex"


# ============================================================================
# Async Tool Functions
# ============================================================================


@with_sync
async def get_coupling_map_presets() -> dict[str, Any]:
    """Get available hardware coupling map presets.

    Returns:
        Dict with preset information
    """
    presets = {}
    for name, config in HARDWARE_PRESETS.items():
        presets[name] = {
            "description": config["description"],
            "num_qubits": config["num_qubits"],
            "topology": config["topology"],
            "basis_gates": config.get("basis_gates"),
        }
        if config["topology"] == "grid":
            presets[name]["dimensions"] = f"{config['rows']}x{config['cols']}"

    return {
        "status": "success",
        "presets": presets,
        "usage": "Pass preset name to create_*_env tools or extract_subtopologies",
    }


@with_sync
async def create_custom_coupling_map(
    edges: list[list[int]] | None = None,
    topology: str | None = None,
    num_qubits: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    bidirectional: bool = True,
) -> dict[str, Any]:
    """Create a custom coupling map.

    Args:
        edges: Custom edges as [[q1, q2], ...] (mutually exclusive with topology)
        topology: Topology type ("grid", "line") (mutually exclusive with edges)
        num_qubits: Number of qubits (for line topology)
        rows: Number of rows (for grid topology)
        cols: Number of columns (for grid topology)
        bidirectional: Whether edges are bidirectional

    Returns:
        Dict with coupling map info and edges
    """
    try:
        if edges is not None and topology is not None:
            return {
                "status": "error",
                "message": "Provide either 'edges' or 'topology', not both",
            }

        if edges is not None:
            _, final_edges, n_qubits = create_coupling_map_from_edges(edges, bidirectional)
            source = "custom"
        elif topology is not None:
            final_edges = generate_coupling_map_edges(
                topology, num_qubits, rows, cols, bidirectional
            )
            n_qubits = num_qubits or (rows * cols if rows and cols else 0)
            source = topology
        else:
            return {
                "status": "error",
                "message": "Provide either 'edges' or 'topology' parameter",
            }

        if n_qubits > QISKIT_GYM_MAX_QUBITS:
            return {
                "status": "error",
                "message": f"Coupling map has {n_qubits} qubits, exceeds max {QISKIT_GYM_MAX_QUBITS}",
            }

        return {
            "status": "success",
            "source": source,
            "num_qubits": n_qubits,
            "num_edges": len(final_edges),
            "edges": final_edges,
            "bidirectional": bidirectional,
        }

    except Exception as e:
        logger.error(f"Failed to create coupling map: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def extract_subtopologies(
    preset: str | None = None,
    edges: list[list[int]] | None = None,
    num_qubits: int = 4,
    max_subtopologies: int = 50,
) -> dict[str, Any]:
    """Extract connected subtopologies of N qubits from a hardware preset or custom coupling map.

    Use this to find all unique connected subgraphs of a specified size from a
    larger coupling map. Useful for training RL models on subtopologies of
    real quantum hardware.

    Args:
        preset: Hardware preset name (e.g., "ibm_nighthawk", "ibm_heron_r1")
        edges: Custom coupling map edges (alternative to preset)
        num_qubits: Number of qubits for subtopologies (default: 4)
        max_subtopologies: Maximum number of subtopologies to return (default: 50)

    Returns:
        Dict with list of subtopologies, each containing edges and metadata
    """
    try:
        if preset is None and edges is None:
            return {
                "status": "error",
                "message": "Provide either 'preset' or 'edges' parameter",
            }

        source = preset if preset else edges
        subtopologies = extract_unique_subtopologies(
            source,  # type: ignore
            num_qubits,
            max_subtopologies,
        )

        return {
            "status": "success",
            "source": preset or "custom",
            "num_qubits": num_qubits,
            "subtopologies_found": len(subtopologies),
            "subtopologies": subtopologies,
        }

    except Exception as e:
        logger.error(f"Failed to extract subtopologies: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def list_subtopology_shapes(
    preset: str,
    num_qubits: int,
) -> dict[str, Any]:
    """List available subtopology shapes for a given hardware and qubit count.

    Summarizes the types of subtopologies available without returning all edges.

    Args:
        preset: Hardware preset name
        num_qubits: Number of qubits for subtopologies

    Returns:
        Dict with shape counts and sample subtopologies
    """
    try:
        subtopologies = extract_unique_subtopologies(preset, num_qubits, max_subtopologies=100)

        # Count shapes
        shape_counts: dict[str, int] = {}
        shape_examples: dict[str, dict[str, Any]] = {}
        for sub in subtopologies:
            shape = sub["shape"]
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            if shape not in shape_examples:
                shape_examples[shape] = {
                    "subtopology_id": sub["subtopology_id"],
                    "num_edges": sub["num_edges"],
                    "edges": sub["edges"],
                }

        return {
            "status": "success",
            "preset": preset,
            "num_qubits": num_qubits,
            "total_subtopologies": len(subtopologies),
            "shape_counts": shape_counts,
            "shape_examples": shape_examples,
        }

    except Exception as e:
        logger.error(f"Failed to list subtopology shapes: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# Fake Backend Coupling Maps (Offline IBM Topologies)
# ============================================================================

# Mapping of fake backend class names for offline access to exact IBM topologies
FAKE_BACKEND_MAP: dict[str, str] = {
    # Heron backends (133 qubits, heavy-hex)
    "ibm_fez": "FakeFez",
    "ibm_marrakesh": "FakeMarrakesh",
    "ibm_torino": "FakeTorino",
    # Eagle backends (127 qubits, heavy-hex)
    "ibm_brisbane": "FakeBrisbane",
    "ibm_kyoto": "FakeKyoto",
    "ibm_osaka": "FakeOsaka",
    "ibm_sherbrooke": "FakeSherbrooke",
    # Older backends
    "ibm_algiers": "FakeAlgiers",
    "ibm_hanoi": "FakeHanoi",
    "ibm_cairo": "FakeCairo",
    "ibm_kolkata": "FakeKolkata",
    "ibm_mumbai": "FakeMumbai",
    "ibm_peekskill": "FakePeekskill",
    "ibm_prague": "FakePrague",
    "ibm_cusco": "FakeCusco",
    "ibm_kawasaki": "FakeKawasaki",
    "ibm_kyiv": "FakeKyiv",
    "ibm_nazca": "FakeNazca",
    "ibm_quebec": "FakeQuebec",
}


def _get_fake_backend(backend_name: str) -> Any:
    """Get a fake backend instance by name.

    Args:
        backend_name: Backend name. Accepts multiple formats:
            - "fake_fez" (exact match)
            - "ibm_fez" (converts to fake_fez)
            - "fez" (converts to fake_fez)

    Returns:
        Fake backend instance

    Raises:
        ValueError: If backend not found in fake provider
    """
    from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

    provider = FakeProviderForBackendV2()
    available = [b.name for b in provider.backends()]

    # Normalize the backend name to try multiple formats
    names_to_try = [backend_name]

    # If name starts with ibm_, try fake_ version
    if backend_name.startswith("ibm_"):
        base_name = backend_name[4:]  # Remove "ibm_"
        names_to_try.append(f"fake_{base_name}")
        names_to_try.append(base_name)

    # If name doesn't have a prefix, try with fake_ prefix
    if not backend_name.startswith("fake_") and not backend_name.startswith("ibm_"):
        names_to_try.append(f"fake_{backend_name}")

    # Try each name variant
    for name in names_to_try:
        for backend in provider.backends():
            if backend.name == name:
                return backend

    raise ValueError(
        f"Fake backend '{backend_name}' not found. Available: {', '.join(sorted(available))}"
    )


@with_sync
async def get_fake_backend_coupling_map(backend_name: str) -> dict[str, Any]:
    """Get the exact coupling map from a fake IBM backend (no credentials needed).

    Use this to get exact IBM Quantum hardware topologies without needing
    IBM Quantum credentials. Fake backends provide accurate coupling maps
    for offline development and testing.

    Args:
        backend_name: Backend name (e.g., "ibm_fez", "ibm_brisbane", "ibm_torino")

    Returns:
        Dict with exact coupling map edges, num_qubits, and backend info
    """
    try:
        backend = _get_fake_backend(backend_name)
        coupling_map = backend.coupling_map

        if coupling_map is None:
            return {
                "status": "error",
                "message": f"Backend '{backend_name}' has no coupling map (fully connected)",
            }

        edges = [[int(e[0]), int(e[1])] for e in coupling_map.get_edges()]
        num_qubits = coupling_map.size()

        return {
            "status": "success",
            "backend_name": backend.name,
            "num_qubits": num_qubits,
            "num_edges": len(edges),
            "edges": edges,
            "source": "fake_backend",
            "note": "Exact IBM topology from qiskit-ibm-runtime fake provider",
        }

    except Exception as e:
        logger.error(f"Failed to get fake backend coupling map: {e}")
        return {"status": "error", "message": str(e)}


@with_sync
async def list_available_fake_backends() -> dict[str, Any]:
    """List all available fake backends for offline topology access.

    Returns:
        Dict with list of fake backends and their properties
    """
    try:
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

        provider = FakeProviderForBackendV2()
        backends = []

        for backend in provider.backends():
            coupling_map = backend.coupling_map
            num_qubits = coupling_map.size() if coupling_map else backend.num_qubits

            backends.append(
                {
                    "name": backend.name,
                    "num_qubits": num_qubits,
                    "has_coupling_map": coupling_map is not None,
                }
            )

        # Sort by qubit count descending
        backends.sort(key=lambda x: x["num_qubits"], reverse=True)

        return {
            "status": "success",
            "num_backends": len(backends),
            "backends": backends,
            "usage": "Use get_fake_backend_coupling_map(backend_name) to get exact topology",
        }

    except Exception as e:
        logger.error(f"Failed to list fake backends: {e}")
        return {"status": "error", "message": str(e)}
