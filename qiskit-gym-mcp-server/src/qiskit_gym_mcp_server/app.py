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

"""FastMCP application instance.

This module creates the shared MCP server instance used by tools and resources.
Separating this allows tools and resources to be defined in their own modules
without circular imports.
"""

from fastmcp import FastMCP


# Initialize MCP server - shared by all tools and resources
mcp = FastMCP(
    "Qiskit Gym",
    instructions="""\
This server provides reinforcement learning-based quantum circuit synthesis \
using qiskit-gym.

Recommended workflow:
1. Create an environment for the target synthesis problem:
   - create_permutation_env_tool for SWAP-based permutation circuits
   - create_linear_function_env_tool for CX/SWAP linear function circuits
   - create_clifford_env_tool for Clifford (H, S, CX) circuits
2. Train a model with start_training_tool (or batch_train_environments_tool \
for multiple environments). Monitor progress with get_training_status_tool \
and get_training_metrics_tool, or use wait_for_training_tool to block until \
completion.
3. Synthesize optimal circuits using the matching synthesis tool: \
synthesize_permutation_tool, synthesize_linear_function_tool, or \
synthesize_clifford_tool.
4. Save trained models with save_model_tool for reuse, and load them later \
with load_model_tool.

Hardware topologies:
- Use extract_subtopologies_tool to get subtopologies from IBM Quantum \
hardware (Heron, Nighthawk).
- Use list_available_fake_backends_tool and \
get_fake_backend_coupling_map_tool to explore backend connectivity.

Browse qiskit-gym:// resources for available algorithms, policies, \
coupling map presets, and step-by-step workflows.\
""",
)
