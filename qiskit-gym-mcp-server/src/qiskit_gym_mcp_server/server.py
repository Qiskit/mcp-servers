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

"""Qiskit Gym MCP Server

A Model Context Protocol server that provides reinforcement learning-based
quantum circuit synthesis capabilities using qiskit-gym.

Features:
- Create RL environments for permutation, linear function, and Clifford synthesis
- Train models using PPO or AlphaZero algorithms
- Extract subtopologies from IBM Quantum hardware (Heron, Nighthawk)
- Synthesize optimal quantum circuits using trained models
- Save, load, and manage trained models
- TensorBoard integration for training visualization

This module serves as the entry point. Tools and resources are defined in
separate modules for maintainability:
- server_tools.py: All @mcp.tool() definitions
- server_resources.py: All @mcp.resource() definitions
- app.py: The shared FastMCP instance
"""

import logging

import qiskit_gym_mcp_server.server_resources

# Import tools and resources to register them with mcp
# These imports have side effects - the decorators register the functions
import qiskit_gym_mcp_server.server_tools  # noqa: F401

# Import the mcp instance
from qiskit_gym_mcp_server.app import mcp


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Re-export mcp for backwards compatibility
__all__ = ["main", "mcp"]


def main() -> None:
    """Run the server."""
    mcp.run()


if __name__ == "__main__":
    main()
