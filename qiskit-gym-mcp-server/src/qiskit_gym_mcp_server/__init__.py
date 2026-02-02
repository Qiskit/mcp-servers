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

"""Qiskit Gym MCP Server.

A Model Context Protocol server that provides reinforcement learning-based
quantum circuit synthesis capabilities using qiskit-gym.

Features:
- Train RL models for permutation, linear function, and Clifford circuit synthesis
- Extract subtopologies from IBM Quantum hardware (Heron, Nighthawk)
- Save, load, and manage trained models
- Synthesize optimal quantum circuits using trained models
- TensorBoard integration for training visualization
"""

from . import server


def main() -> None:
    """Main entry point for the package."""
    server.mcp.run(transport="stdio")


__all__ = ["main", "server"]
