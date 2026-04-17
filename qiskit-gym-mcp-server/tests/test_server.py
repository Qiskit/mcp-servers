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

"""Tests for MCP server registration and configuration."""

from qiskit_gym_mcp_server.app import mcp


class TestServerRegistration:
    """Test that the MCP server is configured correctly."""

    def test_server_name(self):
        """Test the MCP server has the correct name."""
        assert mcp.name == "Qiskit Gym"

    def test_server_instructions(self):
        """Test the MCP server has instructions set."""
        assert mcp.instructions is not None
        assert isinstance(mcp.instructions, str)
        assert len(mcp.instructions) > 0
        # Verify instructions mention key workflow concepts
        assert "create_permutation_env_tool" in mcp.instructions
        assert "start_training_tool" in mcp.instructions
        assert "synthesize_permutation_tool" in mcp.instructions
        assert "qiskit-gym://" in mcp.instructions
