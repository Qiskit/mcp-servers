# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for MCP server registration and configuration."""

from qiskit_mcp_server.server import mcp


class TestServerRegistration:
    """Test that tools, resources, and prompts are registered on the MCP server."""

    def test_server_name(self):
        """Test the server name is correct."""
        assert mcp.name == "Qiskit"

    def test_server_instructions(self):
        """Test the MCP server has instructions set."""
        assert mcp.instructions is not None
        assert isinstance(mcp.instructions, str)
        assert len(mcp.instructions) > 0
        # Verify instructions mention key workflow concepts
        assert "transpile_circuit_tool" in mcp.instructions
        assert "analyze_circuit_tool" in mcp.instructions
        assert "load_circuit_from_qasm_tool" in mcp.instructions
        assert "qiskit://transpiler/" in mcp.instructions

    def test_tools_registered(self):
        """Test that all expected tools are registered."""
        tool_names = {tool.name for tool in mcp._tool_manager._tools.values()}
        expected_tools = {
            "transpile_circuit_tool",
            "analyze_circuit_tool",
            "compare_optimization_levels_tool",
            "convert_qpy_to_qasm3_tool",
            "convert_qasm3_to_qpy_tool",
            "load_circuit_from_qasm_tool",
            "export_circuit_to_qasm_tool",
        }
        assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"

    def test_tool_count(self):
        """Test the expected number of tools."""
        assert len(mcp._tool_manager._tools) == 7

    def test_resources_registered(self):
        """Test that all expected resources are registered."""
        resource_uris = set(mcp._resource_manager._resources.keys())
        expected_resources = {
            "qiskit://transpiler/info",
            "qiskit://transpiler/basis-gates",
            "qiskit://transpiler/topologies",
        }
        assert expected_resources.issubset(resource_uris), (
            f"Missing resources: {expected_resources - resource_uris}"
        )

    def test_resource_count(self):
        """Test the expected number of resources."""
        assert len(mcp._resource_manager._resources) == 3

    def test_prompts_registered(self):
        """Test that all expected prompts are registered."""
        prompt_names = set(mcp._prompt_manager._prompts.keys())
        expected_prompts = {
            "transpile_for_hardware",
            "analyze_and_transpile",
            "convert_circuit_format",
        }
        assert expected_prompts.issubset(prompt_names), (
            f"Missing prompts: {expected_prompts - prompt_names}"
        )

    def test_prompt_count(self):
        """Test the expected number of prompts."""
        assert len(mcp._prompt_manager._prompts) == 3
