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

"""
Qiskit Documentation MCP Server

A Model Context Protocol server that provides access to IBM Qiskit documentation
for querying and retrieving Qiskit documentation content and summaries.
"""

import logging
from typing import Any

from fastmcp import FastMCP

from qiskit_docs_mcp_server.data_fetcher import (
    get_component_docs,
    get_guide_docs,
    get_list_of_addons,
    get_list_of_guides,
    get_list_of_modules,
    search_qiskit_docs,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Qiskit Documentation")

logger.info("Qiskit Documentation MCP Server initialized")


##################################################
## MCP Tools
## - https://modelcontextprotocol.io/docs/concepts/tools
##################################################


@mcp.tool()
async def get_sdk_module_docs(module: str) -> dict[str, Any]:
    """
    Get documentation for a Qiskit SDK module.

    Args:
        module: Module name (e.g., 'circuit', 'primitives', 'transpiler', 'quantum_info')

    Returns:
        Module documentation including API reference and usage examples.
    """
    return await get_component_docs(module)


@mcp.tool()
async def get_guide(guide: str) -> dict[str, Any]:
    """
    Get a Qiskit guide or best practice documentation.

    Args:
        guide: Guide name (e.g., 'optimization', 'error-mitigation', 'dynamic-circuits', 'performance-tuning')

    Returns:
        Complete guide documentation with best practices and implementation patterns.
    """
    return await get_guide_docs(guide)


@mcp.tool()
async def search_docs(query: str, module: str = "documentation") -> dict[str, Any]:
    """
    Search Qiskit documentation for relevant modules, addons, and guides.

    Args:
        query: Search query (e.g., 'optimization', 'circuit', 'error')
        module: Search module (e.g. 'documentation', 'API' etc)

    Returns:
        List of matching documentation entries with URLs and types.
    """
    return await search_qiskit_docs(query, module)


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################


@mcp.resource("qiskit-docs://modules", mime_type="application/json")
async def modules_resource() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules."""
    return get_list_of_modules()


@mcp.resource("qiskit-docs://addons", mime_type="application/json")
async def addons_resource() -> dict[str, Any]:
    """Get list of all Qiskit addon modules and tutorials."""
    return get_list_of_addons()


@mcp.resource("qiskit-docs://guides", mime_type="application/json")
async def guides_resource() -> dict[str, Any]:
    """Get list of Qiskit guides and best practices."""
    return get_list_of_guides()
