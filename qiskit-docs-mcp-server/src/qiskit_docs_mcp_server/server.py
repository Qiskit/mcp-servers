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
    get_addon_docs,
    get_component_docs,
    get_guide_docs,
    get_list_of_addons,
    get_list_of_error_code_categories,
    get_list_of_guides,
    get_list_of_modules,
    lookup_error_code,
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
async def get_sdk_module_docs_tool(module: str) -> dict[str, Any]:
    """Get the full API reference documentation for a Qiskit SDK module.

    Returns the complete API reference in markdown format. Responses can be
    very large (up to 100K+ chars for modules like 'circuit'). Consider using
    search_docs_tool first to find specific topics, or read the
    qiskit-docs://modules resource to see all available modules.

    Args:
        module: Exact module name. Valid values:
            Circuit construction: 'circuit'
            Quantum information: 'quantum_info'
            Transpilation: 'transpiler', 'synthesis', 'dagcircuit',
                'passmanager', 'converters', 'compiler'
            Primitives and providers: 'primitives', 'providers'
            Results and visualization: 'result', 'visualization'
            Serialization: 'qasm2', 'qasm3', 'qpy'
            Utilities: 'utils', 'exceptions'

    Returns:
        Module documentation in markdown format with metadata, or error
        with fuzzy-match suggestions if the module name is invalid.
    """
    return await get_component_docs(module)


@mcp.tool()
async def get_addon_docs_tool(addon: str) -> dict[str, Any]:
    """Get API documentation for a Qiskit addon package.

    Returns the full API reference for Qiskit addon modules. These are
    separate packages that extend Qiskit with additional algorithms
    and capabilities. Read the qiskit-docs://addons resource for the
    full list with descriptions.

    Args:
        addon: Exact addon name. Valid values:
            'aqc-tensor' — Approximate Quantum Compiler with tensor networks
            'cutting' — Circuit cutting for large circuits
            'mpf' — Multi-product formulas for Hamiltonian simulation
            'obp' — Operator backpropagation
            'sqd' — Sample-based Quantum Diagonalization
            'utils' — Shared utilities for addon packages

    Returns:
        Addon API documentation in markdown format with metadata, or error
        with fuzzy-match suggestions if the addon name is invalid.
    """
    return await get_addon_docs(addon)


@mcp.tool()
async def get_guide_tool(guide: str) -> dict[str, Any]:
    """Get a Qiskit implementation guide or best practice document.

    Returns a complete how-to guide in markdown format. Guides cover
    practical topics like circuit construction, transpilation, error
    mitigation, and execution. Read the qiskit-docs://guides resource
    to see all available guides with descriptions.

    Args:
        guide: Exact guide slug name. Common guides:
            Getting started: 'quick-start'
            Circuits: 'construct-circuits', 'dynamic-circuits'
            Transpilation: 'transpile', 'transpiler-stages',
                'transpile-with-pass-managers',
                'defaults-and-configuration-options'
            Error handling: 'configure-error-mitigation',
                'configure-error-suppression',
                'error-mitigation-and-suppression-techniques'
            Execution: 'primitives', 'execution-modes',
                'runtime-options-overview'
            Functions: 'functions', 'ibm-circuit-function'

    Returns:
        Guide documentation in markdown format with metadata, or error
        with fuzzy-match suggestions if the guide name is invalid.
    """
    return await get_guide_docs(guide)


@mcp.tool()
async def search_docs_tool(query: str, module: str = "documentation") -> dict[str, Any]:
    """Search across the entire Qiskit documentation for relevant content.

    Use this tool as a starting point when you're not sure which specific
    module or guide to fetch. Returns ranked results with titles, URLs,
    sections, and text snippets.

    Args:
        query: Search query string (e.g., 'error mitigation',
            'QuantumCircuit', 'transpiler optimization'). More specific
            queries yield better results.
        module: Search scope filter (case-sensitive). Valid values:
            'all' — Search everything
            'documentation' — Guides and general docs (default)
            'api' — API reference pages only
            'learning' — Learning resources and tutorials
            'tutorials' — Tutorial content only

    Returns:
        List of matching documentation entries with URLs, titles,
        sections, and text snippets.
    """
    return await search_qiskit_docs(query, module)


@mcp.tool()
async def lookup_error_code_tool(code: str) -> dict[str, Any]:
    """Look up a Qiskit or IBM Quantum error code to get its description and solution.

    Use this when a user encounters a numeric error code from Qiskit or
    IBM Quantum services. Returns the error message and suggested fix.
    Read the qiskit-docs://error-codes resource for error code categories.

    Error code ranges:
        1XXX: Validation, transpilation, backend, authorization
        2XXX: Backend configuration, booking, data retrieval
        3XXX: Job handling, authentication, analytics
        4XXX: Session management and job limits
        5XXX: Job timeout and cancellation
        6XXX: Shot limits, compiler input, control system
        7XXX: Instruction and basis gate compatibility
        8XXX: Pulse and channel configuration
        9XXX: Hardware loading and internal errors

    Args:
        code: 4-digit numeric error code as a string (e.g., '1002',
            '7001', '8004'). Must be exactly 4 digits.

    Returns:
        Error code details including message, solution, and link to
        the error registry. Returns error if code is invalid or not found.
    """
    return await lookup_error_code(code)


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################


@mcp.resource("qiskit-docs://modules", mime_type="application/json")
def modules_resource() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules."""
    return get_list_of_modules()


@mcp.resource("qiskit-docs://addons", mime_type="application/json")
def addons_resource() -> dict[str, Any]:
    """Get list of all Qiskit addon modules."""
    return get_list_of_addons()


@mcp.resource("qiskit-docs://guides", mime_type="application/json")
def guides_resource() -> dict[str, Any]:
    """Get list of Qiskit guides and best practices."""
    return get_list_of_guides()


@mcp.resource("qiskit-docs://error-codes", mime_type="application/json")
def error_codes_resource() -> dict[str, Any]:
    """Get list of IBM Quantum error code categories and registry URL."""
    return get_list_of_error_code_categories()
