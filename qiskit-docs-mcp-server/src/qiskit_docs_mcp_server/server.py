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
    get_list_of_addons,
    get_list_of_error_code_categories,
    get_list_of_guides,
    get_list_of_modules,
    get_page_docs,
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
async def search_docs_tool(query: str, scope: str = "all") -> dict[str, Any]:
    """Search across the entire Qiskit documentation for relevant content.

    Use this as the primary entry point to discover documentation pages.
    Returns ranked results with titles, URLs, sections, and text snippets.
    Use get_page_tool to fetch the full content of any result URL.

    Args:
        query: Search query string (e.g., 'error mitigation', 'QuantumCircuit',
            'transpiler optimization'). More specific queries yield better results.
        scope: Search scope filter (case-sensitive). Valid values:
            'all' — Search everything (default)
            'documentation' — Guides and general docs
            'api' — API reference pages only
            'learning' — Learning resources and tutorials
            'tutorials' — Tutorial content only

    Returns:
        List of matching documentation entries with URLs, titles, sections,
        and text snippets. Use the URLs with get_page_tool to fetch full content.
    """
    return await search_qiskit_docs(query, scope)


@mcp.tool()
async def get_page_tool(
    url: str,
    max_length: int = 20000,
    offset: int = 0,
) -> dict[str, Any]:
    """Fetch a Qiskit documentation page and return its content as markdown.

    Accepts any URL from the Qiskit documentation site. Use search_docs_tool
    first to find the right page, or use URLs from the resource lists.

    Returns documentation in markdown format with pagination support.
    Default max_length is 20000 chars. Set max_length=0 for unlimited.
    Use offset to retrieve subsequent pages when has_more is true.

    This tool can fetch ANY page in the Qiskit documentation, including:
    - SDK module API references (e.g., 'api/qiskit/circuit')
    - Individual class pages (e.g., 'api/qiskit/qiskit.circuit.QuantumCircuit')
    - Addon documentation (e.g., 'api/qiskit-addon-sqd')
    - Implementation guides (e.g., 'guides/transpile')
    - Any other documentation page

    Args:
        url: Documentation page URL. Accepts:
            - Full URL: 'https://quantum.cloud.ibm.com/docs/guides/transpile'
            - Relative path: 'guides/transpile', 'api/qiskit/circuit'
        max_length: Maximum characters to return (default: 20000, 0 for unlimited)
        offset: Character offset for pagination (default: 0)

    Returns:
        Page content in markdown format with pagination metadata
        (has_more, next_offset, total_length), or error with suggestion
        to use search_docs_tool if the page is not found.
    """
    return await get_page_docs(url, max_length=max_length, offset=offset)


@mcp.tool()
async def lookup_error_code_tool(code: str) -> dict[str, Any]:
    """Look up a Qiskit or IBM Quantum error code to get its description and solution.

    Use this when a user encounters a numeric error code from Qiskit or
    IBM Quantum services. Returns the error message and suggested fix.
    Read the qiskit-docs://error-codes resource for error code categories.

    Error code ranges:
        1XXX: Validation, transpilation, backend, authorization, job management
        2XXX: Backend configuration, booking, data retrieval
        3XXX: Job handling, authentication, analytics
        4XXX: Session management and job limits
        5XXX: Job timeout and cancellation
        6XXX: Shot limits, compiler input, control system
        7XXX: Instruction and basis gate compatibility
        8XXX: Pulse and channel configuration
        9XXX: Hardware loading and internal errors

    Args:
        code: 4-digit numeric error code as a string (e.g., '1002', '7001').
            Must be exactly 4 digits.

    Returns:
        Error code details including message, solution, and link to the
        error registry. Returns error if code format is invalid or not found.
    """
    return await lookup_error_code(code)


##################################################
## MCP Prompts
## - https://modelcontextprotocol.io/docs/concepts/prompts
##################################################


@mcp.prompt()
def explain_error(code: str) -> str:
    """Look up a Qiskit error code and explain what it means and how to fix it."""
    return (
        f"Look up error code {code} using lookup_error_code_tool, then explain the "
        "error in plain language and suggest how to fix it."
    )


@mcp.prompt()
def module_overview(module: str) -> str:
    """Get an overview of a Qiskit SDK module."""
    return (
        f"Fetch the documentation for the '{module}' module using get_page_tool with "
        f"url 'api/qiskit/{module}', then provide a concise overview of the module's "
        "purpose, key classes, and common usage patterns."
    )


@mcp.prompt()
def how_to(task: str) -> str:
    """Find documentation on how to accomplish a task with Qiskit."""
    return (
        f"Search for '{task}' using search_docs_tool, then fetch the most relevant "
        "result using get_page_tool, and explain how to accomplish this task step by step."
    )


##################################################
## MCP Resources
## - https://modelcontextprotocol.io/docs/concepts/resources
##################################################


@mcp.resource("qiskit-docs://modules", mime_type="application/json")
def modules_resource() -> dict[str, Any]:
    """Get list of all Qiskit SDK modules with URL paths.

    Returns curated list of common SDK modules. Use get_page_tool with
    'api/qiskit/{module}' to fetch documentation, or search_docs_tool
    to discover any module page.
    """
    return get_list_of_modules()


@mcp.resource("qiskit-docs://addons", mime_type="application/json")
def addons_resource() -> dict[str, Any]:
    """Get list of Qiskit addon packages with URL paths.

    Returns curated list of addon packages. Use get_page_tool with
    'api/qiskit-addon-{name}' to fetch documentation.
    """
    return get_list_of_addons()


@mcp.resource("qiskit-docs://guides", mime_type="application/json")
def guides_resource() -> dict[str, Any]:
    """Get list of Qiskit implementation guides with URL paths.

    Returns curated list of common guides. Use get_page_tool with
    'guides/{name}' to fetch documentation, or search_docs_tool to
    discover any guide.
    """
    return get_list_of_guides()


@mcp.resource("qiskit-docs://error-codes", mime_type="application/json")
def error_codes_resource() -> dict[str, Any]:
    """Get list of IBM Quantum error code categories and registry URL."""
    return get_list_of_error_code_categories()


##################################################
## MCP Resource Templates
## - https://modelcontextprotocol.io/docs/concepts/resources#resource-templates
##################################################


@mcp.resource("qiskit-docs://modules/{module_name}", mime_type="application/json")
async def module_docs_resource(module_name: str) -> dict[str, Any]:
    """Get documentation for a specific Qiskit SDK module."""
    return await get_page_docs(f"api/qiskit/{module_name}")


@mcp.resource("qiskit-docs://guides/{guide_name}", mime_type="application/json")
async def guide_docs_resource(guide_name: str) -> dict[str, Any]:
    """Get a specific Qiskit implementation guide."""
    return await get_page_docs(f"guides/{guide_name}")


@mcp.resource("qiskit-docs://addons/{addon_name}", mime_type="application/json")
async def addon_docs_resource(addon_name: str) -> dict[str, Any]:
    """Get documentation for a specific Qiskit addon package."""
    return await get_page_docs(f"api/qiskit-addon-{addon_name}")
