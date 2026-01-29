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

from mcp.server.fastmcp import FastMCP
from data_fetcher import *

mcp = FastMCP("qiskit_docs")


@mcp.tool()
def list_sdk_modules() -> list[str]:
    """List all available Qiskit SDK modules (circuit, primitives, transpiler, etc.)."""
    return get_component_list()


@mcp.tool()
def list_addons() -> list[str]:
    """List all available Qiskit addon modules and tutorials (addon-opt-mapper, addon-vqe, etc.)."""
    return get_pattern_list()


@mcp.tool()
def list_guides() -> list[str]:
    """List all available Qiskit guides and best practices (optimization, error-mitigation, etc.)."""
    return get_style_list()


@mcp.tool()
def get_sdk_module_docs(module: str) -> dict:
    """
    Get documentation for a Qiskit SDK module.
    
    Args:
        module: Module name (e.g., 'circuit', 'primitives', 'transpiler', 'quantum_info')
    
    Returns:
        Module documentation including API reference and usage examples.
    """
    docs = get_component_docs(module)
    if docs is None:
        return {"error": f"Module '{module}' not found. Use list_sdk_modules() to see available modules."}
    return {"module": module, "documentation": docs}

@mcp.tool()
def get_guide(guide: str) -> dict:
    """
    Get a Qiskit guide or best practice documentation.
    
    Args:
        guide: Guide name (e.g., 'optimization', 'error-mitigation', 'dynamic-circuits', 'performance-tuning')
    
    Returns:
        Complete guide documentation with best practices and implementation patterns.
    """
    docs = get_guide_docs(guide)
    if docs is None:
        return {"error": f"Guide '{guide}' not found. Use list_guides() to see available guides."}
    return {"guide": guide, "documentation": docs}


@mcp.tool()
def search_docs(query: str, module: str = "documentation") -> list[dict]:
    """
    Search Qiskit documentation for relevant modules, addons, and guides.
    
    Args:
        query: Search query (e.g., 'optimization', 'circuit', 'error')
        module: Search module (e.g. 'documentation', 'API' etc)
    
    Returns:
        List of matching documentation entries with URLs and types.
    """
    results = search_qiskit_docs(query, module)
    if not results:
        return [{"info": f"No results found for '{query}'"}]
    return results

## Resource
@mcp.resource("qdc://modules", mime_type="application/json")
def get_component_list() -> list[str]:
    """Get list of all Qiskit SDK modules."""
    return list(QISKIT_MODULES.keys())

@mcp.resource("qdc://pattern", mime_type="application/json")
def get_pattern_list() -> list[str]:
    """Get list of all Qiskit addon modules and tutorials."""
    return list(QISKIT_ADDON_MODULES.keys())

@mcp.resource("qdc://style", mime_type="application/json")
def get_style_list() -> list[str]:
    """Get list of Qiskit guides and best practices."""
    return [
        "optimization",
        "quantum-circuits",
        "error-mitigation",
        "dynamic-circuits",
        "parametric-compilation",
        "performance-tuning"
    ]
