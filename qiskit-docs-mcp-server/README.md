# qiskit-docs-mcp-server

[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-docs-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-docs-mcp-server)

<!-- mcp-name: io.github.Qiskit/qiskit-docs-mcp-server -->

MCP server for querying and retrieving Qiskit documentation, guides, and API references.

## Overview

The Qiskit Documentation MCP Server provides AI assistants and agents with seamless access to the complete Qiskit documentation ecosystem. It enables intelligent retrieval of SDK module documentation, implementation guides, and best practices through a standardized Model Context Protocol interface.

### Key Features

- **📚 Complete Documentation Access**: Query all Qiskit SDK modules, addon packages, API references, guides, and tutorials
- **🔄 Dynamic Content Discovery**: Automatically discovers available documentation from the live sitemap — no manual updates needed when new content is published
- **📖 Implementation Guides**: Access best practices for optimization, error mitigation, dynamic circuits, and more
- **🔍 Smart Search**: Search across the entire Qiskit documentation with fuzzy matching
- **🎯 No Authentication Required**: Public documentation access without API tokens
- **📝 Markdown Output**: Clean, formatted documentation ready for AI consumption
- **⚡ Fast Retrieval**: Efficient HTTP-based documentation fetching with TTL caching and configurable timeouts

## Components

### Tools

The server implements three tools for documentation access:

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_docs_tool` | Search across the entire Qiskit documentation for relevant content | `query`: Search query string<br>`scope`: Search scope filter — `all` (default), `documentation`, `api`, `learning`, `tutorials` |
| `get_page_tool` | Fetch any Qiskit documentation page and return as markdown | `url`: Full URL or relative path (e.g., `guides/transpile`, `api/qiskit/circuit`)<br>`max_length`: Max characters to return (default: 20000, 0 for unlimited)<br>`offset`: Character offset for pagination (default: 0) |
| `lookup_error_code_tool` | Look up a Qiskit/IBM Quantum error code | `code`: 4-digit error code (e.g., 1002, 7001, 8004) |

### Resources

The server provides six resources for listing available documentation. Content lists for modules, addons, guides, tutorials, and API packages are **dynamically discovered** from the documentation sitemap and cached, with hardcoded fallback values used when the sitemap is unreachable.

| Resource URI | Description |
|--------------|-------------|
| `qiskit-docs://modules` | List of all Qiskit SDK modules with URL paths |
| `qiskit-docs://addons` | List of Qiskit addon packages with URL paths |
| `qiskit-docs://guides` | List of implementation guides and best practices |
| `qiskit-docs://tutorials` | List of Qiskit tutorials with URL paths |
| `qiskit-docs://api-packages` | List of API packages (runtime, transpiler, REST APIs, etc.) |
| `qiskit-docs://error-codes` | List of Qiskit error code categories |

### Resource Templates

| Resource URI | Description |
|--------------|-------------|
| `qiskit-docs://modules/{module_name}` | Documentation for a specific SDK module |
| `qiskit-docs://guides/{guide_name}` | A specific implementation guide |
| `qiskit-docs://addons/{addon_name}` | Documentation for a specific addon package |

## Prerequisites

- Python 3.10 or higher
- [uv](https://astral.sh/uv) package manager (recommended)
- Internet connection to access [IBM Quantum Documentation](https://quantum.cloud.ibm.com/docs/)

## Installation

### Install from PyPI

The easiest way to install is via pip:

```bash
pip install qiskit-docs-mcp-server
```

Or using uvx (recommended):

```bash
uvx qiskit-docs-mcp-server
```

### Install from Source

This project uses [uv](https://astral.sh/uv) for virtual environments and dependencies management. If you don't have `uv` installed, check out the instructions in <https://docs.astral.sh/uv/getting-started/installation/>

#### Setting up the Project with uv

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Qiskit/mcp-servers.git
   cd mcp-servers/qiskit-docs-mcp-server
   ```

2. **Initialize or sync the project**:
   ```bash
   # This will create a virtual environment and install dependencies
   uv sync
   ```

## Configuration

### Environment Variables

The server can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QISKIT_DOCS_BASE` | Base URL for Qiskit documentation | `https://quantum.cloud.ibm.com/docs/` |
| `QISKIT_HTTP_TIMEOUT` | HTTP request timeout in seconds | `10.0` |
| `QISKIT_DOCS_CACHE_TTL` | Page cache TTL in seconds | `3600.0` |
| `QISKIT_SEARCH_CACHE_TTL` | Search/JSON cache TTL in seconds | `300.0` |
| `QISKIT_SEARCH_BASE_URL` | Search API base URL | `https://quantum.cloud.ibm.com/` |

### Optional Configuration

Create a `.env` file in the project directory:

```env
# Optional: Customize documentation URLs
QISKIT_DOCS_BASE=https://quantum.cloud.ibm.com/docs/
QISKIT_HTTP_TIMEOUT=15.0
QISKIT_DOCS_CACHE_TTL=3600.0
QISKIT_SEARCH_CACHE_TTL=300.0
```

## Quick Start

### Running the Server

```bash
uv run qiskit-docs-mcp-server
```

The server will start and listen for MCP connections.

### Using with MCP Clients

#### Claude Desktop Configuration

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "qiskit-docs": {
      "command": "uvx",
      "args": ["qiskit-docs-mcp-server"]
    }
  }
}
```

#### Cline Configuration

Add to your Cline MCP settings:

```json
{
  "mcpServers": {
    "qiskit-docs": {
      "command": "uvx",
      "args": ["qiskit-docs-mcp-server"]
    }
  }
}
```

### LangChain Integration Example

> **Note:** To run LangChain examples you will need to install the dependencies:
> ```bash
> pip install langchain-mcp-adapters langchain-openai langgraph
> ```

```python
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

async def main():
    # Configure MCP client
    async with MultiServerMCPClient(
        {
            "qiskit-docs": {
                "command": "qiskit-docs-mcp-server",
                "args": [],
                "transport": "stdio",
            }
        }
    ) as client:
        # Create agent with LLM and MCP tools
        agent = create_react_agent(
            ChatOpenAI(model="gpt-4o"),
            client.get_tools(),
        )

        # Query documentation
        response = await agent.ainvoke(
            {"messages": [("user", "How do I create a quantum circuit in Qiskit?")]}
        )
        print(response["messages"][-1].content)

asyncio.run(main())
```

## Usage Examples

### Search Documentation

```python
# Search for transpiler information
result = await search_docs_tool("transpiler optimization")
print(f"Found {result['total_results']} results")
for item in result["results"]:
    print(f"- {item['title']}: {item['url']}")
```

### Fetch a Documentation Page

```python
# Fetch circuit module API reference
result = await get_page_tool("api/qiskit/circuit")
print(result["documentation"])

# Fetch with pagination for large pages
result = await get_page_tool("api/qiskit/circuit", max_length=5000)
if result["has_more"]:
    next_page = await get_page_tool("api/qiskit/circuit", offset=result["next_offset"])
```

### Look Up an Error Code

```python
# Look up error code 1002
result = await lookup_error_code_tool("1002")
print(result["details"])
```

## Available Documentation

### SDK Modules

| Module | Description |
|--------|-------------|
| `circuit` | Quantum circuit construction and manipulation |
| `quantum_info` | Quantum information theory utilities |
| `transpiler` | Circuit transpilation and optimization |
| `synthesis` | Circuit synthesis algorithms |
| `dagcircuit` | DAG representation of quantum circuits |
| `passmanager` | Transpiler pass manager framework |
| `converters` | Circuit format converters |
| `compiler` | High-level compilation routines |
| `primitives` | Sampler and Estimator primitives |
| `providers` | Backend providers and job management |
| `result` | Job result handling and analysis |
| `visualization` | Circuit and result visualization |
| `qasm2` | OpenQASM 2.0 support |
| `qasm3` | OpenQASM 3.0 support |
| `qpy` | Qiskit Python serialization format |
| `utils` | General utility functions |
| `exceptions` | Qiskit exception classes |

### Implementation Guides

| Guide | Description |
|-------|-------------|
| `quick-start` | Get started with Qiskit |
| `construct-circuits` | Build and manipulate quantum circuits |
| `transpile` | Transpile circuits for target backends |
| `transpiler-stages` | Understand transpiler pipeline stages |
| `configure-error-mitigation` | Configure error mitigation for primitives |
| `configure-error-suppression` | Configure error suppression techniques |
| `primitives` | Use Sampler and Estimator primitives |
| `execution-modes` | Job, session, and batch execution modes |
| `dynamic-circuits` | Mid-circuit measurements and classical control |
| `functions` | Overview of Qiskit Functions |

> See the `qiskit-docs://guides` resource for the complete list of 30+ available guides.

## Features

### Pagination Support

Large documentation pages are automatically paginated. Use `max_length` and `offset` to control content retrieval:

```python
# Fetch first 5000 characters
result = await get_page_tool("api/qiskit/circuit", max_length=5000)
# result["has_more"] == True, result["next_offset"] == 5000

# Fetch unlimited content
result = await get_page_tool("api/qiskit/circuit", max_length=0)
```

### Metadata Inclusion

All responses include rich metadata:

```python
{
    "status": "success",
    "url": "https://quantum.cloud.ibm.com/docs/api/qiskit/circuit",
    "documentation": "...",
    "has_more": False,
    "total_length": 15420,
    "metadata": {
        "url": "https://quantum.cloud.ibm.com/docs/api/qiskit/circuit",
        "timestamp": "2026-03-03T03:00:00Z",
        "content_type": "markdown",
        "content_length": 15420
    }
}
```

### Dynamic Sitemap Discovery

Resource lists (modules, addons, guides, tutorials, API packages) are automatically discovered from the live documentation sitemap at startup. This means the server adapts to new content without code changes. If the sitemap is unreachable, the server falls back to hardcoded values in `constants.py`.

To update the hardcoded fallback values from the live sitemap:

```bash
cd qiskit-docs-mcp-server
uv run python scripts/update_fallback_constants.py
```

This prints updated constant lists that can be copied into `constants.py`.

### HTML to Markdown Conversion

Documentation is automatically converted from HTML to clean Markdown format, optimized for AI consumption and human readability.

## Development

### Project Structure

```
src/qiskit_docs_mcp_server/
├── server.py           # MCP server definition (tools, resources, prompts)
├── data_fetcher.py     # Business logic for fetching and processing documentation
├── http.py             # HTTP infrastructure: client management, caching, retries
├── sitemap.py          # Dynamic sitemap discovery and page classification
├── html_processing.py  # HTML content extraction and markdown conversion
└── constants.py        # Configuration constants and hardcoded fallback values

scripts/
└── update_fallback_constants.py  # Regenerate fallback values from live sitemap
```

### Running Tests

```bash
# Run all tests
./run_tests.sh

# Or use pytest directly
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- [`langchain_agent.py`](examples/langchain_agent.py) - LangChain agent with multiple LLM providers
- [`langchain_agent.ipynb`](examples/langchain_agent.ipynb) - Interactive Jupyter notebook tutorial
- [`README.md`](examples/README.md) - Detailed examples documentation

## Troubleshooting

### Connection Issues

**Problem**: "Failed to fetch documentation"
**Solution**: Check your internet connection and verify access to https://quantum.cloud.ibm.com/docs/

### Timeout Errors

**Problem**: "Request timed out"
**Solution**: Increase the timeout value:
```bash
export QISKIT_HTTP_TIMEOUT=30.0
```

### Page Not Found

**Problem**: "Failed to fetch" error from `get_page_tool`
**Solution**: Use `search_docs_tool` to find the correct URL, or check the `qiskit-docs://modules` and `qiskit-docs://guides` resources for valid paths

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](../CONTRIBUTING.md) file in the repository root for guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Links

- [Qiskit Documentation](https://quantum.cloud.ibm.com/docs/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [MCP Registry](https://registry.modelcontextprotocol.io/)
- [GitHub Repository](https://github.com/Qiskit/mcp-servers)

## Support

For issues and questions:
- [GitHub Issues](https://github.com/Qiskit/mcp-servers/issues)
- [Qiskit Slack](https://qisk.it/join-slack)
