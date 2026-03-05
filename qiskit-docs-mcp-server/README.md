# qiskit-docs-mcp-server

[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-docs-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-docs-mcp-server)

<!-- mcp-name: io.github.Qiskit/qiskit-docs-mcp-server -->

MCP server for querying and retrieving Qiskit documentation, guides, and API references.

## Overview

The Qiskit Documentation MCP Server provides AI assistants and agents with seamless access to the complete Qiskit documentation ecosystem. It enables intelligent retrieval of SDK module documentation, implementation guides, and best practices through a standardized Model Context Protocol interface.

### Key Features

- **📚 Complete Documentation Access**: Query all Qiskit SDK modules (circuit, primitives, transpiler, quantum_info, result, visualization)
- **📖 Implementation Guides**: Access best practices for optimization, error mitigation, dynamic circuits, and more
- **🔍 Smart Search**: Search across the entire Qiskit documentation with fuzzy matching
- **🎯 No Authentication Required**: Public documentation access without API tokens
- **📝 Markdown Output**: Clean, formatted documentation ready for AI consumption
- **⚡ Fast Retrieval**: Efficient HTTP-based documentation fetching with configurable timeouts

## Components

### Tools

The server implements three tools for documentation access:

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_sdk_module_docs_tool` | Get documentation for Qiskit SDK modules | `module`: Module name (circuit, primitives, transpiler, quantum_info, result, visualization) |
| `get_guide_tool` | Get Qiskit implementation guides and best practices | `guide`: Guide name (optimization, quantum-circuits, error-mitigation, dynamic-circuits, parametric-compilation, performance-tuning) |
| `search_docs_tool` | Search Qiskit documentation for relevant content | `query`: Search query string<br>`module`: Search scope (default: "documentation") |

### Resources

The server provides three resources for listing available documentation:

| Resource URI | Description |
|--------------|-------------|
| `qiskit-docs://modules` | List of all Qiskit SDK modules with descriptions |
| `qiskit-docs://addons` | List of Qiskit addon modules and tutorials |
| `qiskit-docs://guides` | List of implementation guides and best practices |

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
| `QISKIT_SDK_DOCS` | SDK documentation URL | `https://quantum.cloud.ibm.com/docs/` |
| `QISKIT_HTTP_TIMEOUT` | HTTP request timeout in seconds | `10.0` |
| `QISKIT_SEARCH_BASE_URL` | Search API base URL | `https://quantum.cloud.ibm.com/` |

### Optional Configuration

Create a `.env` file in the project directory:

```env
# Optional: Customize documentation URLs
QISKIT_DOCS_BASE=https://quantum.cloud.ibm.com/docs/
QISKIT_HTTP_TIMEOUT=15.0
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
> pip install langchain langchain-mcp-adapters langchain-openai python-dotenv
> ```

```python
import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

async def main():
    # Configure MCP client
    mcp_client = MultiServerMCPClient({
        "qiskit-docs": {
            "transport": "stdio",
            "command": "qiskit-docs-mcp-server",
            "args": [],
        }
    })

    # Use persistent session for efficient tool calls
    async with mcp_client.session("qiskit-docs") as session:
        # Load MCP tools
        tools = await load_mcp_tools(session)

        # Create agent with LLM
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent = create_agent(
            llm,
            tools,
            system_prompt="You are a helpful quantum computing documentation assistant."
        )

        # Query documentation
        response = await agent.ainvoke({
            "messages": [("user", "How do I create a quantum circuit in Qiskit?")]
        })
        print(response["messages"][-1].content)

asyncio.run(main())
```

## Usage Examples

### Get SDK Module Documentation

```python
# Query circuit module documentation
result = await get_sdk_module_docs_tool("circuit")
print(result["documentation"])
```

### Get Implementation Guide

```python
# Get error mitigation guide
result = await get_guide_tool("error-mitigation")
print(result["documentation"])
```

### Search Documentation

```python
# Search for transpiler information
result = await search_docs_tool("transpiler optimization")
print(f"Found {result['total_results']} results")
for item in result["results"]:
    print(f"- {item['name']}: {item['url']}")
```

## Available Documentation

### SDK Modules

| Module | Description |
|--------|-------------|
| `circuit` | Quantum circuit construction and manipulation |
| `primitives` | Sampler and Estimator primitives for quantum execution |
| `transpiler` | Circuit transpilation and optimization |
| `quantum_info` | Quantum information theory utilities |
| `result` | Job result handling and analysis |
| `visualization` | Circuit and result visualization tools |

### Implementation Guides

| Guide | Description |
|-------|-------------|
| `optimization` | Quantum optimization techniques and algorithms |
| `quantum-circuits` | Circuit design patterns and best practices |
| `error-mitigation` | Error mitigation strategies for noisy quantum devices |
| `dynamic-circuits` | Mid-circuit measurements and classical control |
| `parametric-compilation` | Parameterized circuit compilation techniques |
| `performance-tuning` | Performance optimization tips and tricks |

## Features

### Fuzzy Matching

The server includes intelligent fuzzy matching for module and guide names:

```python
# Typo in module name - server suggests correct spelling
result = await get_sdk_module_docs_tool("circuitt")
# Returns: {"status": "error", "suggestions": ["circuit"]}
```

### Metadata Inclusion

All responses include rich metadata:

```python
{
    "status": "success",
    "module": "circuit",
    "documentation": "...",
    "metadata": {
        "url": "https://quantum.cloud.ibm.com/docs/api/qiskit/circuit",
        "timestamp": "2026-03-03T03:00:00Z",
        "content_type": "markdown",
        "content_length": 15420
    }
}
```

### HTML to Markdown Conversion

Documentation is automatically converted from HTML to clean Markdown format, optimized for AI consumption and human readability.

## Development

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

### Module Not Found

**Problem**: "Module 'xyz' not found"
**Solution**: Check available modules using the `qiskit-docs://modules` resource or see the Available Documentation section above

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
