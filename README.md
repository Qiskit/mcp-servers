# Qiskit MCP Servers

[![Tests](https://github.com/Qiskit/mcp-servers/actions/workflows/test.yml/badge.svg)](https://github.com/Qiskit/mcp-servers/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

[![qiskit-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-mcp-server)
[![qiskit-ibm-runtime-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-ibm-runtime-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-ibm-runtime-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-ibm-runtime-mcp-server)
[![qiskit-ibm-transpiler-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-ibm-transpiler-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-ibm-transpiler-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-ibm-transpiler-mcp-server)
[![qiskit-code-assistant-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-code-assistant-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-code-assistant-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-code-assistant-mcp-server)
[![qiskit-docs-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-docs-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-docs-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-docs-mcp-server)
[![qiskit-gym-mcp-server](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.Qiskit%252Fqiskit-gym-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=qiskit-gym-mcp-server&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.Qiskit%2Fqiskit-gym-mcp-server)

A collection of [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction) servers that give AI assistants, LLMs, and agents seamless access to IBM Quantum services and Qiskit libraries. Generate quantum code, connect to real hardware, analyze backends, execute circuits, and search Qiskit documentation — all through a standardized protocol that works with any MCP-compatible client.

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **IBM Quantum account** and [API token](https://quantum.ibm.com) (only required for IBM cloud servers: Runtime, Transpiler, Code Assistant)

### Install from PyPI

```bash
# Install all servers (core + community)
pip install "qiskit-mcp-servers[all]"

# Install core servers only (default, excludes community)
pip install qiskit-mcp-servers
```

<details>
<summary>Install individual servers</summary>

```bash
pip install "qiskit-mcp-servers[qiskit]"          # Qiskit server only
pip install "qiskit-mcp-servers[code-assistant]"  # Code Assistant server only
pip install "qiskit-mcp-servers[runtime]"         # IBM Runtime server only
pip install "qiskit-mcp-servers[transpiler]"      # IBM Transpiler server only
pip install "qiskit-mcp-servers[docs]"            # Documentation server only
pip install "qiskit-mcp-servers[gym]"             # Qiskit Gym server only (community)
```

</details>

### Install from Source

Each server runs independently — the directory name and command are the same:

```bash
cd qiskit-mcp-server  # or any server directory
pip install -e .
qiskit-mcp-server
```

### Configuration

Servers that connect to IBM Quantum cloud services require a `QISKIT_IBM_TOKEN` environment variable. See each server's README for the full list of environment variables and configuration options.

```bash
export QISKIT_IBM_TOKEN="your_ibm_quantum_token_here"
```

### Client Setup

All servers are compatible with any MCP client. Pick your client below.

#### Claude Code

<video src="docs/videos/QiskitMCPServers-ClaudeCode.mp4" controls width="100%"></video>

```bash
# No auth required
claude mcp add qiskit -- uvx qiskit-mcp-server
claude mcp add qiskit-docs -- uvx qiskit-docs-mcp-server
claude mcp add qiskit-gym -- uvx qiskit-gym-mcp-server

# Require QISKIT_IBM_TOKEN (https://quantum.ibm.com)
claude mcp add qiskit-code-assistant -e QISKIT_IBM_TOKEN=$QISKIT_IBM_TOKEN -- uvx qiskit-code-assistant-mcp-server
claude mcp add qiskit-ibm-runtime -e QISKIT_IBM_TOKEN=$QISKIT_IBM_TOKEN -- uvx qiskit-ibm-runtime-mcp-server
claude mcp add qiskit-ibm-transpiler -e QISKIT_IBM_TOKEN=$QISKIT_IBM_TOKEN -- uvx qiskit-ibm-transpiler-mcp-server
```

#### IBM Bob

<video src="docs/videos/QiskitMCPServers-IBMBob.mp4" controls width="100%"></video>

Add to your `~/.bob/settings/mcp_settings.json`:

```json
{
    "mcpServers": {
        "qiskit": {
            "command": "uvx",
            "args": ["qiskit-mcp-server"],
            "alwaysAllow": [],
            "disabled": false
        },
        "qiskit-docs": {
            "command": "uvx",
            "args": ["qiskit-docs-mcp-server"],
            "alwaysAllow": [],
            "disabled": false
        },
        "qiskit-gym": {
            "command": "uvx",
            "args": ["qiskit-gym-mcp-server"],
            "alwaysAllow": [],
            "disabled": false
        },
        "qiskit-code-assistant": {
            "command": "uvx",
            "args": ["qiskit-code-assistant-mcp-server"],
            "env": {
                "QISKIT_IBM_TOKEN": "<your IBM Quantum token>"
            },
            "alwaysAllow": [],
            "disabled": false
        },
        "qiskit-ibm-runtime": {
            "command": "uvx",
            "args": ["qiskit-ibm-runtime-mcp-server"],
            "env": {
                "QISKIT_IBM_TOKEN": "<your IBM Quantum token>",
                "QISKIT_IBM_RUNTIME_MCP_INSTANCE": "<Optional: Your IBM Quantum instance CRN>"
            },
            "alwaysAllow": [],
            "disabled": false
        },
        "qiskit-ibm-transpiler": {
            "command": "uvx",
            "args": ["qiskit-ibm-transpiler-mcp-server"],
            "env": {
                "QISKIT_IBM_TOKEN": "<your IBM Quantum token>"
            },
            "alwaysAllow": [],
            "disabled": false
        }
    }
}
```

<details>
<summary>Claude Desktop / Cline</summary>

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "qiskit": {
      "command": "uvx",
      "args": ["qiskit-mcp-server"]
    },
    "qiskit-docs": {
      "command": "uvx",
      "args": ["qiskit-docs-mcp-server"]
    },
    "qiskit-gym": {
      "command": "uvx",
      "args": ["qiskit-gym-mcp-server"]
    },
    "qiskit-code-assistant": {
      "command": "uvx",
      "args": ["qiskit-code-assistant-mcp-server"],
      "env": {
        "QISKIT_IBM_TOKEN": "your_ibm_quantum_token_here"
      }
    },
    "qiskit-ibm-runtime": {
      "command": "uvx",
      "args": ["qiskit-ibm-runtime-mcp-server"],
      "env": {
        "QISKIT_IBM_TOKEN": "your_ibm_quantum_token_here"
      }
    },
    "qiskit-ibm-transpiler": {
      "command": "uvx",
      "args": ["qiskit-ibm-transpiler-mcp-server"],
      "env": {
        "QISKIT_IBM_TOKEN": "your_ibm_quantum_token_here"
      }
    }
  }
}
```

</details>

<details>
<summary>MCP Inspector (interactive testing)</summary>

```bash
npx @modelcontextprotocol/inspector uvx qiskit-mcp-server
npx @modelcontextprotocol/inspector uvx qiskit-docs-mcp-server
npx @modelcontextprotocol/inspector uvx qiskit-ibm-runtime-mcp-server
# etc.
```

</details>

## Available Servers

### Core Servers

| Server | Description | Directory |
|--------|-------------|-----------|
| **Qiskit MCP Server** | Circuit creation, transpilation, and serialization (QASM3, QPY) using [Qiskit](https://github.com/Qiskit/qiskit) | [`qiskit-mcp-server/`](./qiskit-mcp-server/) |
| **Qiskit Code Assistant** | AI-assisted quantum programming via [IBM Qiskit Code Assistant](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant) | [`qiskit-code-assistant-mcp-server/`](./qiskit-code-assistant-mcp-server/) |
| **Qiskit IBM Runtime** | Full access to IBM Quantum hardware via [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime/) | [`qiskit-ibm-runtime-mcp-server/`](./qiskit-ibm-runtime-mcp-server/) |
| **Qiskit IBM Transpiler** | AI-optimized circuit routing and optimization via [qiskit-ibm-transpiler](https://github.com/Qiskit/qiskit-ibm-transpiler) | [`qiskit-ibm-transpiler-mcp-server/`](./qiskit-ibm-transpiler-mcp-server/) |
| **Qiskit Docs** | Search and retrieve [Qiskit documentation](https://quantum.cloud.ibm.com/docs/), guides, and API references. No auth required. | [`qiskit-docs-mcp-server/`](./qiskit-docs-mcp-server/) |

### Community Servers

| Server | Description | Directory |
|--------|-------------|-----------|
| **Qiskit Gym** | RL-based quantum circuit synthesis using [qiskit-gym](https://github.com/rl-institut/qiskit-gym) (permutation routing, linear functions, Clifford circuits) | [`qiskit-gym-mcp-server/`](./qiskit-gym-mcp-server/) |

## Examples

Each server includes an `examples/` directory with a **Jupyter notebook** (`langchain_agent.ipynb`) and a **Python script** (`langchain_agent.py`) showing how to build AI agents with LangChain.

The root [`examples/`](./examples/) directory contains a multi-agent **Quantum Volume Finder** that orchestrates multiple MCP servers to find the highest achievable Quantum Volume on IBM Quantum hardware. See the [examples README](./examples/README.md) for details.

## Architecture

All servers are built on [FastMCP](https://github.com/jlowin/fastmcp) and share a consistent structure:

- **Async-first** — all tool and resource handlers are async, using FastMCP's native async support
- **Standalone packages** — each server is an independent PyPI package with its own `pyproject.toml`, tests, and CI
- **MCP Registry** — every server ships a [`server.json`](https://registry.modelcontextprotocol.io/) manifest for automatic discovery
- **Unified meta-package** — `qiskit-mcp-servers` installs any combination via pip extras
- **Full MCP protocol** — tools (quantum operations), resources (backend info, service status), and stdio transport

## Development

### Running Tests
```bash
cd qiskit-code-assistant-mcp-server  # or any server directory
./run_tests.sh
```

### Code Quality
- **Linting**: `ruff check` and `ruff format`
- **Type checking**: `mypy src/`
- **Testing**: `pytest` with async support and coverage reporting
- **CI/CD**: GitHub Actions for automated testing

## Resources

- [Model Context Protocol](https://modelcontextprotocol.io/introduction) — understanding MCP
- [Qiskit IBM Runtime docs](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime) — quantum cloud services
- [Qiskit Code Assistant docs](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant) — AI code assistance
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — interactive testing tool
- [FastMCP](https://github.com/jlowin/fastmcp) — high-performance MCP framework
- [AGENTS.md](AGENTS.md) — guidance for AI coding assistants (IBM Bob, Claude Code, Copilot, Cursor, and others)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) guide for details.

## License

This project is licensed under the **Apache License 2.0**.
