# Qiskit MCP Servers

[![Tests](https://github.com/Qiskit/mcp-servers/actions/workflows/test.yml/badge.svg)](https://github.com/Qiskit/mcp-servers/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

A collection of **Model Context Protocol (MCP)** servers that provide AI assistants, LLMs, and agents with seamless access to IBM Quantum services and Qiskit libraries for quantum computing development and research.

## 🌟 What is This?

This repository contains production-ready MCP servers that enable AI systems to interact with quantum computing resources through Qiskit. Instead of manually configuring quantum backends, writing boilerplate code, or managing IBM Quantum accounts, AI assistants can now:

- 🤖 **Generate intelligent quantum code** with context-aware suggestions
- 🔌 **Connect to real quantum hardware** automatically  
- 📊 **Analyze quantum backends** and find optimal resources
- 🚀 **Execute quantum circuits** and monitor job status
- 💡 **Provide quantum computing assistance** with expert knowledge

## 🛠️ Available Servers

### 🔬 Qiskit MCP Server
**Core Qiskit quantum computing capabilities**

Provides quantum circuit creation, manipulation, transpilation, and serialization utilities (QASM3, QPY) for local quantum development using [Qiskit](https://github.com/Qiskit/qiskit)

**📁 Directory**: [`./qiskit-mcp-server/`](./qiskit-mcp-server/)

---

### 🧠 Qiskit Code Assistant MCP Server
**Intelligent quantum code completion and assistance**

Provides access to [IBM's Qiskit Code Assistant](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant) for AI-assisted quantum programming

**📁 Directory**: [`./qiskit-code-assistant-mcp-server/`](./qiskit-code-assistant-mcp-server/)

---

### ⚙️ Qiskit IBM Runtime MCP Server
**Complete access to IBM Quantum cloud services**

Comprehensive interface to IBM Quantum hardware via [Qiskit IBM Runtime](https://github.com/Qiskit/qiskit-ibm-runtime/)

**📁 Directory**: [`./qiskit-ibm-runtime-mcp-server/`](./qiskit-ibm-runtime-mcp-server/)

---

### 🚀 Qiskit IBM Transpiler MCP Server
**AI-powered circuit transpilation**

Access to the [qiskit-ibm-transpiler](https://github.com/Qiskit/qiskit-ibm-transpiler) library for AI-optimized circuit routing and optimization.

**📁 Directory**: [`./qiskit-ibm-transpiler-mcp-server/`](./qiskit-ibm-transpiler-mcp-server/)

## 📚 Examples

Each MCP server includes example code demonstrating how to build AI agents using LangChain:

| Server | Examples |
|--------|----------|
| Qiskit MCP Server | [`qiskit-mcp-server/examples/`](./qiskit-mcp-server/examples/) |
| Qiskit Code Assistant MCP Server | [`qiskit-code-assistant-mcp-server/examples/`](./qiskit-code-assistant-mcp-server/examples/) |
| Qiskit IBM Runtime MCP Server | [`qiskit-ibm-runtime-mcp-server/examples/`](./qiskit-ibm-runtime-mcp-server/examples/) |
| Qiskit IBM Transpiler MCP Server | [`qiskit-ibm-transpiler-mcp-server/examples/`](./qiskit-ibm-transpiler-mcp-server/examples/) |

Each examples directory contains:
- **Jupyter Notebook** (`langchain_agent.ipynb`) - Interactive tutorial with step-by-step examples
- **Python Script** (`langchain_agent.py`) - Command-line agent with multiple LLM provider support

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **[uv](https://astral.sh/uv)** package manager (fastest Python package manager)
- **IBM Quantum account** and API token
- **Qiskit Code Assistant access** (for code assistant server)

### Installation

#### Install from PyPI

```bash
# Install all MCP servers at once
pip install qiskit-mcp-servers

# Or install individual servers
pip install qiskit-mcp-server
pip install qiskit-code-assistant-mcp-server
pip install qiskit-ibm-runtime-mcp-server
pip install qiskit-ibm-transpiler-mcp-server
```

#### Install from Source

Each server is designed to run independently. Choose the server you need:

#### 🔬 Qiskit Server
```bash
cd qiskit-mcp-server
uv run qiskit-mcp-server
```

#### 🧠 Qiskit Code Assistant Server
```bash
cd qiskit-code-assistant-mcp-server
uv run qiskit-code-assistant-mcp-server
```

#### ⚙️ IBM Runtime Server
```bash
cd qiskit-ibm-runtime-mcp-server
uv run qiskit-ibm-runtime-mcp-server
```

#### 🚀 IBM Transpiler Server
```bash
cd qiskit-ibm-transpiler-mcp-server
uv run qiskit-ibm-transpiler-mcp-server
```

### 🔧 Configuration

#### Environment Variables
```bash
# For IBM Runtime Server
export QISKIT_IBM_TOKEN="your_ibm_quantum_token_here"

# For Code Assistant Server  
export QISKIT_IBM_TOKEN="your_ibm_quantum_token_here"
export QCA_TOOL_API_BASE="https://qiskit-code-assistant.quantum.ibm.com"
```

#### Using with MCP Clients

Both servers are compatible with any MCP client. Test interactively with MCP Inspector:

```bash
# Test Code Assistant Server
npx @modelcontextprotocol/inspector uv run qiskit-code-assistant-mcp-server

# Test IBM Runtime Server
npx @modelcontextprotocol/inspector uv run qiskit-ibm-runtime-mcp-server
```

## 🏗️ Architecture & Design

### 🎯 Unified Design Principles

Both servers follow a **consistent, production-ready architecture**:

- **🔄 Async-first**: Built with FastMCP for high-performance async operations
- **🧪 Test-driven**: Comprehensive test suites with 65%+ coverage
- **🛡️ Type-safe**: Full mypy type checking and validation
- **📦 Modern packaging**: Standard `pyproject.toml` with hatchling build system
- **🔧 Developer-friendly**: Automated formatting (ruff), linting, and CI/CD

### 🔌 MCP Protocol Support

Both servers implement the full **Model Context Protocol specification**:

- **🛠️ Tools**: Execute quantum operations (code completion, job submission, backend queries)
- **📚 Resources**: Access quantum data (service status, backend information, model details)
- **⚡ Real-time**: Async operations for responsive AI interactions
- **🔒 Secure**: Proper authentication and error handling

## 🧪 Development

### 🏃‍♂️ Running Tests
```bash
# Run tests for Code Assistant server
cd qiskit-code-assistant-mcp-server
./run_tests.sh

# Run tests for IBM Runtime server  
cd qiskit-ibm-runtime-mcp-server
./run_tests.sh
```

### 🔍 Code Quality
Both servers maintain high code quality standards:
- **✅ Linting**: `ruff check` and `ruff format`  
- **🛡️ Type checking**: `mypy src/`
- **🧪 Testing**: `pytest` with async support and coverage reporting
- **🚀 CI/CD**: GitHub Actions for automated testing

## 📖 Resources & Documentation

### 🔗 Essential Links
- **[Model Context Protocol](https://modelcontextprotocol.io/introduction)** - Understanding MCP
- **[Qiskit IBM Runtime](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime)** - Quantum cloud services
- **[Qiskit Code Assistant](https://quantum.cloud.ibm.com/docs/en/guides/qiskit-code-assistant)** - AI code assistance  
- **[MCP Inspector](https://github.com/modelcontextprotocol/inspector)** - Interactive testing tool
- **[FastMCP](https://github.com/jlowin/fastmcp)** - High-performance MCP framework

### AI Development Assistant Support

This repository includes AI-generated code and offers comprehensive guidance for AI coding assistants (like [IBM Bob](https://www.ibm.com/products/bob), Claude Code, GitHub Copilot, Cursor AI, and others) in [AGENTS.md](AGENTS.md). This helps AI assistants provide more accurate, context-aware suggestions when working with this codebase. 

## 📄 License

This project is licensed under the **Apache License 2.0**.
