# Qiskit MCP Servers

A collection of **Model Context Protocol (MCP)** servers that provide AI assistants, LLMs, and agents with seamless access to IBM Quantum services and Qiskit libraries for quantum computing development and research.

## 🌟 What is This?

This repository contains production-ready MCP servers that enable AI systems to interact with quantum computing resources through Qiskit. Instead of manually configuring quantum backends, writing boilerplate code, or managing IBM Quantum accounts, AI assistants can now:

- 🤖 **Generate intelligent quantum code** with context-aware suggestions
- 🔌 **Connect to real quantum hardware** automatically  
- 📊 **Analyze quantum backends** and find optimal resources
- 🚀 **Execute quantum circuits** and monitor job status
- 💡 **Provide quantum computing assistance** with expert knowledge

## 🛠️ Available Servers

### 🧠 Qiskit Code Assistant MCP Server
**Intelligent quantum code completion and assistance**

Provides access to IBM's Qiskit Code Assistant AI for intelligent quantum programming

**📁 Directory**: [`./qiskit-code-assistant-mcp-server/`](./qiskit-code-assistant-mcp-server/)

---

### ⚙️ Qiskit IBM Runtime MCP Server  
**Complete access to IBM Quantum cloud services**

Comprehensive interface to IBM Quantum hardware via Qiskit IBM Runtime

**📁 Directory**: [`./qiskit-ibm-runtime-mcp-server/`](./qiskit-ibm-runtime-mcp-server/)

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **[uv](https://astral.sh/uv)** package manager (fastest Python package manager)
- **IBM Quantum account** and API token
- **Qiskit Code Assistant access** (for code assistant server)

### Installation & Usage

Each server is designed to run independently. Choose the server you need:

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
