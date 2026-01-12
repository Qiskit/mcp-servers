# AGENTS.md

This file provides guidance to AI development assistants when working with code in this repository.

**Supported AI Assistants:**
- IBM Bob
- Claude Code
- GitHub Copilot
- Cursor AI
- Windsurf
- Gemini CLI
- Any AI assistant with codebase context awareness

## Project Overview

qiskit-mcp-servers is a collection of **Model Context Protocol (MCP)** servers that provide AI assistants, LLMs, and agents with seamless access to IBM Quantum services and Qiskit libraries for quantum computing development and research.

### Core Purpose
- Enable AI systems to interact with quantum computing resources through Qiskit
- Provide production-ready MCP servers for quantum computing workflows
- Connect AI assistants to real quantum hardware automatically
- Simplify quantum circuit execution and job management
- Provide intelligent quantum code completion and assistance
- Enable AI-powered quantum circuit optimization and transpilation

### Key Technologies
- **Protocol**: Model Context Protocol (MCP)
- **Language**: Python 3.10+ (3.11+ recommended)
- **Framework**: FastMCP (async-first MCP framework)
- **Package Manager**: uv (modern Python package manager with workspace support)
- **Testing**: pytest with async support, 65%+ coverage
- **Code Quality**: ruff (formatting + linting), mypy (type checking)
- **Build System**: hatchling with pyproject.toml

## Architecture

### Repository Structure

This is a **monorepo** using uv workspace containing four independent MCP servers:

```
qiskit-mcp-servers/
├── qiskit-mcp-server/                       # Core Qiskit transpilation
├── qiskit-code-assistant-mcp-server/        # AI code completion
├── qiskit-ibm-runtime-mcp-server/           # IBM Quantum cloud services
├── qiskit-ibm-transpiler-mcp-server/        # AI-powered transpilation
├── .claude/skills/                          # Claude Code skills
├── .github/                                 # GitHub templates and workflows
├── pyproject.toml                           # Workspace configuration & meta-package
├── mypy.ini                                 # Shared mypy configuration
├── ruff.toml                                # Shared ruff configuration
├── README.md                                # Main repository documentation
├── PUBLISHING.md                            # PyPI publishing guide
├── CONTRIBUTING.md                          # Contribution guidelines
├── CODE_OF_CONDUCT.md                       # Community guidelines
└── LICENSE                                  # Apache 2.0 license
```

Each server is:
- **Independent**: Can be installed and run separately
- **Self-contained**: Has its own dependencies and tests
- **Publishable**: Separate PyPI packages
- **Consistent**: Follows unified design principles

### Workspace Configuration

The root `pyproject.toml` defines a uv workspace:

```toml
[tool.uv.workspace]
members = [
    "qiskit-mcp-server",
    "qiskit-code-assistant-mcp-server",
    "qiskit-ibm-runtime-mcp-server",
    "qiskit-ibm-transpiler-mcp-server",
]
```

The root package is also a **meta-package** that installs all servers:
```bash
pip install qiskit-mcp-servers  # Installs all four servers
```

### Component Structure

Each MCP server follows this standard structure:

```
<server-name>/
├── src/
│   └── <package_name>/
│       ├── __init__.py          # Main entry point
│       ├── server.py            # FastMCP server definition
│       ├── <core>.py            # Core functionality (async)
│       └── utils.py             # Utilities (optional)
├── tests/
│   ├── conftest.py              # Test fixtures
│   ├── unit/                    # Unit tests (optional subdirectory)
│   ├── integration/             # Integration tests (optional subdirectory)
│   └── test_*.py                # Test files
├── examples/
│   ├── README.md                # MCP server example documentation
│   ├── langchain_agent.ipynb    # Interactive tutorial with step-by-step examples
│   └── langchain_agent.py       # Command-line agent with multiple LLM provider support
├── pyproject.toml               # Project metadata & dependencies
├── pytest.ini                   # pytest configuration (optional)
├── LICENSE                      # Apache 2.0 license (copy from root)
├── README.md                    # Server-specific documentation
├── .env.example                 # Environment variable template (optional)
└── run_tests.sh                 # Test execution script
```

## Key Components

### 1. Qiskit MCP Server

**Purpose**: Core quantum circuit transpilation using Qiskit pass managers

**Directory**: [`qiskit-mcp-server/`](qiskit-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `transpiler.py`: Qiskit transpilation functions (async)
- `circuit_serialization.py`: QASM3/QPY conversion utilities

**Tools Provided**:
| Tool | Description |
|------|-------------|
| `transpile_circuit_tool` | Transpile circuit with configurable optimization (0-3) |
| `analyze_circuit_tool` | Analyze circuit structure without transpiling |
| `compare_optimization_levels_tool` | Compare all optimization levels (0-3) |
| `convert_qpy_to_qasm3_tool` | Convert QPY to human-readable QASM3 |
| `convert_qasm3_to_qpy_tool` | Convert QASM3 to base64-encoded QPY |

**Resources Provided**:
| Resource URI | Description |
|--------------|-------------|
| `qiskit://transpiler/info` | Transpiler capabilities and documentation |
| `qiskit://transpiler/basis-gates` | Available basis gate presets |
| `qiskit://transpiler/topologies` | Available coupling map topologies |

**Environment Variables**:
- `QISKIT_MCP_MAX_QUBITS`: Maximum allowed qubits (default: 100)
- `QISKIT_MCP_MAX_GATES`: Maximum allowed gates (default: 10000)

---

### 2. Qiskit Code Assistant MCP Server

**Purpose**: Intelligent quantum code completion and assistance

**Directory**: [`qiskit-code-assistant-mcp-server/`](qiskit-code-assistant-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `qca.py`: Qiskit Code Assistant API integration (async)
- `constants.py`: API endpoints and configuration
- `utils.py`: HTTP client management and utilities

**Tools Provided**:
| Tool | Description |
|------|-------------|
| `get_completion_tool` | Get code completion for quantum code prompts |
| `get_rag_completion_tool` | RAG-based completion with documentation context |
| `accept_completion_tool` | Mark a completion as accepted (telemetry) |
| `accept_model_disclaimer_tool` | Accept disclaimer for a model |

**Resources Provided**:
| Resource URI | Description |
|--------------|-------------|
| `qca://status` | Service status and connection info |
| `qca://models` | List available models |
| `qca://model/{model_id}` | Specific model information |
| `qca://disclaimer/{model_id}` | Model disclaimer information |

**Environment Variables**:
- `QISKIT_IBM_TOKEN`: IBM Quantum API token (required)
- `QCA_TOOL_API_BASE`: API base URL (default: https://qiskit-code-assistant.quantum.ibm.com)
- `QCA_TOOL_MODEL_NAME`: Model to use for completions
- `QCA_MCP_DEBUG_LEVEL`: Logging level (default: INFO)

---

### 3. Qiskit IBM Runtime MCP Server

**Purpose**: Complete access to IBM Quantum cloud services

**Directory**: [`qiskit-ibm-runtime-mcp-server/`](qiskit-ibm-runtime-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `ibm_runtime.py`: Qiskit IBM Runtime integration (async)

**Tools Provided**:
| Tool | Description |
|------|-------------|
| `setup_ibm_quantum_account_tool` | Configure IBM Quantum account |
| `list_backends_tool` | Get available quantum backends |
| `least_busy_backend_tool` | Find least busy operational backend |
| `get_backend_properties_tool` | Get detailed backend properties |
| `get_backend_calibration_tool` | Get calibration data (T1, T2, error rates) |
| `list_my_jobs_tool` | List recent jobs |
| `get_job_status_tool` | Check job status |
| `cancel_job_tool` | Cancel a running/queued job |

**Resources Provided**:
| Resource URI | Description |
|--------------|-------------|
| `ibm://status` | Service status and connection info |

**Environment Variables**:
- `QISKIT_IBM_TOKEN`: IBM Quantum API token (optional, can use saved credentials)

**Credential Resolution Priority**:
1. Explicit token passed to `setup_ibm_quantum_account()`
2. `QISKIT_IBM_TOKEN` environment variable
3. Saved credentials in `~/.qiskit/qiskit-ibm.json`

---

### 4. Qiskit IBM Transpiler MCP Server

**Purpose**: AI-powered circuit transpilation with routing and synthesis

**Directory**: [`qiskit-ibm-transpiler-mcp-server/`](qiskit-ibm-transpiler-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `qta.py`: AI transpilation functions (async)
- `utils.py`: Account setup and circuit format utilities

**Tools Provided**:
| Tool | Description |
|------|-------------|
| `setup_ibm_quantum_account_tool` | Configure IBM Quantum account |
| `ai_routing_tool` | AI-powered circuit routing with SWAP insertion |
| `ai_clifford_synthesis_tool` | AI synthesis for Clifford circuits (H, S, CX; up to 9 qubits) |
| `ai_linear_function_synthesis_tool` | AI synthesis for Linear Function circuits (CX, SWAP; up to 9 qubits) |
| `ai_permutation_synthesis_tool` | AI synthesis for Permutation circuits (SWAP; 27, 33, 65 qubits) |
| `ai_pauli_network_synthesis_tool` | AI synthesis for Pauli Network circuits (up to 6 qubits) |
| `hybrid_ai_transpile_tool` | End-to-end hybrid transpilation combining Qiskit heuristics with AI passes |

**Environment Variables**:
- `QISKIT_IBM_TOKEN`: IBM Quantum API token (required)

**Circuit Format Support**:
- Input: QASM 3.0 string or base64-encoded QPY
- Output: Base64-encoded QPY (for precision when chaining tools)

---

## Data Flow

### Qiskit MCP Server
```
AI Assistant → MCP Client → transpile_circuit_tool
                                  ↓
                         transpiler.py (async functions)
                                  ↓
                         Qiskit preset pass managers
                                  ↓
                         Transpiled circuit (QPY format)
```

### Qiskit Code Assistant Server
```
AI Assistant → MCP Client → get_completion_tool
                                  ↓
                            qca.py (async functions)
                                  ↓
                    IBM Qiskit Code Assistant API
                                  ↓
                        Code completion response
```

### IBM Runtime Server
```
AI Assistant → MCP Client → setup_ibm_quantum_account tool
                                  ↓
                       ibm_runtime.py (async functions)
                                  ↓
                         QiskitRuntimeService
                                  ↓
                    Backend info / Job management / Results
```

### IBM Transpiler Server
```
AI Assistant → MCP Client → ai_routing_tool / ai_*_synthesis_tool
                                  ↓
                            qta.py (async functions)
                                  ↓
                         qiskit-ibm-transpiler (AI passes)
                                  ↓
                    Optimized circuit (QPY format)
```

## Development Guidelines

### Environment Setup

1. **Prerequisites**:
   - Python 3.10+ (3.11+ recommended)
   - [uv](https://astral.sh/uv) package manager
   - IBM Quantum account and API token
   - Git

2. **Installation**:
   ```bash
   # Clone the repository
   git clone https://github.com/Qiskit/mcp-servers.git
   cd mcp-servers

   # Install all workspace dependencies
   uv sync

   # Or navigate to specific server
   cd qiskit-ibm-runtime-mcp-server
   uv sync
   ```

3. **Configuration**:
   ```bash
   # Copy environment template
   cp .env.example .env

   # Edit .env and add your IBM Quantum API token
   # Get token from: https://quantum.cloud.ibm.com/
   ```

4. **Running from Source**:
   ```bash
   # Run specific servers
   uv run qiskit-mcp-server
   uv run qiskit-code-assistant-mcp-server
   uv run qiskit-ibm-runtime-mcp-server
   uv run qiskit-ibm-transpiler-mcp-server
   ```

5. **Interactive Testing**:
   ```bash
   # Test with MCP Inspector (requires Node.js)
   npx @modelcontextprotocol/inspector uv run qiskit-mcp-server
   ```

### Code Conventions

1. **Python Standards**:
   - Python 3.10+ features allowed
   - Async/await preferred for MCP operations
   - Type hints required (mypy strict mode)
   - Naming: snake_case for functions/variables, PascalCase for classes
   - Docstrings for public functions (Google style)

2. **License Header** (required for all new files):
   ```python
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
   ```

3. **MCP Server Patterns**:
   - All servers use FastMCP framework
   - Tools defined with `@mcp.tool()` decorator
   - Resources defined with `@mcp.resource()` decorator
   - Async functions for all MCP handlers
   - Tool functions should delegate to core module functions
   - Tool function names end with `_tool` suffix
   - Return type is `dict[str, Any]`

4. **Error Handling**:
   - Return `{"status": "error", "message": "..."}` for errors
   - Return `{"status": "success", ...}` for success
   - Provide clear error messages
   - Log errors for debugging
   - Handle network failures gracefully
   - Validate inputs before API calls

5. **Testing**:
   - Write tests in `tests/` directory
   - Use pytest with async support (`pytest-asyncio`)
   - Mock external APIs (`pytest-mock`, `respx` for HTTP)
   - Target 65%+ code coverage
   - Never call real IBM Quantum APIs in unit tests
   - Run tests: `./run_tests.sh` or `uv run pytest`

6. **Code Quality**:
   - Format with `ruff format`
   - Lint with `ruff check`
   - Type check with `mypy src/`
   - All checks must pass before committing

### Security Best Practices

When developing MCP servers that handle quantum computing resources:

1. **Credential Management**:
   - Never hardcode API tokens or credentials in source code
   - Use environment variables (`QISKIT_IBM_TOKEN`) or secure credential files
   - Support fallback to saved credentials (`~/.qiskit/qiskit-ibm.json`)
   - Never log or expose credentials in error messages or debug output

2. **Input Validation**:
   - Validate all circuit inputs before processing (size, format, qubit count)
   - Enforce configurable limits (`QISKIT_MCP_MAX_QUBITS`, `QISKIT_MCP_MAX_GATES`)
   - Sanitize QASM strings before parsing
   - Reject malformed or suspicious inputs early

3. **API Security**:
   - Use HTTPS for all external API calls
   - Implement proper error handling without leaking sensitive information
   - Handle authentication failures gracefully
   - Rate limit awareness for IBM Quantum API calls

4. **Testing Security**:
   - Never use real credentials in unit tests
   - Mock all external service calls
   - Test error paths and edge cases
   - Verify credential handling doesn't expose sensitive data


### Testing Strategy

Comprehensive testing ensures MCP server reliability:

1. **Test Organization**:
   ```
   tests/
   ├── conftest.py           # Shared fixtures and mocks
   ├── unit/                  # Fast, isolated unit tests
   ├── integration/           # Tests with mocked external services
   └── test_*.py              # Test files (pytest auto-discovery)
   ```

2. **Unit Tests**:
   - Test individual functions in isolation
   - Mock all external dependencies
   - Fast execution (no network calls)
   - Target 65%+ code coverage

3. **Integration Tests** (marked with `@pytest.mark.integration`):
   - Test tool and resource interactions
   - Use mocked IBM Quantum services
   - Verify end-to-end data flow
   - Can be skipped with `pytest -m "not integration"`

4. **Common Fixtures** (in `conftest.py`):
   ```python
   @pytest.fixture
   def mock_runtime_service():
       """Mock QiskitRuntimeService with fake backends and jobs."""
       ...

   @pytest.fixture
   def mock_env_vars():
       """Set test environment variables."""
       ...

   @pytest.fixture(autouse=True)
   def reset_service():
       """Reset global service state between tests."""
       ...
   ```

5. **Async Testing Pattern**:
   ```python
   @pytest.mark.asyncio
   async def test_async_tool(mock_service):
       """Test async MCP tool."""
       result = await my_tool_function(params)
       assert result["status"] == "success"
   ```

6. **Running Tests**:
   ```bash
   # All tests
   ./run_tests.sh

   # Unit tests only
   uv run pytest -m "not integration"

   # With coverage
   uv run pytest --cov=src --cov-report=html

   # Specific test file
   uv run pytest tests/test_server.py -v
   ```

### Adding New Features

1. **Adding a New Tool**:
   ```python
   # In server.py
   @mcp.tool()
   async def my_new_tool(param: str, optional_param: int = 10) -> dict[str, Any]:
       """Tool description for AI assistant.

       Args:
           param: Description of the parameter
           optional_param: Description with default behavior

       Returns:
           Dictionary with result data
       """
       return await my_core_function(param, optional_param)
   ```

2. **Adding a New Resource**:
   ```python
   # In server.py
   @mcp.resource("protocol://path", mime_type="application/json")
   async def my_resource() -> dict[str, Any]:
       """Resource description."""
       return await get_resource_data()
   ```

3. **Adding Tests**:
   ```python
   # In tests/test_*.py
   import pytest
   from unittest.mock import Mock, patch

   @pytest.mark.asyncio
   async def test_my_tool(mock_service_fixture):
       """Test description."""
       with patch("my_package.module.ExternalService", return_value=mock_service_fixture):
           result = await my_function()
           assert result["status"] == "success"
   ```

4. **Adding a New Server**:
   - Create new directory: `qiskit-<name>-mcp-server/`
   - Copy structure from existing server
   - Create `pyproject.toml` with unique package name
   - Add to workspace members in root `pyproject.toml`
   - Implement server using FastMCP
   - Add comprehensive tests
   - Document in server-specific README.md
   - Update main README.md with new server info
   - **Update GitHub CI/CD** (see below)

### GitHub Integration for New Servers

When adding a new MCP server, you must update the following GitHub configurations:

1. **Update `.github/workflows/test.yml`**:
   - Add server to the `lint` job's install, ruff check, ruff format, mypy, and bandit steps
   - Create a new `test-<name>` job (copy from existing server job):
     ```yaml
     test-<name>:
       runs-on: ubuntu-latest
       needs: lint
       strategy:
         matrix:
           python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]
       steps:
       - uses: actions/checkout@v4
       - name: Install uv
         uses: astral-sh/setup-uv@v3
         with:
           version: "latest"
           enable-cache: true
       - name: Set up Python ${{ matrix.python-version }}
         run: uv python install ${{ matrix.python-version }}
       - name: Install dependencies
         working-directory: ./qiskit-<name>-mcp-server
         run: uv sync --group dev --group test
       - name: Run tests
         working-directory: ./qiskit-<name>-mcp-server
         env:
           QISKIT_IBM_TOKEN: ${{ secrets.QISKIT_IBM_TOKEN }}
         run: uv run pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
       - name: Upload coverage
         uses: codecov/codecov-action@v3
         with:
           file: ./qiskit-<name>-mcp-server/coverage.xml
           flags: <name>
           name: <name>-coverage
     ```

2. **Update `.github/workflows/publish-pypi.yml`**:
   - Add new option to `workflow_dispatch` inputs
   - Create a new `publish-<name>` job (copy from existing)
   - Use release tag pattern: `<name>` (e.g., releases tagged `transpiler-v0.1.0`)
   - Add to `publish-meta-package` job's `needs` array

3. **Update `.github/CODEOWNERS`** (if server has specific owners):
   ```
   /qiskit-<name>-mcp-server/ @owner1 @owner2
   ```

4. **Release Tag Naming Convention**:
   | Server | Release Tag Pattern |
   |--------|---------------------|
   | qiskit-mcp-server | `qiskit-v*` |
   | qiskit-code-assistant-mcp-server | `code-assistant*` |
   | qiskit-ibm-runtime-mcp-server | `runtime*` |
   | qiskit-ibm-transpiler-mcp-server | `transpiler*` |
   | Meta-package | `meta*` |

5. **GitHub Secrets Required**:
   - `QISKIT_IBM_TOKEN`: IBM Quantum API token for integration tests
   - PyPI trusted publishing is configured (no token needed for publish)

## Common Tasks

### Building and Testing

```bash
# Navigate to specific server directory first
cd qiskit-mcp-server

# Install dependencies (including dev/test groups)
uv sync --group dev --group test

# Run all tests
./run_tests.sh
# OR
uv run pytest

# Run only unit tests (skip integration)
uv run pytest -m "not integration"

# Run only integration tests
uv run pytest -m "integration"

# Run with coverage report
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_transpiler.py -v

# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Fix linting issues automatically
uv run ruff check --fix src/ tests/
```

### Debugging

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Test Individual Functions**:
   ```python
   from qiskit_mcp_server.transpiler import transpile_circuit

   result = transpile_circuit.sync(qasm_circuit, optimization_level=2)
   print(result)
   ```

3. **Use MCP Inspector**:
   ```bash
   npx @modelcontextprotocol/inspector uv run qiskit-mcp-server
   ```

4. **Check Environment Variables**:
   ```bash
   echo $QISKIT_IBM_TOKEN
   ```

### Publishing to PyPI

Each server is published independently to PyPI. See [PUBLISHING.md](PUBLISHING.md) for details.

**Quick publishing workflow**:
```bash
# Navigate to server directory
cd qiskit-mcp-server

# Update version in pyproject.toml
# Edit version = "0.2.0"

# Build package
uv build

# Publish to PyPI (requires credentials)
uv publish

# Or publish to Test PyPI first
uv publish --repository testpypi
```

## Documentation Structure

### Repository-Level Documentation
- [README.md](README.md): Overview, quick start, architecture
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): Community guidelines
- [PUBLISHING.md](PUBLISHING.md): PyPI publishing guide
- [LICENSE](LICENSE): Apache 2.0 license
- [AGENTS.md](AGENTS.md): This file

### Server-Specific Documentation
- `qiskit-mcp-server/README.md`: Core Qiskit server docs
- `qiskit-code-assistant-mcp-server/README.md`: Code Assistant server docs
- `qiskit-ibm-runtime-mcp-server/README.md`: IBM Runtime server docs
- `qiskit-ibm-transpiler-mcp-server/README.md`: IBM Transpiler server docs

### GitHub Configuration
- `.github/CODEOWNERS`: Default reviewers (@vabarbosa @cbjuan)
- `.github/PULL_REQUEST_TEMPLATE.md`: PR template with checklist
- `.github/ISSUE_TEMPLATE/bug_report.md`: Bug report template
- `.github/ISSUE_TEMPLATE/feature_request.md`: Feature request template
- `.github/workflows/test.yml`: CI workflow (lint + test all servers)
- `.github/workflows/publish-pypi.yml`: PyPI publishing workflow

## Important Constraints

### What This Project Provides
- **MCP Servers**: Production-ready servers for quantum computing
- **Async Operations**: High-performance async-first design
- **Type Safety**: Full type checking with mypy
- **Test Coverage**: 65%+ coverage with comprehensive tests
- **Multiple Servers**: Independent, specialized servers
- **Circuit Format Support**: QASM3 and QPY for full fidelity

### What This Project Does NOT Provide
- Does NOT include AI agent implementations (only MCP servers)
- Does NOT execute quantum circuits directly (delegates to IBM Quantum or local simulators)
- Does NOT provide GUI or web interface
- Does NOT work without IBM Quantum credentials (for cloud services)
- Does NOT guarantee quantum hardware availability (depends on IBM)

### Design Principles
- **Async-first**: All MCP operations are async
- **Type-safe**: Full mypy type checking
- **Test-driven**: Comprehensive test coverage
- **Modern tooling**: uv, ruff, pytest, FastMCP
- **Modular**: Independent servers, shared patterns
- **Production-ready**: Error handling, validation, logging
- **Chainable**: QPY output format for tool chaining

## Troubleshooting

### Common Issues

1. **"401 Unauthorized" or authentication errors**:
   - Check: Is IBM Quantum token set correctly?
   - Verify: `echo $QISKIT_IBM_TOKEN`
   - Check: Token is valid on https://quantum.cloud.ibm.com/
   - Try: Set token directly in `.env` file
   - For Runtime: Check saved credentials in `~/.qiskit/qiskit-ibm.json`

2. **"Module not found" errors**:
   - Ensure: Virtual environment is activated
   - Run: `uv sync` to install dependencies
   - Check: Running from correct directory
   - Verify: Python version is 3.10+

3. **Tests failing**:
   - Install test dependencies: `uv sync --group dev --group test`
   - Check: No environment variables interfering
   - Verify: Mock services are working
   - Run: Individual test to isolate issue

4. **MCP Inspector not working**:
   - Ensure: Node.js and npm are installed
   - Check: Port 5173 is available
   - Try: `npx @modelcontextprotocol/inspector --help`
   - Verify: Server command is correct

5. **Transpilation slow with optimization level 3**:
   - Use level 2 for most use cases
   - Level 3 can be very slow for circuits with >20 qubits or >500 gates
   - Use `compare_optimization_levels` to find optimal level

### Debug Commands

```bash
# Check Python version
python --version

# Check uv installation
uv --version

# List installed packages
uv pip list

# Check environment variables
env | grep -i qiskit

# Verify package installation
uv run python -c "import qiskit_mcp_server; print('OK')"

# Test connectivity to IBM Quantum
uv run python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService()
print(f'Connected. Backends: {len(service.backends())}')
"
```

## File Structure Reference

```
qiskit-mcp-servers/
├── .claude/
│   └── skills/
│       └── qiskit-mcp-dev/
│           └── SKILL.md                 # Claude Code development skill
├── .github/
│   ├── CODEOWNERS                       # Default PR reviewers
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       ├── test.yml                     # CI: lint + test all servers
│       └── publish-pypi.yml             # CD: publish to PyPI on release
├── qiskit-mcp-server/
│   ├── src/qiskit_mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py                    # FastMCP server
│   │   ├── transpiler.py                # Core transpilation
│   │   └── circuit_serialization.py     # QASM3/QPY utilities
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_transpiler.py
│   │   └── test_circuit_serialization.py
│   ├── examples/
│   │   ├── README.md
│   │   ├── langchain_agent.ipynb
│   │   └── langchain_agent.py
│   ├── pyproject.toml
│   ├── LICENSE
│   ├── README.md
│   └── run_tests.sh
├── qiskit-code-assistant-mcp-server/
│   ├── src/qiskit_code_assistant_mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py                    # FastMCP server
│   │   ├── qca.py                       # Core async functions
│   │   ├── constants.py                 # Configuration
│   │   └── utils.py                     # Utilities
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_*.py
│   ├── examples/
│   │   ├── README.md
│   │   ├── langchain_agent.ipynb
│   │   └── langchain_agent.py
│   ├── pyproject.toml
│   ├── pytest.ini
│   ├── LICENSE
│   ├── README.md
│   └── run_tests.sh
├── qiskit-ibm-runtime-mcp-server/
│   ├── src/qiskit_ibm_runtime_mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py                    # FastMCP server
│   │   └── ibm_runtime.py               # Core async functions
│   ├── tests/
│   │   ├── conftest.py
│   │   └── test_*.py
│   ├── examples/
│   │   ├── README.md
│   │   ├── langchain_agent.ipynb
│   │   └── langchain_agent.py
│   ├── pyproject.toml
│   ├── pytest.ini
│   ├── LICENSE
│   ├── README.md
│   └── run_tests.sh
├── qiskit-ibm-transpiler-mcp-server/
│   ├── src/qiskit_ibm_transpiler_mcp_server/
│   │   ├── __init__.py
│   │   ├── server.py                    # FastMCP server
│   │   ├── qta.py                       # AI transpilation functions
│   │   └── utils.py                     # Utilities
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/                        # Unit tests
│   │   ├── integration/                 # Integration tests
│   │   ├── qasm/                        # Test QASM files
│   │   └── utils/                       # Test helpers
│   ├── examples/
│   │   ├── README.md
│   │   ├── langchain_agent.ipynb
│   │   └── langchain_agent.py
│   ├── pyproject.toml
│   ├── pytest.ini
│   ├── LICENSE
│   ├── README.md
│   └── run_tests.sh
├── src/
│   └── qiskit_mcp_servers/              # Meta-package
├── pyproject.toml                       # Workspace config & meta-package
├── mypy.ini                             # Shared mypy config
├── ruff.toml                            # Shared ruff config
├── README.md
├── AGENTS.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── PUBLISHING.md
├── LICENSE
├── uv.lock
└── .gitignore
```

## Best Practices for AI Assistants

When helping with this repository:

1. **Identify the correct server**: Ask which server the user is working with or check context
2. **Read before suggesting**: Use Read tool on relevant files before making changes
3. **Follow existing patterns**: Match code style and architecture from the specific server
4. **Don't hallucinate features**: Only reference capabilities that exist in the codebase
5. **Check documentation**: Point to correct README (main or server-specific)
6. **Test suggestions**: Verify code works with the async patterns
7. **Respect server boundaries**: Don't mix code between different servers
8. **Use proper tools**: Grep for searching, Read for files, Edit for changes
9. **Async by default**: MCP functions should be async
10. **Maintain independence**: Each server should remain independently runnable
11. **Include license headers**: All new files need the Apache 2.0 header
12. **Mock external services**: Never call real IBM Quantum APIs in tests

### Quick Reference

**Adding a tool?** → Edit `server.py`, add `@mcp.tool()` decorated function

**Adding a resource?** → Edit `server.py`, add `@mcp.resource("uri")` decorated function

**Adding tests?** → Write in `tests/test_*.py` with pytest, mock external services

**Updating docs?** → Server-specific in `<server>/README.md`, general in main `README.md`

**Publishing?** → See [PUBLISHING.md](PUBLISHING.md)

**New server?** → Copy structure from existing, add to workspace members, update all names/imports

**Architecture questions?** → Read `server.py` and main [README.md](README.md)

**Circuit format?** → Use QASM3 for human-readable, QPY for chaining tools (preserves exact parameters)
