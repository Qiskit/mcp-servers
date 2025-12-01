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

### Key Technologies
- **Protocol**: Model Context Protocol (MCP)
- **Language**: Python 3.10+ (3.11+ recommended)
- **Framework**: FastMCP (async-first MCP framework)
- **Package Manager**: uv (modern Python package manager)
- **Testing**: pytest with async support, 65%+ coverage
- **Code Quality**: ruff (formatting + linting), mypy (type checking)
- **Build System**: hatchling with pyproject.toml

## Architecture

### Repository Structure

This is a **monorepo** containing multiple independent MCP servers:

```
qiskit-mcp-servers/
├── qiskit-code-assistant-mcp-server/    # AI code completion server
├── qiskit-ibm-runtime-mcp-server/       # IBM Quantum cloud services
├── README.md                             # Main repository 
├── PUBLISHING.md                         # PyPI publishing guide
└── LICENSE                               # Apache 2.0 license
```

Each server is:
- **Independent**: Can be installed and run separately
- **Self-contained**: Has its own dependencies and tests
- **Publishable**: Separate PyPI packages
- **Consistent**: Follows unified design principles

### Component Structure

Each MCP server follows this standard structure:

```
<server-name>/
├── src/
│   └── <package_name>/
│       ├── __init__.py          # Main entry point
│       ├── server.py            # FastMCP server definition
│       ├── <core>.py            # Core functionality (async)
│       ├── sync.py              # Synchronous wrappers for DSPy/Jupyter
│       ├── constants.py         # Configuration (optional)
│       └── utils.py             # Utilities (optional)
├── tests/
│   ├── conftest.py              # Test fixtures
│   ├── test_*.py                # Unit tests
│   └── test_integration.py      # Integration tests
├── pyproject.toml               # Project metadata & dependencies
├── README.md                    # Server-specific documentation
├── .env.example                 # Environment variable template
└── run_tests.sh                 # Test execution script
```

### Data Flow

#### 1. Qiskit Code Assistant Server
```
AI Assistant → MCP Client → qca_completion tool
                                  ↓
                            qca.py (async functions)
                                  ↓
                    IBM Qiskit Code Assistant API
                                  ↓
                        Code completion response
```

#### 2. IBM Runtime Server
```
AI Assistant → MCP Client → setup_ibm_quantum_account tool
                                  ↓
                       ibm_runtime.py (async functions)
                                  ↓
                         QiskitRuntimeService
                                  ↓
                    Backend info / Job management / Results
```

## Key Components

### 1. Qiskit Code Assistant MCP Server

**Purpose**: Intelligent quantum code completion and assistance

**Directory**: [`qiskit-code-assistant-mcp-server/`](qiskit-code-assistant-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `qca.py`: Qiskit Code Assistant API integration (async)
- `sync.py`: Synchronous wrappers for DSPy/scripts/Jupyter
- `constants.py`: API endpoints and configuration
- `utils.py`: HTTP client management and utilities

**Tools Provided**:
- `qca_completion`: Get code completion for a prompt
- `qca_rag_completion`: RAG-based completion with documentation context
- `qca_accept_completion`: Mark a completion as accepted (telemetry)

**Resources Provided**:
- `qca://status`: Service status and connection info
- `qca://models`: List available models
- `qca://model/{id}`: Specific model information

**Environment Variables**:
- `QISKIT_IBM_TOKEN`: IBM Quantum API token (required)
- `QCA_TOOL_API_BASE`: API base URL (default: https://qiskit-code-assistant.quantum.ibm.com)

### 2. Qiskit IBM Runtime MCP Server

**Purpose**: Complete access to IBM Quantum cloud services

**Directory**: [`qiskit-ibm-runtime-mcp-server/`](qiskit-ibm-runtime-mcp-server/)

**Core Files**:
- `server.py`: FastMCP server with tool/resource definitions
- `ibm_runtime.py`: Qiskit IBM Runtime integration (async)
- `sync.py`: Synchronous wrappers for DSPy/scripts/Jupyter

**Tools Provided**:
- `setup_ibm_quantum_account`: Configure IBM Quantum account
- `list_backends`: Get available quantum backends
- `least_busy_backend`: Find least busy backend
- `get_backend_properties`: Get detailed backend properties
- `list_my_jobs`: List recent jobs
- `get_job_status`: Check job status
- `cancel_job`: Cancel a running/queued job

**Resources Provided**:
- `ibm_quantum://status`: Service status and connection info

**Environment Variables**:
- `IBM_QUANTUM_TOKEN`: IBM Quantum API token (optional, can use saved credentials)

**Credential Resolution Priority**:
1. Explicit token passed to `setup_ibm_quantum_account()`
2. `IBM_QUANTUM_TOKEN` environment variable
3. Saved credentials in `~/.qiskit/qiskit-ibm.json`

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
   git clone https://github.com/Qiskit/qiskit-mcp-servers.git
   cd qiskit-mcp-servers

   # Navigate to specific server
   cd qiskit-code-assistant-mcp-server
   # OR
   cd qiskit-ibm-runtime-mcp-server

   # Install dependencies
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
   # Run the server
   uv run qiskit-code-assistant-mcp-server
   # OR
   uv run qiskit-ibm-runtime-mcp-server
   ```

5. **Interactive Testing**:
   ```bash
   # Test with MCP Inspector (requires Node.js)
   npx @modelcontextprotocol/inspector uv run qiskit-code-assistant-mcp-server
   ```

### Code Conventions

1. **Python Standards**:
   - Python 3.10+ features allowed
   - Async/await preferred for MCP operations
   - Type hints required (mypy strict mode)
   - Naming: snake_case for functions/variables, PascalCase for classes
   - Docstrings for public functions (Google style)

2. **MCP Server Patterns**:
   - All servers use FastMCP framework
   - Tools defined with `@mcp.tool()` decorator
   - Resources defined with `@mcp.resource()` decorator
   - Async functions for all MCP handlers
   - Synchronous wrappers in `sync.py` for DSPy/Jupyter compatibility

3. **Error Handling**:
   - Use appropriate HTTP status codes
   - Provide clear error messages
   - Log errors for debugging
   - Handle network failures gracefully
   - Validate inputs before API calls

4. **Testing**:
   - Write tests in `tests/` directory
   - Use pytest with async support (`pytest-asyncio`)
   - Mock external APIs (`pytest-mock`, `respx` for HTTP)
   - Target 65%+ code coverage
   - Run tests: `./run_tests.sh` or `uv run pytest`

5. **Code Quality**:
   - Format with `ruff format`
   - Lint with `ruff check`
   - Type check with `mypy src/`
   - All checks must pass before committing

### Adding New Features

1. **Adding a New Tool**:
   ```python
   # In server.py
   @mcp.tool()
   async def my_new_tool(param: str) -> dict:
       """Tool description for AI assistant."""
       # Implementation
       return {"result": "data"}
   ```

2. **Adding a New Resource**:
   ```python
   # In server.py
   @mcp.resource("protocol://path")
   async def my_resource() -> str:
       """Resource description."""
       return "resource content"
   ```

3. **Adding Synchronous Wrappers**:
   ```python
   # In sync.py
   import nest_asyncio
   import asyncio
   from .core_module import async_function

   nest_asyncio.apply()

   def function_name_sync(*args, **kwargs):
       """Synchronous wrapper for DSPy/Jupyter."""
       loop = asyncio.get_event_loop()
       return loop.run_until_complete(async_function(*args, **kwargs))
   ```

4. **Adding a New Server**:
   - Create new directory: `qiskit-<name>-mcp-server/`
   - Copy structure from existing server
   - Create `pyproject.toml` with unique package name
   - Implement server using FastMCP
   - Add comprehensive tests
   - Document in server-specific README.md
   - Update main README.md with new server info

## Common Tasks

### Building and Testing

```bash
# Navigate to specific server directory first
cd qiskit-code-assistant-mcp-server

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
uv run pytest tests/test_qca.py -v

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
   # Create test script
   from qiskit_code_assistant_mcp_server.sync import qca_get_completion_sync

   result = qca_get_completion_sync("Write a quantum circuit")
   print(result)
   ```

3. **Use MCP Inspector**:
   ```bash
   # Interactive testing environment
   npx @modelcontextprotocol/inspector uv run qiskit-code-assistant-mcp-server
   ```

4. **Check Environment Variables**:
   ```bash
   # Verify token is set
   echo $QISKIT_IBM_TOKEN
   # OR
   echo $IBM_QUANTUM_TOKEN
   ```

### Publishing to PyPI

Each server is published independently to PyPI. See [PUBLISHING.md](PUBLISHING.md) for details.

**Quick publishing workflow**:
```bash
# Navigate to server directory
cd qiskit-code-assistant-mcp-server

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
- [PUBLISHING.md](PUBLISHING.md): PyPI publishing guide
- [LICENSE](LICENSE): Apache 2.0 license
- [AGENTS.md](AGENTS.md): This file

### Server-Specific Documentation
- `qiskit-code-assistant-mcp-server/README.md`: Code Assistant server docs
- `qiskit-ibm-runtime-mcp-server/README.md`: IBM Runtime server docs

## Important Constraints

### What This Project Provides
- **MCP Servers**: Production-ready servers for quantum computing
- **Async Operations**: High-performance async-first design
- **Sync Wrappers**: DSPy/Jupyter compatibility
- **Type Safety**: Full type checking with mypy
- **Test Coverage**: 65%+ coverage with comprehensive tests
- **Multiple Servers**: Independent, specialized servers

### What This Project Does NOT Provide
- Does NOT include AI agent implementations (only MCP servers)
- Does NOT execute quantum circuits directly (delegates to IBM Quantum)
- Does NOT provide GUI or web interface
- Does NOT work without IBM Quantum credentials
- Does NOT guarantee quantum hardware availability (depends on IBM)

### Design Principles
- **Async-first**: All MCP operations are async
- **Type-safe**: Full mypy type checking
- **Test-driven**: Comprehensive test coverage
- **Modern tooling**: uv, ruff, pytest, FastMCP
- **Modular**: Independent servers, shared patterns
- **Production-ready**: Error handling, validation, logging

## Troubleshooting

### Common Issues

1. **"401 Unauthorized" or authentication errors**:
   - Check: Is IBM Quantum token set correctly?
   - Verify: `echo $QISKIT_IBM_TOKEN` or `echo $IBM_QUANTUM_TOKEN`
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

4. **Import errors in sync wrappers**:
   - Symptom: "cannot import name 'X_sync'"
   - Check: Running latest version
   - Verify: `sync.py` is present in package
   - Try: Reinstall with `uv sync`

5. **MCP Inspector not working**:
   - Ensure: Node.js and npm are installed
   - Check: Port 5173 is available
   - Try: `npx @modelcontextprotocol/inspector --help`
   - Verify: Server command is correct

### Debug Commands

```bash
# Check Python version
python --version

# Check uv installation
uv --version

# List installed packages
uv pip list

# Check environment variables
env | grep -i quantum

# Verify package installation
uv run python -c "import qiskit_code_assistant_mcp_server; print('OK')"

# Test token authentication
uv run python -c "
from qiskit_code_assistant_mcp_server.sync import qca_get_service_status_sync
print(qca_get_service_status_sync())
"
```

## File Structure Reference

```
qiskit-mcp-servers/
├── .github/
│   └── workflows/              # CI/CD (if present)
├── qiskit-code-assistant-mcp-server/
│   ├── src/
│   │   └── qiskit_code_assistant_mcp_server/
│   │       ├── __init__.py     # Entry point
│   │       ├── server.py       # FastMCP server
│   │       ├── qca.py          # Core async functions
│   │       ├── sync.py         # Sync wrappers
│   │       ├── constants.py    # Configuration
│   │       └── utils.py        # Utilities
│   ├── tests/
│   │   ├── conftest.py         # Test fixtures
│   │   ├── test_qca.py         # Unit tests
│   │   ├── test_utils.py       # Utility tests
│   │   ├── test_constants.py   # Config tests
│   │   └── test_integration.py # Integration tests
│   ├── pyproject.toml          # Project metadata
│   ├── README.md               # Server documentation
│   ├── .env.example            # Env template
│   └── run_tests.sh            # Test runner
├── qiskit-ibm-runtime-mcp-server/
│   ├── src/
│   │   └── qiskit_ibm_runtime_mcp_server/
│   │       ├── __init__.py     # Entry point
│   │       ├── server.py       # FastMCP server
│   │       ├── ibm_runtime.py  # Core async functions
│   │       └── sync.py         # Sync wrappers
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_server.py
│   │   └── test_integration.py
│   ├── pyproject.toml
│   ├── README.md
│   ├── .env.example
│   └── run_tests.sh
├── README.md                   # Main repository docs
├── PUBLISHING.md               # Publishing guide
├── AGENTS.md                   # This file
├── LICENSE                     # Apache 2.0
└── .gitignore
```

## Best Practices for AI Assistants

When helping with this repository:

1. **Identify the correct server**: Ask which server the user is working with or check context
2. **Read before suggesting**: Use Read tool on relevant files before making changes
3. **Follow existing patterns**: Match code style and architecture from the specific server
4. **Don't hallucinate features**: Only reference capabilities that exist in the codebase
5. **Check documentation**: Point to correct README (main or server-specific)
6. **Test suggestions**: Verify code works with the async/sync patterns
7. **Respect server boundaries**: Don't mix code between different servers
8. **Use proper tools**: Grep for searching, Read for files, Edit for changes
9. **Async by default**: MCP functions should be async, sync wrappers go in `sync.py`
10. **Maintain independence**: Each server should remain independently runnable

### Quick Reference

**Adding a tool?** → Edit `server.py`, add `@mcp.tool()` decorated function

**Adding tests?** → Write in `tests/test_*.py` with pytest

**Need sync wrapper?** → Add to `sync.py` using `nest_asyncio` pattern

**Updating docs?** → Server-specific in `<server>/README.md`, general in main `README.md`

**Want examples?** → Check `example*.ipynb` notebooks

**Publishing?** → See [PUBLISHING.md](PUBLISHING.md)

**New server?** → Copy structure from existing, update all names/imports

**Architecture questions?** → Read `server.py` and main [README.md](README.md)
