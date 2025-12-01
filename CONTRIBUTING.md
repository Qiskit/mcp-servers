# Contributing to Qiskit MCP Servers

Thank you for your interest in contributing to Qiskit MCP Servers! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by the [Qiskit Code of Conduct](CODE_OF_CONDUCT.md). Please treat all community members with respect and create a welcoming environment for everyone.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (3.11+ recommended)
- **[uv](https://astral.sh/uv)** - Modern Python package manager
- **Git** - Version control
- **IBM Quantum account** - Get your API token from [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com/)

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/mcp-servers.git
   cd mcp-servers
   ```

2. **Navigate to the server you want to work on**:
   ```bash
   cd qiskit-code-assistant-mcp-server
   # OR
   cd qiskit-ibm-runtime-mcp-server
   ```

3. **Install dependencies**:
   ```bash
   uv sync --group dev --group test
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your IBM Quantum API token
   ```

5. **Run the server locally**:
   ```bash
   uv run qiskit-code-assistant-mcp-server
   # OR
   uv run qiskit-ibm-runtime-mcp-server
   ```

6. **Test interactively with MCP Inspector** (requires Node.js):
   ```bash
   npx @modelcontextprotocol/inspector uv run qiskit-code-assistant-mcp-server
   ```

## Contributing Workflow

### 1. Find or Create an Issue

- Browse [existing issues](https://github.com/Qiskit/mcp-servers/issues) to find something to work on
- If you have a new idea, create an issue first to discuss it
- Assign yourself to the issue you're working on for visibility

### 2. Create a Feature Branch

```bash
# Ensure your main branch is up to date
git checkout main
git pull origin main

# Create a new branch with a descriptive name
git checkout -b feature/your-feature-name
# OR
git checkout -b fix/bug-description
```

### 3. Make Your Changes

- Follow the [code conventions](#code-conventions) below
- Write tests for new functionality
- Update documentation as needed

### 4. Run Tests and Quality Checks

Before submitting, ensure all checks pass:

```bash
# Run tests
./run_tests.sh
# OR
uv run pytest

# Format code
uv run ruff format src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: description of what was added"
# OR
git commit -m "fix: description of bug that was fixed"
```

### 6. Submit a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request against the `main` branch

3. Fill out the PR template with:
   - A summary of changes
   - Link to related issue(s)
   - Description of how it was tested

4. Wait for review - at least one approval is required before merging

## Code Conventions

### Python Standards

- **Python version**: 3.10+ features allowed
- **Async/await**: Primary implementation for all MCP operations, with sync wrappers in `sync.py`
- **Type hints**: Required (mypy strict mode)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings**: Google style for public functions

### MCP Server Patterns

- All servers use the FastMCP framework
- Tools are defined with `@mcp.tool()` decorator
- Resources are defined with `@mcp.resource()` decorator
- Async functions for all MCP handlers
- Synchronous wrappers go in `sync.py` for DSPy/Jupyter compatibility

### Error Handling

- Use appropriate HTTP status codes
- Provide clear error messages
- Log errors for debugging
- Handle network failures gracefully
- Validate inputs before API calls

### Testing

- Write tests in `tests/` directory
- Use pytest with async support (`pytest-asyncio`)
- Mock external APIs (`pytest-mock`, `respx` for HTTP)
- Target 65%+ code coverage

## Project Structure

This is a monorepo with multiple independent MCP servers:

```
qiskit-mcp-servers/
├── qiskit-code-assistant-mcp-server/    # AI code completion server
├── qiskit-ibm-runtime-mcp-server/       # IBM Quantum cloud services
├── README.md                            # Main documentation
├── CONTRIBUTING.md                      # This file
├── AGENTS.md                            # AI assistant guidance
└── LICENSE                              # Apache 2.0
```

Each server follows this structure:

```
<server-name>/
├── src/<package_name>/
│   ├── __init__.py          # Main entry point
│   ├── server.py            # FastMCP server definition
│   ├── <core>.py            # Core async functionality
│   ├── sync.py              # Synchronous wrappers
│   └── utils.py             # Utilities (optional)
├── tests/
│   ├── conftest.py          # Test fixtures
│   └── test_*.py            # Unit/integration tests
├── pyproject.toml           # Project metadata
├── README.md                # Server documentation
└── run_tests.sh             # Test runner
```

## Adding New Features

### Adding a New Tool

```python
# In server.py
@mcp.tool()
async def my_new_tool(param: str) -> dict:
    """Tool description for AI assistant."""
    # Implementation
    return {"result": "data"}
```

### Adding a New Resource

```python
# In server.py
@mcp.resource("protocol://path")
async def my_resource() -> str:
    """Resource description."""
    return "resource content"
```

### Adding Synchronous Wrappers

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

## Getting Help

- **Questions?** Open an [issue](https://github.com/Qiskit/mcp-servers/issues)
- **Found a bug?** [Report it](https://github.com/Qiskit/mcp-servers/issues/new?template=bug_report.md)
- **Have an idea?** [Request a feature](https://github.com/Qiskit/mcp-servers/issues/new?template=feature_request.md)

## AI Development Assistants

This repository includes [AGENTS.md](AGENTS.md) which provides comprehensive guidance for AI coding assistants (like IBM Bob, Claude Code, GitHub Copilot, and others). If you're using an AI assistant to help with contributions, it will have context about the codebase structure, conventions, and best practices.

## License

By contributing to Qiskit MCP Servers, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
