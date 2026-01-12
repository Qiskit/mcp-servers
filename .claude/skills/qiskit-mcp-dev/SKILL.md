---
name: qiskit-mcp-dev
description: Guide development of Qiskit MCP servers. Use when creating tools, resources, tests, or new servers. Helps with async patterns, pytest fixtures, GitHub CI/CD setup, and code quality. Trigger on "add tool", "add resource", "write test", "new server", "FastMCP", "MCP development", or "GitHub workflow".
allowed-tools: Read, Grep, Glob, Edit, Write, Bash
---

# Qiskit MCP Server Development Guide

## Instructions

When helping with MCP server development in this repository:

1. **Read existing code first** - Check `server.py` and core modules before adding new functionality
2. **Follow established patterns** - Match the style of existing tools and tests
3. **Include license headers** - All new files need the Apache 2.0 header
4. **Keep tools async** - MCP tools use `async def`, delegate to core module functions
5. **Mock external services** - Never call real IBM Quantum APIs in tests
6. **Update GitHub CI/CD** - New servers require workflow updates (see below)

## License Header (Required for all new files)

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

## Adding a Tool

Tools go in `server.py` and delegate to async functions in the core module:

```python
# Example: from qiskit_<name>_mcp_server.<core> import my_function
from qiskit_ibm_runtime_mcp_server.ibm_runtime import my_function

@mcp.tool()
async def my_tool_name(param: str, optional_param: int = 10) -> dict[str, Any]:
    """Clear description of what this tool does.

    Args:
        param: Description of the parameter
        optional_param: Description with default behavior

    Returns:
        Description of the return structure
    """
    return await my_function(param, optional_param)
```

Key patterns:
- Tool function name ends with `_tool` suffix
- Docstring describes the tool for AI assistants
- Return type is `dict[str, Any]`
- Delegate to core module, don't implement logic here

## Error Handling Pattern

All tools return consistent response format:

```python
# Success
return {"status": "success", "data": result, ...}

# Error
return {"status": "error", "message": "Description of what went wrong"}
```

## Adding a Resource

```python
@mcp.resource("protocol://path", mime_type="application/json")
async def get_resource_name() -> dict[str, Any]:
    """Description of the resource."""
    return await get_data_from_core_module()
```

## Writing Tests

Tests use pytest with mocked services. See `tests/conftest.py` for fixtures.

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_my_tool(mock_service):
    """Test description."""
    with patch(
        "qiskit_<name>_mcp_server.<core>.ExternalService",
        return_value=mock_service,
    ):
        result = await my_function()
        assert result["status"] == "success"
```

Common fixture patterns in `conftest.py` (names vary by server):
- `mock_*_service` - Mocked external service (e.g., `mock_runtime_service`, `mock_http_client`)
- `mock_env_vars` - Sets environment variables like `QISKIT_IBM_TOKEN`
- `reset_*` - Resets global state between tests (often `autouse=True`)

## Code Quality

Run from the server directory:

```bash
uv run ruff format src/ tests/    # Format
uv run ruff check src/ tests/     # Lint
uv run mypy src/                   # Type check
uv run pytest                      # Test
```

## Adding a New Server

When creating a new server `qiskit-<name>-mcp-server`:

1. **Create server directory** with standard structure (see below)
2. **Add to workspace** in root `pyproject.toml`:
   ```toml
   [tool.uv.workspace]
   members = [..., "qiskit-<name>-mcp-server"]
   ```
3. **Create examples/** with LangChain agent demos:
   - `README.md` - Setup instructions and usage
   - `langchain_agent.ipynb` - Interactive Jupyter tutorial
   - `langchain_agent.py` - CLI agent supporting multiple LLM providers
4. **Update GitHub CI** - `.github/workflows/test.yml`:
   - Add to `lint` job (install, ruff, mypy, bandit steps)
   - Add new `test-<name>` job
5. **Update GitHub CD** - `.github/workflows/publish-pypi.yml`:
   - Add to `workflow_dispatch` options
   - Add `publish-<name>` job
   - Add to `publish-meta-package` needs array
6. **Update documentation** - README.md, AGENTS.md

## Server Structure

```
qiskit-<name>-mcp-server/
├── src/qiskit_<name>_mcp_server/
│   ├── __init__.py      # Package init, version
│   ├── server.py        # FastMCP server, @mcp.tool() and @mcp.resource()
│   └── <core>.py        # Async business logic
├── tests/
│   ├── conftest.py      # Fixtures for mocking
│   └── test_*.py        # Test files
├── examples/
│   ├── README.md                # Example documentation
│   ├── langchain_agent.ipynb    # Jupyter notebook tutorial
│   └── langchain_agent.py       # CLI agent example
├── pyproject.toml
├── pytest.ini           # (optional) pytest configuration
├── .env.example         # (optional) Environment variable template
├── LICENSE              # Apache 2.0 license (copy from root)
├── README.md
└── run_tests.sh
```

## Quick Reference

| Task | File | Pattern |
|------|------|---------|
| Add tool | `server.py` | `@mcp.tool()` decorator |
| Add resource | `server.py` | `@mcp.resource("uri")` decorator |
| Core logic | `<core>.py` | Async functions, return dicts |
| Test fixtures | `tests/conftest.py` | `@pytest.fixture` |
| Unit tests | `tests/test_*.py` | `@pytest.mark.asyncio` |
| New server CI | `.github/workflows/test.yml` | Add lint steps + test job |
| New server CD | `.github/workflows/publish-pypi.yml` | Add publish job |

For comprehensive documentation, read [AGENTS.md](AGENTS.md) in the repository root.
