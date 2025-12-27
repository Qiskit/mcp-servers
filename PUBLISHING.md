# Publishing to PyPI

This guide covers how to publish the Qiskit MCP servers to PyPI, both manually and via automated workflows.

## Packages

This repository contains multiple PyPI packages:

1. **qiskit-mcp-server** - MCP server for Qiskit quantum computing capabilities with circuit serialization utilities
2. **qiskit-code-assistant-mcp-server** - MCP server for Qiskit Code Assistant
3. **qiskit-ibm-runtime-mcp-server** - MCP server for IBM Quantum Runtime
4. **qiskit-ibm-transpiler-mcp-server** - MCP server for transpilation using the AI-powered transpiler passes.
5. **qiskit-mcp-servers** - Meta-package that installs all MCP servers

### Meta-Package

The `qiskit-mcp-servers` meta-package provides a convenient way to install all servers at once:

```bash
# Install all MCP servers
pip install qiskit-mcp-servers

# Or install individual servers via extras
pip install qiskit-mcp-servers[qiskit]           # Only Qiskit
pip install qiskit-mcp-servers[code-assistant]   # Only Code Assistant
pip install qiskit-mcp-servers[runtime]          # Only Runtime
pip install qiskit-mcp-servers[transpiler]       # Only Transpiler
```

## Automated Publishing (Recommended)

### Prerequisites

To create releases using the automated approach, you'll need:

- **Git**: For creating and pushing tags
- **GitHub CLI (`gh`)**: For creating releases from the command line
  - Install from https://cli.github.com/ or via package manager:
    ```bash
    # macOS
    brew install gh

    # Linux
    sudo apt install gh  # Debian/Ubuntu
    sudo dnf install gh  # Fedora

    # Windows
    winget install GitHub.cli
    ```
  - Authenticate with: `gh auth login`

Alternatively, you can create releases manually through the GitHub web interface instead of using the `gh` CLI.

### Setup: Configure Trusted Publishing

**One-time setup** - Configure trusted publishing on PyPI (no API tokens needed):

1. Go to PyPI and create the project (if it doesn't exist):
   - For `qiskit-mcp-server`: https://pypi.org/manage/project/qiskit-mcp-server/settings/publishing/
   - For `qiskit-code-assistant-mcp-server`: https://pypi.org/manage/project/qiskit-code-assistant-mcp-server/settings/publishing/
   - For `qiskit-ibm-runtime-mcp-server`: https://pypi.org/manage/project/qiskit-ibm-runtime-mcp-server/settings/publishing/
   - For `qiskit-ibm-transpiler-mcp-server`: https://pypi.org/manage/project/qiskit-ibm-transpiler-mcp-server/settings/publishing/
   - For `qiskit-mcp-servers`: https://pypi.org/manage/project/qiskit-mcp-servers/settings/publishing/

2. Add a "trusted publisher" with these settings:
   - **PyPI Project Name**: `qiskit-mcp-server` (or `qiskit-code-assistant-mcp-server`, `qiskit-ibm-runtime-mcp-server`, `qiskit-ibm-transpiler-mcp-server`, or `qiskit-mcp-servers`)
   - **Owner**: `AI4quantum`
   - **Repository**: `mcp-servers`
   - **Workflow name**: `publish-pypi.yml`
   - **Environment name**: (leave blank)

### Publishing via GitHub Releases

The workflow automatically publishes when you create a GitHub release. The tag name determines which package is published.

#### Tag Naming Convention

| Tag Pattern | Package Published |
|-------------|-------------------|
| `qiskit-v*` | qiskit-mcp-server |
| `code-assistant-v*` | qiskit-code-assistant-mcp-server |
| `runtime-v*` | qiskit-ibm-runtime-mcp-server |
| `transpiler-v*` | qiskit-ibm-transpiler-mcp-server |
| `meta-v*` | qiskit-mcp-servers (meta-package) |

#### Complete Release Workflow

Follow these steps to release a package:

##### Step 1: Update Version

Edit the version in the appropriate `pyproject.toml`:
- **Qiskit**: `qiskit-mcp-server/pyproject.toml`
- **Code Assistant**: `qiskit-code-assistant-mcp-server/pyproject.toml`
- **Runtime**: `qiskit-ibm-runtime-mcp-server/pyproject.toml`
- **Transpiler**: `qiskit-ibm-transpiler-mcp-server/pyproject.toml`
- **Meta-package**: `pyproject.toml` (root)

##### Step 2: Commit and Push Changes

```bash
# Stage and commit the version change
git add -A
git commit -m "Bump qiskit-code-assistant-mcp-server to v0.1.1"

# Push to main branch
git push origin main
```

##### Step 3: Create and Push Tag

```bash
# Create an annotated tag
git tag -a code-assistant-v0.1.1 -m "Release qiskit-code-assistant-mcp-server v0.1.1"

# Push the tag to GitHub
git push origin code-assistant-v0.1.1
```

##### Step 4: Create GitHub Release

```bash
# Create the release (this triggers the publish workflow)
gh release create code-assistant-v0.1.1 \
  --title "qiskit-code-assistant-mcp-server v0.1.1" \
  --generate-notes
```

Or use `--notes "Your release notes here"` instead of `--generate-notes` for custom notes.

#### Quick Reference Examples

**Qiskit Server:**
```bash
# After updating version in qiskit-mcp-server/pyproject.toml
git add -A && git commit -m "Bump qiskit-mcp-server to v0.1.1" && git push origin main
git tag -a qiskit-v0.1.1 -m "Release v0.1.1" && git push origin qiskit-v0.1.1
gh release create qiskit-v0.1.1 --title "qiskit-mcp-server v0.1.1" --generate-notes
```

**Code Assistant Server:**
```bash
# After updating version in qiskit-code-assistant-mcp-server/pyproject.toml
git add -A && git commit -m "Bump code-assistant to v0.1.1" && git push origin main
git tag -a code-assistant-v0.1.1 -m "Release v0.1.1" && git push origin code-assistant-v0.1.1
gh release create code-assistant-v0.1.1 --title "qiskit-code-assistant-mcp-server v0.1.1" --generate-notes
```

**Runtime Server:**
```bash
# After updating version in qiskit-ibm-runtime-mcp-server/pyproject.toml
git add -A && git commit -m "Bump runtime to v0.1.1" && git push origin main
git tag -a runtime-v0.1.1 -m "Release v0.1.1" && git push origin runtime-v0.1.1
gh release create runtime-v0.1.1 --title "qiskit-ibm-runtime-mcp-server v0.1.1" --generate-notes
```

**Transpiler Server:**
```bash
# After updating version in qiskit-ibm-transpiler-mcp-server/pyproject.toml
git add -A && git commit -m "Bump transpiler to v0.1.0" && git push origin main
git tag -a transpiler-v0.1.0 -m "Release v0.1.0" && git push origin transpiler-v0.1.0
gh release create transpiler-v0.1.0 --title "qiskit-ibm-transpiler-mcp-server v0.1.0" --generate-notes
```

**Meta-Package:**
```bash
# After updating version in pyproject.toml (root)
git add -A && git commit -m "Bump meta-package to v0.1.1" && git push origin main
git tag -a meta-v0.1.1 -m "Release v0.1.1" && git push origin meta-v0.1.1
gh release create meta-v0.1.1 --title "qiskit-mcp-servers v0.1.1" --generate-notes
```

### Manual Workflow Trigger

You can also trigger publishing manually via GitHub Actions using the CLI:

```bash
# Publish all packages (individual servers + meta-package)
gh workflow run "Publish to PyPI" -f package=all

# Publish only qiskit
gh workflow run "Publish to PyPI" -f package=qiskit

# Publish only code-assistant
gh workflow run "Publish to PyPI" -f package=code-assistant

# Publish only runtime
gh workflow run "Publish to PyPI" -f package=runtime

# Publish only transpiler
gh workflow run "Publish to PyPI" -f package=transpiler

# Publish only meta-package
gh workflow run "Publish to PyPI" -f package=meta-package
```

Alternatively, you can trigger via the GitHub web interface:

1. Go to **Actions** → **Publish to PyPI**
2. Click **Run workflow**
3. Select which package to publish: `all`, `meta-package`, `qiskit`, `code-assistant`, `runtime`, or `transpiler`

## Manual Publishing

### Prerequisites

**Python version**: Python 3.10 or higher is required (as specified in each mcp-server `pyproject.toml`).

Install build tools:
```bash
pip install build twine
```

Or use `uv` (recommended):
```bash
pip install uv
```

### Step-by-Step Manual Publishing

#### 1. Update Version

Edit the version in `pyproject.toml`:
- **Qiskit**: `qiskit-mcp-server/pyproject.toml`
- **Code Assistant**: `qiskit-code-assistant-mcp-server/pyproject.toml`
- **Runtime**: `qiskit-ibm-runtime-mcp-server/pyproject.toml`
- **Transpiler**: `qiskit-ibm-transpiler-mcp-server/pyproject.toml`
- **Meta-package**: `pyproject.toml` (root)

#### 2. Build the Package

**For Qiskit:**
```bash
cd qiskit-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

**For Code Assistant:**
```bash
cd qiskit-code-assistant-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

**For Runtime:**
```bash
cd qiskit-ibm-runtime-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

**For Transpiler:**
```bash
cd qiskit-ibm-transpiler-mcp-server

# Build with uv (recommended)
uv build

# Or with build
python -m build
```

**For Meta-Package:**
```bash
# From repository root
uv build

# Or with build
python -m build
```

This creates `.whl` and `.tar.gz` files in the `dist/` directory.

#### 3. Verify the Build

Check the contents:
```bash
# List files in the wheel
unzip -l dist/*.whl

# Check package metadata
twine check dist/*
```

#### 4. Upload to PyPI

**Test on TestPyPI first (recommended):**
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ qiskit-mcp-server
# or
pip install --index-url https://test.pypi.org/simple/ qiskit-code-assistant-mcp-server
# or
pip install --index-url https://test.pypi.org/simple/ qiskit-ibm-runtime-mcp-server
# or
pip install --index-url https://test.pypi.org/simple/ qiskit-ibm-transpiler-mcp-server
```

**Upload to production PyPI:**
```bash
# With twine
twine upload dist/*

# Or with uv
uv publish
```

You'll be prompted for your PyPI username and password (or API token).

#### 5. Verify Installation

```bash
# For Qiskit
pip install qiskit-mcp-server

# For Code Assistant
pip install qiskit-code-assistant-mcp-server

# For Runtime
pip install qiskit-ibm-runtime-mcp-server

# For Transpiler
pip install qiskit-ibm-transpiler-mcp-server

# For Meta-Package (installs all servers)
pip install qiskit-mcp-servers
```

## Version Management

### Versioning Strategy

Both packages use **semantic versioning**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes to the API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Current Versions

The current version for each package is defined in their respective `pyproject.toml` files:

- **qiskit-mcp-server**: See [qiskit-mcp-server/pyproject.toml](qiskit-mcp-server/pyproject.toml) (search for `version =`)
- **qiskit-code-assistant-mcp-server**: See [qiskit-code-assistant-mcp-server/pyproject.toml](qiskit-code-assistant-mcp-server/pyproject.toml) (search for `version =`)
- **qiskit-ibm-runtime-mcp-server**: See [qiskit-ibm-runtime-mcp-server/pyproject.toml](qiskit-ibm-runtime-mcp-server/pyproject.toml) (search for `version =`)
- **qiskit-ibm-transpiler-mcp-server**: See [qiskit-ibm-transpiler-mcp-server/pyproject.toml](qiskit-ibm-transpiler-mcp-server/pyproject.toml) (search for `version =`)
- **qiskit-mcp-servers**: See [pyproject.toml](pyproject.toml) (search for `version =`)

## Pre-Publication Checklist

Before publishing, ensure:

- [ ] Version number updated in `pyproject.toml`
- [ ] All tests pass: `./run_tests.sh`
- [ ] Code is formatted: `uv run ruff format`
- [ ] Linting passes: `uv run ruff check`
- [ ] Type checking passes: `uv run mypy src`
- [ ] README is up to date
- [ ] CHANGELOG updated (if you have one)
- [ ] Git commit and tag created

## Troubleshooting

### "Package already exists" error

You cannot overwrite a version on PyPI. You must:
1. Increment the version number in `pyproject.toml`
2. Rebuild and upload

### Authentication issues

For manual uploads, create a PyPI API token:
1. Go to https://pypi.org/manage/account/token/
2. Create a token with upload permissions
3. Use `__token__` as username and the token as password

Or configure in `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE
```

### Build artifacts in wrong location

Make sure you're running build commands from the package directory:
```bash
cd qiskit-mcp-server  # or qiskit-code-assistant-mcp-server, qiskit-ibm-runtime-mcp-server, etc.
uv build
```

## Resources

- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Trusted Publishers (PyPI)](https://docs.pypi.org/trusted-publishers/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions - PyPI Publish](https://github.com/marketplace/actions/pypi-publish)
