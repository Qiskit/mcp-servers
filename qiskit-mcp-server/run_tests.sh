#!/bin/bash
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

set -e

echo "=== Qiskit Transpiler MCP Server Test Suite ==="
echo ""

# Install dependencies
echo "Installing dependencies..."
uv sync --group dev --group test

# Linting
echo ""
echo "=== Running Linting ==="
uv run ruff check src tests
uv run ruff format --check src tests

# Type checking
echo ""
echo "=== Running Type Checking ==="
uv run mypy src

# Unit tests
echo ""
echo "=== Running Unit Tests ==="
uv run pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

# Integration tests (if any)
echo ""
echo "=== Running Integration Tests ==="
uv run pytest tests/ -v -m "integration" --cov=src --cov-append --cov-report=term-missing || true

# Generate coverage report
echo ""
echo "=== Generating Coverage Report ==="
uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml

echo ""
echo "=== All Tests Completed ==="
