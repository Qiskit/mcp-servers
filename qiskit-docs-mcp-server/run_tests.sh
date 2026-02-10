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

# Qiskit Doc MCP Server - Test Runner Script

set -e

echo "🧪 Running Qiskit Doc MCP Server Tests"
echo "======================================"

# Install test dependencies
echo "📦 Installing test dependencies..."
uv sync --group dev --group test

# Run linting
echo ""
echo "🔍 Running code linting..."
uv run ruff check src tests
uv run ruff format --check src tests

# # Run type checking
# echo ""
# echo "🔬 Running type checking..."
# uv run mypy src

# Run unit tests
echo ""
echo "🧪 Running unit tests..."
uv run pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

# Run integration tests
echo ""
echo "🔗 Running integration tests..."
uv run pytest tests/ -v -m "integration" --cov=src --cov-append --cov-report=term-missing || true

# Generate coverage report
echo ""
echo "📊 Generating coverage report..."
uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml

echo ""
echo "✅ All tests completed successfully!"
echo "📋 Coverage report generated in htmlcov/index.html"

