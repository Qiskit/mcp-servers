#!/bin/bash

# Qiskit IBM Transpiler MCP Server - Test Runner Script

set -e

echo "🧪 Running Qiskit IBM Transpiler MCP Server Tests"
echo "=============================================="

# Install test dependencies
echo "📦 Installing test dependencies..."
uv sync --extra dev --extra test

# Run linting
echo ""
echo "🔍 Running code linting..."
uv run ruff check src tests --fix
uv run ruff format src tests

echo ""
echo "🛡️ Running Bandit security checks..."
uv run bandit -r src -f json -o bandit_report.json

# Run type checking
echo ""
echo "🔬 Running type checking..."
uv run mypy src

# Run unit tests
echo ""
echo "🧪 Running unit tests..."
uv run pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

# Run integration tests
echo ""
echo "🔗 Running integration tests..."
uv run pytest tests/ -v -m "integration" --cov=src --cov-append --cov-report=term-missing

# Generate coverage report
echo ""
echo "📊 Generating coverage report..."
uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml

echo ""
echo "✅ All tests completed successfully!"
echo "📋 Coverage report generated in htmlcov/index.html"

# Assisted by watsonx Code Assistant