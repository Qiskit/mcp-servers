#!/bin/bash
# Run tests for qiskit-gym-mcp-server
set -e

echo "=== Installing dependencies ==="
uv sync --group dev --group test

echo ""
echo "=== Running Linting (ruff) ==="
uv run ruff check src tests
uv run ruff format --check src tests

echo ""
echo "=== Running Type Checking (mypy) ==="
uv run mypy src

echo ""
echo "=== Running Security Scan (bandit) ==="
uv run bandit -c pyproject.toml -r src

echo ""
echo "=== Running Unit Tests ==="
uv run pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

echo ""
echo "=== Running Integration Tests (optional) ==="
uv run pytest tests/ -v -m "integration" --cov=src --cov-append --cov-report=term-missing || true

echo ""
echo "=== Generating Coverage Report ==="
uv run pytest tests/ --cov=src --cov-report=html --cov-report=xml

echo ""
echo "=== All tests completed! ==="
