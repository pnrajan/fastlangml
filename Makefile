.PHONY: install dev test lint format check clean build

# Install package
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev,all]"

# Run tests
test:
	python -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	python -m pytest tests/ -v --cov=fastlangml --cov-report=html --cov-report=term

# Run ruff linter
lint:
	ruff check src/ tests/

# Run ruff with auto-fix
fix:
	ruff check --fix src/ tests/

# Format code with ruff
format:
	ruff format src/ tests/

# Run type checking
typecheck:
	mypy src/fastlangml --ignore-missing-imports

# Run all checks (lint + typecheck + test)
check: lint typecheck test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Run benchmarks
bench:
	fastlangml bench --n-samples 1000

# Publish to TestPyPI
publish-test:
	./scripts/publish.sh --test

# Publish to PyPI
publish:
	./scripts/publish.sh

# Show help
help:
	@echo "Available targets:"
	@echo "  install    - Install package"
	@echo "  dev        - Install with dev dependencies"
	@echo "  test       - Run tests"
	@echo "  test-cov   - Run tests with coverage"
	@echo "  lint       - Run ruff linter"
	@echo "  fix        - Run ruff with auto-fix"
	@echo "  format     - Format code with ruff"
	@echo "  typecheck  - Run mypy type checking"
	@echo "  check      - Run all checks (lint + typecheck + test)"
	@echo "  clean      - Clean build artifacts"
	@echo "  build      - Build package"
	@echo "  bench      - Run benchmarks"
	@echo "  publish-test - Publish to TestPyPI"
	@echo "  publish    - Publish to PyPI"
