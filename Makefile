.PHONY: install dev test lint format check clean build commit

# Install package
install:
	poetry install

# Install with all dependencies
dev:
	poetry install --all-extras

# Run tests
test:
	poetry run pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	poetry run pytest tests/ -v --cov=fastlangml --cov-report=html --cov-report=term

# Run ruff linter
lint:
	poetry run ruff check src/ tests/

# Run ruff with auto-fix
fix:
	poetry run ruff check --fix src/ tests/

# Format code with ruff
format:
	poetry run ruff format src/ tests/

# Check formatting (CI mode - no changes)
format-check:
	poetry run ruff format --check src/ tests/

# Run type checking
typecheck:
	poetry run mypy src/fastlangml --ignore-missing-imports

# Run all checks (format + lint + typecheck + test)
check: format-check lint typecheck test

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
	poetry build

# Run benchmarks
bench:
	poetry run fastlangml bench --n-samples 1000

# Publish to TestPyPI (uses ~/.pypirc)
publish-test: build
	twine upload --repository testpypi dist/*

# Publish to PyPI (uses ~/.pypirc)
publish: build
	twine upload dist/*

# Update poetry lock file
lock:
	poetry lock

# Interactive commit helper
commit:
	./scripts/commit.sh

# Show help
help:
	@echo "Available targets:"
	@echo "  install      - Install package with poetry"
	@echo "  dev          - Install with all dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run ruff linter"
	@echo "  fix          - Run ruff with auto-fix"
	@echo "  format       - Format code with ruff"
	@echo "  format-check - Check formatting (CI mode)"
	@echo "  typecheck    - Run mypy type checking"
	@echo "  check        - Run all checks (format + lint + typecheck + test)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  bench        - Run benchmarks"
	@echo "  lock         - Update poetry lock file"
	@echo "  publish-test - Build and publish to TestPyPI (uses ~/.pypirc)"
	@echo "  publish      - Build and publish to PyPI (uses ~/.pypirc)"
	@echo "  commit       - Interactive commit helper (generates message, confirms, pushes)"
