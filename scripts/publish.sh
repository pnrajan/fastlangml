#!/bin/bash
# Publish FastLangML to PyPI
#
# Usage:
#   ./scripts/publish.sh [--test] [--skip-tests]
#
# Options:
#   --test        Upload to TestPyPI instead of PyPI
#   --skip-tests  Skip running tests (use with caution)
#
# Authentication (in order of priority):
#   1. ~/.pypirc file (recommended)
#   2. PYPI_TOKEN / TEST_PYPI_TOKEN environment variables
#
# ~/.pypirc format:
#   [pypi]
#   username = __token__
#   password = pypi-xxx
#
#   [testpypi]
#   username = __token__
#   password = pypi-xxx
#
# Examples:
#   ./scripts/publish.sh              # Uses ~/.pypirc
#   ./scripts/publish.sh --test       # Upload to TestPyPI
#   PYPI_TOKEN=pypi-xxx ./scripts/publish.sh  # Use env var

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
USE_TEST_PYPI=false
SKIP_TESTS=false

for arg in "$@"; do
    case $arg in
        --test) USE_TEST_PYPI=true ;;
        --skip-tests) SKIP_TESTS=true ;;
    esac
done

if [[ "$USE_TEST_PYPI" == true ]]; then
    echo -e "${YELLOW}=== Publishing to TestPyPI ===${NC}"
else
    echo -e "${YELLOW}=== Publishing to PyPI ===${NC}"
fi

# Step 1: Clean
echo -e "\n${GREEN}[1/6] Cleaning...${NC}"
rm -rf dist/ build/ *.egg-info/

# Step 2: Lint
echo -e "\n${GREEN}[2/6] Running ruff...${NC}"
python -m ruff check src/ tests/
echo "Linting passed!"

# Step 3: Tests
if [[ "$SKIP_TESTS" == true ]]; then
    echo -e "\n${YELLOW}[3/6] Skipping tests${NC}"
else
    echo -e "\n${GREEN}[3/6] Running tests...${NC}"
    python -m pytest tests/ -v --tb=short
    echo "Tests passed!"
fi

# Step 4: Build
echo -e "\n${GREEN}[4/6] Building...${NC}"
python -m build

# Step 5: Check
echo -e "\n${GREEN}[5/6] Checking package...${NC}"
python -m twine check dist/*

# Step 6: Upload
echo -e "\n${GREEN}[6/6] Uploading...${NC}"

if [[ "$USE_TEST_PYPI" == true ]]; then
    REPO="testpypi"
    TOKEN="${TEST_PYPI_TOKEN:-}"
    INSTALL_CMD="pip install --index-url https://test.pypi.org/simple/ fastlangml"
else
    REPO="pypi"
    TOKEN="${PYPI_TOKEN:-}"
    INSTALL_CMD="pip install fastlangml"
fi

# Use ~/.pypirc for authentication
if [[ -f ~/.pypirc ]]; then
    echo "Using ~/.pypirc for authentication..."
    python -m twine upload --config-file ~/.pypirc --repository "$REPO" dist/*
else
    echo -e "${BLUE}~/.pypirc not found.${NC}"
    echo "Create ~/.pypirc with:"
    echo ""
    echo "[pypi]"
    echo "username = __token__"
    echo "password = pypi-YOUR_TOKEN"
    echo ""
    exit 1
fi

echo -e "\n${GREEN}=== Published! ===${NC}"
echo -e "Install: ${BLUE}${INSTALL_CMD}${NC}"
