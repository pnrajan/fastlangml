# Contributing to FastLangML

Thank you for your interest in contributing to FastLangML! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Release Process](#release-process)

---

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

**Our Standards:**
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Accept constructive criticism gracefully

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Finding Issues to Work On

1. **Good First Issues**: Look for issues labeled `good-first-issue` - these are ideal for newcomers
2. **Help Wanted**: Issues labeled `help-wanted` need community contribution
3. **Bug Fixes**: Issues labeled `bug` are always appreciated
4. **Feature Requests**: Check `enhancement` labels for new feature ideas

Before starting work on a significant change, please open an issue to discuss your approach.

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/fastlangml.git
cd fastlangml

# Add upstream remote
git remote add upstream https://github.com/pnrajan/fastlangml.git
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Or using conda
conda create -n fastlangml python=3.10
conda activate fastlangml
```

### 3. Install Dependencies

```bash
# Install package in development mode with all dependencies
pip install -e ".[dev,all]"

# Or use make
make dev
```

### 4. Verify Installation

```bash
# Run tests
make test

# Run linter
make lint

# Run type checker
make typecheck
```

---

## Project Architecture

Understanding the codebase structure helps you contribute effectively:

```
src/fastlangml/
├── __init__.py              # Public API exports
├── detector.py              # Main FastLangDetector class
├── api.py                   # Simple detect() function
├── config.py                # Configuration classes
├── exceptions.py            # Custom exceptions
│
├── backends/                # Language detection backends
│   ├── __init__.py          # Backend factory & registry
│   ├── base.py              # Abstract Backend interface
│   ├── fasttext_backend.py  # FastText implementation
│   ├── langdetect_backend.py# langdetect implementation
│   ├── lingua_backend.py    # Lingua implementation
│   └── pycld3_backend.py    # pyCLD3 implementation
│
├── ensemble/                # Multi-backend voting
│   ├── __init__.py
│   ├── voting.py            # Voting strategies
│   └── confusion.py         # Language confusion resolution
│
├── context/                 # Conversation tracking
│   ├── __init__.py
│   └── conversation.py      # ConversationContext
│
├── hints/                   # Word-to-language hints
│   ├── __init__.py
│   ├── dictionary.py        # HintDictionary
│   └── persistence.py       # File loading/saving
│
├── preprocessing/           # Text preprocessing
│   ├── __init__.py
│   ├── name_filter.py       # Proper noun filtering
│   └── script_filter.py     # Script detection
│
├── codeswitching.py         # Code-switching detection
│
├── cli/                     # Command-line interface
│   └── main.py              # CLI entry point
│
└── data/                    # Static data files
    └── default_hints.json   # Default hint dictionary
```

### Key Components

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| `detector.py` | Main detection orchestration | `FastLangDetector` |
| `backends/` | Individual detector implementations | `Backend`, `FastTextBackend`, etc. |
| `ensemble/voting.py` | Combine multiple backend results | `VotingStrategy`, `WeightedVoting` |
| `context/` | Track conversation history | `ConversationContext` |
| `hints/` | Word-level language hints | `HintDictionary` |

---

## Making Changes

### 1. Create a Branch

```bash
# Update your local main
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Your Changes

- Write clear, focused commits
- Include tests for new functionality
- Update documentation if needed

### 3. Keep Your Branch Updated

```bash
# Regularly sync with upstream
git fetch upstream
git rebase upstream/main
```

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_detector.py -v

# Run specific test
python -m pytest tests/test_detector.py::TestDetector::test_basic -v

# Run with coverage
make test-cov

# Run tests matching a pattern
python -m pytest tests/ -k "context" -v
```

### Writing Tests

Tests live in the `tests/` directory. Follow these conventions:

```python
# tests/test_myfeature.py

import pytest
from fastlangml import detect, FastLangDetector


class TestMyFeature:
    """Tests for MyFeature functionality."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance."""
        return FastLangDetector()

    def test_basic_case(self, detector):
        """Test basic functionality."""
        result = detector.detect("Hello world")
        assert result.lang == "en"

    def test_edge_case_empty_string(self, detector):
        """Test edge case with empty input."""
        result = detector.detect("")
        assert result.lang == "und"

    @pytest.mark.parametrize("text,expected", [
        ("Bonjour", "fr"),
        ("Hola", "es"),
        ("Guten Tag", "de"),
    ])
    def test_multiple_languages(self, detector, text, expected):
        """Test detection across multiple languages."""
        result = detector.detect(text)
        assert result.lang == expected
```

### Test Requirements

- All new features must have tests
- Bug fixes should include a regression test
- Maintain or improve code coverage
- Tests should be deterministic (no flaky tests)

---

## Code Style

We use **ruff** for linting and formatting.

### Linting

```bash
# Check for issues
make lint

# Auto-fix issues
make fix

# Format code
make format
```

### Style Guidelines

1. **Line Length**: 100 characters maximum
2. **Imports**: Sorted with isort (handled by ruff)
3. **Type Hints**: Required for all public APIs
4. **Docstrings**: Google style

```python
def detect(text: str, context: ConversationContext | None = None) -> DetectionResult:
    """Detect the language of the given text.

    Args:
        text: The input text to analyze.
        context: Optional conversation context for multi-turn detection.

    Returns:
        DetectionResult containing language code and confidence.

    Raises:
        DetectionError: If detection fails unexpectedly.

    Example:
        >>> result = detect("Hello world")
        >>> result.lang
        'en'
    """
```

### Type Checking

```bash
make typecheck
```

All code should pass mypy strict mode for the `src/fastlangml` package.

---

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes bug nor adds feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process, dependencies, etc. |

### Examples

```bash
# Feature
git commit -m "feat(backends): add support for XLM-RoBERTa backend"

# Bug fix
git commit -m "fix(context): prevent crash on empty conversation history"

# Documentation
git commit -m "docs(readme): add code-switching detection examples"

# Breaking change
git commit -m "feat(api)!: change detect() return type to DetectionResult

BREAKING CHANGE: detect() now returns DetectionResult instead of str"
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass (`make test`)
- [ ] Code is linted (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventions

### 2. Submit PR

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template
4. Link related issues

### 3. PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Follows code style guidelines
```

### 4. Review Process

1. Maintainers will review your PR
2. Address feedback and push updates
3. Once approved, a maintainer will merge

### Tips for Quick Reviews

- Keep PRs focused and small (< 400 lines ideal)
- Write clear descriptions
- Respond to feedback promptly
- Be open to suggestions

---

## Issue Guidelines

### Bug Reports

Include:
1. FastLangML version (`fastlangml --version`)
2. Python version (`python --version`)
3. Operating system
4. Installed backends
5. Minimal reproducible example
6. Expected vs actual behavior
7. Full error traceback (if applicable)

**Template:**
```markdown
## Bug Description
Brief description of the bug.

## Environment
- FastLangML version: x.x.x
- Python version: 3.x.x
- OS: macOS/Linux/Windows
- Backends installed: fasttext, lingua

## Reproduction Steps
1. Step one
2. Step two

## Code Example
```python
from fastlangml import detect
result = detect("problematic text")
# Expected: "en"
# Actual: "fr"
```

## Error Traceback
```
Traceback (most recent call last):
...
```
```

### Feature Requests

Include:
1. Problem you're trying to solve
2. Proposed solution
3. Alternative approaches considered
4. Willingness to implement

---

## Release Process

Releases are handled by maintainers. The process:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v1.2.3`
4. Push tag: `git push origin v1.2.3`
5. GitHub Actions builds and publishes to PyPI

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

---

## Questions?

- Open a [GitHub Discussion](https://github.com/pnrajan/fastlangml/discussions)
- Check existing issues for similar questions
- Read the [documentation](README.md)

Thank you for contributing to FastLangML!
