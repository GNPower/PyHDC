# Contributing to PyHDC

Thank you for your interest in contributing to PyHDC! This document explains how to set up a development environment, run tests, and submit pull requests.

## Development Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/GNPower/PyHDC.git
cd PyHDC
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
make install
# or manually:
pip install -r requirements.txt -r requirements_dev.txt -e .
```

### 3. Install pre-commit hooks

```bash
pre-commit install
```

The hooks run autoflake, isort, black, pylint, and mypy on every commit. This keeps the codebase consistent without requiring a manual `make lint` step.

## Running Checks Locally

```bash
make lint         # autoflake + isort + black + pylint
make type-check   # mypy
make security     # bandit
make test         # pytest with coverage
make bench        # performance benchmarks (slow — run manually)
make build        # build wheel and sdist
```

## Project Layout

```
pyhdc/              Core library
  encodings/        All 14 encoding types (MAP, HRR, BSC, …)
  generation/       Random number generators (LCG, LFSR, PCG, …)
  recovery/         Sequence recovery algorithms (BM, hill climb, …)
  components/       Similarity metrics, binding/bundling ops
tests/
  conftest.py       Shared fixtures
  test_*.py         Unit tests (one file per module)
  benchmarks/       pytest-benchmark suites (excluded from normal runs)
```

## Running Tests

```bash
# Full suite with coverage
pytest --cov=pyhdc --cov-report=term-missing

# Single file
pytest tests/test_encodings.py -v

# Benchmarks only
pytest tests/benchmarks/ --benchmark-only --benchmark-autosave
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests for any changed behaviour.
3. Ensure `make lint` and `make test` pass locally.
4. Open a pull request against `main`. The CI suite (lint, type-check, security, test) must be green before merging.
5. Squash commits if the history is noisy; otherwise separate commits are fine.

## Versioning

PyHDC uses [bump2version](https://github.com/c4urself/bump2version) to manage version numbers. The canonical version lives in `pyproject.toml` (`[project].version`) and is kept in sync with `pyhdc/__init__.py` (`__version__`) via `.bumpversion.cfg`.

To cut a release:

```bash
# Patch: 1.0.0 → 1.0.1
bump2version patch

# Minor: 1.0.0 → 1.1.0
bump2version minor

# Major: 1.0.0 → 2.0.0
bump2version major

# Push the commit and the new tag
git push --follow-tags
```

Creating a GitHub Release from the tag triggers the `pypi.yml` workflow and publishes to PyPI.

## Code Style

- **Formatting**: black (line length 88).
- **Import order**: isort (black-compatible profile).
- **Linting**: pylint with project rules in `pyproject.toml`.
- **Type hints**: encouraged; mypy runs in `continue-on-error` mode so missing annotations will not block CI.
- **Comments**: only when the *why* is non-obvious.

## Reporting Bugs

Please open an issue on [GitHub](https://github.com/GNPower/PyHDC/issues) with a minimal reproducible example. For security vulnerabilities, see [SECURITY.md](SECURITY.md).