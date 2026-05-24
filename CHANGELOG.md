# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive unit test suite covering all 14 encoding types, all 7 generator families,
  all components, recovery algorithms, exceptions, and the hypervector API
- Performance benchmark suite using `pytest-benchmark` for encodings (generate, bundle, bind, similarity)
- pytest configuration in `pyproject.toml` with coverage reporting (`fail_under = 49`)
- mypy static type checking configuration in `pyproject.toml`
- `Makefile` with targets: `install`, `lint`, `type-check`, `security`, `test`, `bench`, `clean`, `build`, `help`
- `.pre-commit-config.yaml` with hooks for autoflake, isort, black, pylint, and mypy
- `CONTRIBUTING.md` with developer setup and PR process
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1)
- `SECURITY.md` with vulnerability reporting guidance
- `CHANGELOG.md` (this file)
- Codecov upload step in CI test workflow
- Optional benchmark job in CI test workflow (triggered via `workflow_dispatch`)
- Bandit security scanning job in CI lint workflow
- mypy type checking step in CI lint workflow
- TestPyPI publish workflow (`pypi-test.yml`) with OIDC Trusted Publishing, manual trigger

### Fixed
- All internal imports changed from `hdc.` namespace to `pyhdc.` namespace (45 files)
- `DefaultGenerator._next_word` integer overflow for `word_size >= 32` (used `np.int64` to avoid overflow)
- `MBAT.bind` storing tuple as hypervector data (non-dict metadata from `MatrixMultiplication` now handled correctly)
- `MAP_I_Bits` wrong keyword argument names (`min`/`max` → `min_val`/`max_val`) for `ElementAdditionBits`
- `FeistelCounterGenerator` non-deterministic round key generation (now uses seeded local `random.Random`)
- `__version__` in `pyhdc/__init__.py` corrected from `"1.0.0"` to `"0.0.1"` to match PyPI and `pyproject.toml`
- Removed `setuptools_scm>=8` from `[build-system].requires` (version managed by bump2version, not git tags)
- Removed `[tool.distutils.bdist_wheel]` `universal = true` (incorrect py2/py3 flag)

### Changed
- `pypi-test.yml` updated to actually upload to TestPyPI (test.pypi.org) with OIDC, manual trigger
- CI workflows updated to use current action versions (checkout@v4, setup-python@v5)
- `requirements_dev.txt` expanded with: `pytest-benchmark`, `mypy`, `bandit`, `pre-commit`

## [0.0.1] - 2024-01-01

### Added
- Initial template release to PyPI
- Core hypervector encoding types: MAP_C, MAP_I, MAP_I_Bits, MAP_B, HRR, HRR_NoNorm, HRR_ConstNorm,
  FHRR, VTB, MBAT, BSC, BSDC_CDT, BSDC_S, BSDC_SEG
- Random number generator families: LCG, DLFSR, LFSR, LCA, PCG, Xorshift, ShiftedCounter
- Recovery algorithm framework: Berlekamp-Massey (Fibonacci/Galois), LCG, Xorshift, PCG,
  CA, DLFSR recovery algorithms
- NumPy backend (PyTorch optional)
- `pyproject.toml` with setuptools build configuration
- GitHub Actions CI: lint, test, PyPI publish workflows

[Unreleased]: https://github.com/GNPower/PyHDC/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/GNPower/PyHDC/releases/tag/v0.0.1