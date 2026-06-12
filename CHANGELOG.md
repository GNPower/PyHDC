# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-06-12

### Added

- Batched hypervectors with a dimension-first `(D, N)` layout (each column is a
  hypervector). `enc.generate(size=(D, N))` returns a single `(D, N)` Hypervector;
  `bundle` collapses a `(D, N)` batch to one `(D,)` prototype; `bind`/`unbind` of
  two equal-shaped batches operate per column (element-wise binding only — MAP
  multiply, BSC xor); `similarity` accepts batches (see below).
- `Hypervector.select(indices)`: select hypervectors along the batch axis (axis 1)
  for numpy and torch (numpy/list indices are moved to a `long` tensor on the
  tensor's device).
- `pyhdc.stack(hypervectors)`: backend-agnostic combine of hypervectors/batches
  along the batch axis (a 1D `(D,)` vector is treated as `(D, 1)`), so a prototype
  and a `(D, N)` codebook combine into `(D, N + 1)`.
- Global default backend/device: `pyhdc.prefer_torch()`, `pyhdc.prefer_cuda()`,
  `pyhdc.prefer_numpy()`, `pyhdc.prefer_cpu()`, `pyhdc.get_default_backend()`,
  `pyhdc.get_default_device()`. Encodings created without an explicit
  `backend`/`device` inherit these defaults.
- Single-argument and broadcast similarity: `Encoding.similarity(batch)` returns
  the similarity of column 0 against each remaining column; two equal-shaped
  `(D, N)` batches give `N` per-column scores; a vector against a `(D, N)` batch
  broadcasts to `N` scores.
- Per-operation input normalizers (`_normalize_bundling`, `_normalize_binding`,
  `_normalize_similarity`, `_normalize_thinning`) so every HDC operation receives
  an identically-shaped input.

### Changed

- **Breaking**: batched 2D similarity is now dimension-first `(D, N)` (compare
  along axis 0) instead of the batch-first `(N, D)` introduced in 1.1.0.
  `CosineSimilarity(batch)` now returns `sim(col_0, col_i)`; pass `(D, N)`-shaped
  arrays where you previously passed `(N, D)`. Single-vector `(D,)` similarity is
  unchanged.
- Encoding `backend`/`device` now default to `None` and resolve from the global
  preference; an explicit `backend`/`device` argument still overrides it.

### Fixed

- Bundling a single 2D `(D, N)` batch now reduces across the batch axis to one
  `(D,)` hypervector (previously returned unchanged, with band/normalization
  thresholds using `N = 1`). Affects every bundling function.
- `generate(size=(D, N))` now produces vectors in sequence as columns, so under a
  fixed seed it matches `N` successive `generate(size=D)` calls, and works for the
  `NormalReal` (HRR/VTB/MBAT) and sparse generators (previously raised on a tuple
  `size`).

## [1.1.0] - 2026-05-24

### Added

- `BSDC_THIN` encoding: Binary Sparse Distributed Code with post-bundling random thinning
  to enforce a density constraint (Rachkovskij 2001, Schlegel et al. 2022). Configurable
  `density` parameter (default 0.5). Uses Shifting/InverseShifting for binding.
- `DisjunctionThinned` bundling function: bitwise OR followed by random thinning to a target
  density. Available in `pyhdc.components.bundling`.
- `similarity_remap` parameter on all encoding classes: optional callable applied to every
  similarity result before returning, enabling custom output range transformations.
- `remap_to_unit` function in `pyhdc.components.similarity`: remaps the standard [-1, 1]
  similarity range to [0, 1] (0.5 = orthogonal). Works on scalars, numpy arrays, and tensors.
- PyTorch support for all four similarity functions (`CosineSimilarity`, `HammingDistance`,
  `Overlap`, `AngleDistance`), including all three calling conventions.
- Batched similarity calling conventions for all similarity functions:
  - `(a, b)` where both are 2D: pairwise per-row similarities, returns 1D array
  - `(arr,)` single 2D array: similarity of `row_0` against each remaining row, returns 1D array

### Changed

- **Breaking**: `HammingDistance` now returns [-1, 1] instead of [0, 1]. Use
  `remap_to_unit` to restore the previous [0, 1] behaviour.
- **Breaking**: `Overlap` now returns [-1, 1] instead of [0, 1]. Use `remap_to_unit`
  to restore the previous [0, 1] behaviour.
- All similarity functions now use `_normalize_inputs` for consistent input handling
  (Hypervectors, raw numpy arrays, and raw torch tensors all accepted).
- PyTorch binary bundling (`Disjunction`, `DisjunctionThinned`) replaced sequential
  `logical_or` loop with vectorized `torch.stack(...).any(dim=0)`.

### Fixed

- `MAP_I_Bits` integer overflow on Python 3.9: clip bounds now use `np.iinfo(np.int32).min/max`
  instead of exceeding int32 range.
- All similarity functions now handle PyTorch tensors without falling back to numpy.

## [1.0.1] - 2026-05-23

### Changed

- Added `README.md` with PyPI, CI, coverage, and licence badges, installation instructions,
  and a quick-start example. (Omitted from the v1.0.0 tag; this patch ensures it appears
  in the PyPI release page.)

## [1.0.0] - 2026-05-23

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

[Unreleased]: https://github.com/GNPower/PyHDC/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/GNPower/PyHDC/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/GNPower/PyHDC/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/GNPower/PyHDC/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GNPower/PyHDC/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/GNPower/PyHDC/releases/tag/v0.0.1
