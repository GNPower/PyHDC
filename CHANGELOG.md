# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-06-27

### Added

- Data encoders in the new `pyhdc.encoders` package. Each wraps an `Encoding`
  instance and maps a value, feature vector, or batch to a dimension-first
  `(D, B)` Hypervector via `encode` (or calling the encoder directly). Codebook
  encoders: `Empty`, `Identity`, `Random`, `Level`, `Thermometer`, `Circular`.
  Functional encoders: `Projection`, `Sinusoid`, `Density`, `FractionalPower`.
  Family-specific encoders raise `NotImplementedError` where the family has no
  definition (`Identity` on VTB/MBAT/BSDC, Thermometer/Density on continuous or
  phase families, Projection on BSC/BSDC, FractionalPower outside FHRR and the HRR
  family). `Identity` returns the binding-identity element (the `e` where
  `bind(x, e) == x`): all-ones for MAP, all-zeros for BSC, the impulse for the HRR
  family, zero phase for FHRR.
- Family-aware basis builders in the new `pyhdc.components.basis` package:
  `empty`, `identity`, `random`, `level`, `circular`, `thermometer`, plus
  `family_endpoints`. Each returns a `(D, count)` codebook in the encoding's value
  domain and backend.
- Cross similarity. `similarity(A, B, mode="cross")` with `A=(D, P)` and
  `B=(D, M)` returns the full `(P, M)` matrix of every column of A against every
  column of B, backed by a single matmul with no `(D, P, M)` intermediate.
  Available on `Encoding.similarity`, `Hypervector.similarity`, and the new
  module-level `pyhdc.similarity`. Implemented for Cosine, Hamming, Overlap, and
  Angle. An encoding whose metric is outside that set raises `NotImplementedError`
  so the caller can fall back to a per-pair loop. Binary metrics cast to float64
  for a BLAS matmul, cosine guards a zero-norm column (scores 0, not nan).
- Module-level convenience function `pyhdc.similarity`.
- Composable component helpers, each in an operation-named module: random-selection
  bundling `randsel` / `multirandsel` and additive `multiset` / `multibundle` in
  `pyhdc.components.bundling`, multiplicative `multibind` in `pyhdc.components.binding`,
  and `hard_quantize` / `soft_quantize` in `pyhdc.components.quantization`.
- `MAP_I_Bits` gains a `bit_width` parameter to set the signed saturation width
  explicitly (overrides `mask`).

### Fixed

- `Encoding.zeros` now works on the torch backend. It previously passed the
  encoding's numpy dtype straight to `torch.zeros`, which raised a `TypeError`. It
  now builds in numpy and converts, preserving the dtype.
- `MAP_I_Bits` now honors its bit width. The post-bundle saturation bounds and the
  storage dtype are derived from `mask` (which must be `2**k - 1`) or the new
  `bit_width`, instead of being hard-coded to int32 with the `mask` ignored. The
  default `mask=(2**32) - 1` is unchanged (int32 bounds, int32 storage). A narrow
  width now saturates correctly (an 8-bit mask clips to `[-128, 127]` and stores
  int8), a width wider than 32 widens the storage dtype (up to int64) so the sum
  no longer wraps on cast.

### Changed

- **Breaking (narrow):** `MAP_I_Bits` rejects a `mask` that is not of the form
  `2**k - 1` (contiguous low bits). Such a value was previously accepted and
  silently ignored (always clipping at int32). It now raises `ValueError`. Pass
  `bit_width=k` for an explicit k-bit limit. Default construction is unaffected.

## [2.1.0] - 2026-06-18

### Added

- Multi-dimensional `(D, N, M)` batches. `generate(size=(D, N, M, ...))` builds a
  dimension-first tensor where axis 0 is always the hypervector dimension and the
  trailing axes are the batch.
- `axis=` on `bundle` and `similarity`. `bundle(batch, axis=k)` folds a chosen
  batch axis (defaults to the last one, so `(D, N)` still collapses to `(D,)` and
  `(D, N, M)` collapses to `(D, N)`); a tuple of axes is supported for the additive
  bundlers. `similarity(batch, axis=k)` splits index 0 vs the rest along axis `k`
  for a `(D, N, M, ...)` batch. Axis 0 is never a legal reduce axis.
- `bind` and `unbind` batch automatically. The element-wise binders (MAP
  multiply, BSC xor, FHRR angle add/sub) broadcast a batch natively (a single
  `(D,)` key binds against every column; mixed ranks align by trailing-axis
  broadcasting); every other binder (convolution, shifting, matrix, VTB, CDT) is
  applied per column internally. A batched `bind(A, B)` returns one batched
  `(D, N)` Hypervector without `batch_dim`.
- Two-input `similarity` broadcasts over trailing axes, so `(D, N)` vs
  `(D, N, M)` returns an `(N, M)` score array.
- First-class `permute` (cyclic shift along the dimension axis), `inverse`
  (binding inverse), `negative` (bundling inverse), and `normalize`, available on
  `Encoding`, on `Hypervector`, and as module-level functions. `permute` works
  for every encoding; `inverse`/`negative`/`normalize` are defined per family and
  raise `NotImplementedError` where the algebra has none (e.g. MAP_C inverse, BSC
  normalize).
- Operator overloading on `Hypervector`: `+` (bundle), `*` (bind), `/` (unbind),
  `~` (inverse), `>>`/`<<` (permute by `+`/`-k`). Operators dispatch to the
  encoding, so they stay per-family correct; a non-hypervector operand yields a
  standard `TypeError`.
- Module-level `unbind`, `permute`, `inverse`, `negative`, `normalize` to match
  the existing `bundle`/`bind` convenience functions.
- `BSDC_THIN` is now exported from the top-level package (it was previously only
  reachable via `pyhdc.encodings`).

### Changed

- **Breaking:** the misspelled `BernoulliBiploar` element generator is renamed to
  `BernoulliBipolar`. `pyhdc.components.elements.BernoulliBiploar` is no longer
  importable; update any direct import to the corrected name. Element generators
  are low-level internals, and the public encodings that use it (`MAP_I`,
  `MAP_I_Bits`, `MAP_B`) are unchanged.
- Batched generation has a vectorized fast path for the i.i.d. element generators
  (bipolar/binary/uniform/normal/sparse): the whole batch is drawn in one
  `(D, *batch)` call. It is reproducible under a fixed seed for a given shape but
  is no longer value-identical to generating the vectors one at a time. Dropping
  that cross-consistency guarantee removes a full-array transpose copy (about
  10-24% faster than the prior order-preserving draw). Ordered/custom generators
  and `SparseSegmented` keep the per-vector loop and still match their sequential
  output.
- Non-batch-safe binders (circular convolution/correlation, shifting, segment
  shifting, matrix binding, VTB, context-dependent thinning) are applied per
  column when `bind`/`unbind` receives a batch, returning one batched result.
  (They previously produced a wrong result silently; 2.0 single-vector inputs are
  unaffected.)
- Randomized-bundling metadata `random_zone_count` is a Python `int` for a `(D,)`
  result (unchanged) and a per-output-vector count array for a batched result.
- `ElementAdditionBits` (MAP_I_Bits bundling) sums in a wide (int64) accumulator
  and clips the total once, saturating at the bounds. This replaces the previous
  per-addition saturating clip, so results change when the running sum would have
  saturated mid-accumulation; it is vectorized (no Python loop) and supports a
  tuple of axes.
- `DisjunctionThinned` (BSDC_THIN bundling) thins a batched result without a
  per-column Python loop: each surviving column keeps a uniformly random
  `ceil(D * density)`-subset of its set bits via a vectorized random-key
  selection.
- `bundle(array, batch_dim=k)` on a 3D array no longer Python-loops over the
  split slices: it reduces the other batch axis in one vectorized op and splits
  the result into the same list of hypervectors (about 8x faster on a
  1000x20x500 array). Ragged nested-list inputs, `batch_dim=0`, and 4D+ arrays
  keep the per-group path. For tie-randomizing bundlers the random values at tie
  coordinates now differ from the previous per-group draws (still random;
  `batch_dim` has no fixed-seed guarantee). `axis=` remains the preferred
  vectorized form (returns a single tensor instead of a list).

### Deprecated

- `batch_dim` on `bundle`/`bind`/`unbind` is deprecated and will be removed in a
  future release. Pass a batched array directly (operations batch automatically)
  or use `axis=` on `bundle`. Passing `batch_dim` now emits a `DeprecationWarning`.

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

[Unreleased]: https://github.com/GNPower/PyHDC/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/GNPower/PyHDC/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/GNPower/PyHDC/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/GNPower/PyHDC/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/GNPower/PyHDC/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/GNPower/PyHDC/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/GNPower/PyHDC/compare/v0.0.1...v1.0.0
[0.0.1]: https://github.com/GNPower/PyHDC/releases/tag/v0.0.1
