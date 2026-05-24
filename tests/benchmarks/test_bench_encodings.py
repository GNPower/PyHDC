"""Performance benchmarks for core HDC operations using pytest-benchmark."""

import pytest

import pyhdc

DIM = 10_000


@pytest.fixture(scope="module")
def map_c_enc():
    return pyhdc.MAP_C(dimension=DIM)


@pytest.fixture(scope="module")
def bsc_enc():
    return pyhdc.BSC(dimension=DIM)


@pytest.fixture(scope="module")
def hrr_enc():
    return pyhdc.HRR(dimension=DIM)


@pytest.fixture(scope="module")
def fhrr_enc():
    return pyhdc.FHRR(dimension=DIM)


# --- Generate benchmarks ---


def test_bench_map_c_generate(benchmark, map_c_enc):
    benchmark(map_c_enc.generate)


def test_bench_bsc_generate(benchmark, bsc_enc):
    benchmark(bsc_enc.generate)


def test_bench_hrr_generate(benchmark, hrr_enc):
    benchmark(hrr_enc.generate)


def test_bench_fhrr_generate(benchmark, fhrr_enc):
    benchmark(fhrr_enc.generate)


# --- Bundle benchmarks ---


def test_bench_map_c_bundle(benchmark, map_c_enc):
    hv1 = map_c_enc.generate()
    hv2 = map_c_enc.generate()
    benchmark(hv1.bundle, hv2)


def test_bench_bsc_bundle(benchmark, bsc_enc):
    hv1 = bsc_enc.generate()
    hv2 = bsc_enc.generate()
    benchmark(hv1.bundle, hv2)


def test_bench_hrr_bundle(benchmark, hrr_enc):
    hv1 = hrr_enc.generate()
    hv2 = hrr_enc.generate()
    benchmark(hv1.bundle, hv2)


# --- Bind benchmarks ---


def test_bench_map_c_bind(benchmark, map_c_enc):
    hv1 = map_c_enc.generate()
    hv2 = map_c_enc.generate()
    benchmark(hv1.bind, hv2)


def test_bench_bsc_bind(benchmark, bsc_enc):
    hv1 = bsc_enc.generate()
    hv2 = bsc_enc.generate()
    benchmark(hv1.bind, hv2)


def test_bench_hrr_bind(benchmark, hrr_enc):
    hv1 = hrr_enc.generate()
    hv2 = hrr_enc.generate()
    benchmark(hv1.bind, hv2)


# --- Similarity benchmarks ---


def test_bench_map_c_similarity(benchmark, map_c_enc):
    hv1 = map_c_enc.generate()
    hv2 = map_c_enc.generate()
    benchmark(hv1.similarity, hv2)


def test_bench_bsc_similarity(benchmark, bsc_enc):
    hv1 = bsc_enc.generate()
    hv2 = bsc_enc.generate()
    benchmark(hv1.similarity, hv2)
