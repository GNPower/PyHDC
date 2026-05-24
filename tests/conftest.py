"""Shared pytest fixtures for the pyhdc test suite."""

import pytest

import pyhdc

DIM = 512


@pytest.fixture(scope="module")
def map_c_enc():
    return pyhdc.MAP_C(dimension=DIM)


@pytest.fixture(scope="module")
def map_i_enc():
    return pyhdc.MAP_I(dimension=DIM)


@pytest.fixture(scope="module")
def map_b_enc():
    return pyhdc.MAP_B(dimension=DIM)


@pytest.fixture(scope="module")
def hrr_enc():
    return pyhdc.HRR(dimension=DIM)


@pytest.fixture(scope="module")
def fhrr_enc():
    return pyhdc.FHRR(dimension=DIM)


@pytest.fixture(scope="module")
def bsc_enc():
    return pyhdc.BSC(dimension=DIM)


@pytest.fixture
def map_c_hv(map_c_enc):
    return map_c_enc.generate()


@pytest.fixture
def bsc_hv(bsc_enc):
    return bsc_enc.generate()
