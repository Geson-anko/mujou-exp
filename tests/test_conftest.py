import os

import pytest


def test_openmp_runs_one_thread() -> None:
    assert os.environ["OMP_NUM_THREADS"] == "1"


@pytest.mark.cuda(1_000_000)
def test_cuda_marker_skips_on_too_few_devices() -> None:
    pytest.fail("ran without a million CUDA devices")


@pytest.mark.cuda(n=1_000_000)
def test_cuda_marker_takes_n_by_keyword() -> None:
    pytest.fail("ran without a million CUDA devices")
