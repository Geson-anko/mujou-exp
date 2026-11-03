import os

import pytest

# mujou runs the hooks on worker threads, where OpenMP starts with as many
# threads as cores: `torch.set_num_threads(1)` of the host reaches a worker only
# once torch initializes its threads there, and MKL does not, so a matrix
# product may split over the cores and round differently from run to run.
# OpenMP reads this when torch loads it: set it before any test imports torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test marked ``cuda(n)`` (``n`` 1 by default) on fewer than ``n``
    CUDA devices; torch is imported only for such a test."""
    marker = item.get_closest_marker("cuda")
    if marker is None:
        return
    n = int(marker.args[0] if marker.args else marker.kwargs.get("n", 1))
    import torch

    have = torch.cuda.device_count()
    if have < n:
        pytest.skip(f"needs {n} CUDA device(s), found {have}")
