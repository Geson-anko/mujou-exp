"""What a run ran on, and the settings that make it deterministic.

``collect`` gathers the environment of an attempt (``provenance/attempt-<n>.json``).
``configure_determinism`` fixes the threads and the kernels of torch, and
``assert_free_threaded`` stops a run in which something brought the GIL back
(a free-threaded Python enables it again when an extension that does not
declare itself free-threaded is imported; ``torch.optim`` imports
``torch._dynamo``, which does so on Linux).

The reproducible unit: on CPU, the same snapshot and settings give the same
bytes of the metrics files; on CUDA, not even on the same kind of GPU (the
deterministic algorithms there only warn), so the kind of device is recorded.
"""

import datetime
import importlib.machinery
import os
import platform
import socket
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from exp.lib.rundir import EnvError

__all__ = [
    "CUBLAS_WORKSPACE",
    "EnvError",
    "assert_free_threaded",
    "collect",
    "configure_determinism",
]

CUBLAS_WORKSPACE = (":4096:8", ":16:8")  # what cuBLAS needs to be deterministic
_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHON_GIL",
    "FLEET_HOST",
)
# The modules at the previous check: none before the first, which names all.
_seen: set[str] = set()


def _free_threaded_build() -> bool:
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _gil_enabled() -> bool:
    is_enabled = getattr(sys, "_is_gil_enabled", None)
    return True if is_enabled is None else bool(is_enabled())


def _extensions(names: set[str]) -> list[str]:
    """The modules of ``names`` that are C extensions, sorted."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    out = []
    for name in names:
        file = getattr(sys.modules.get(name), "__file__", None)
        if isinstance(file, str) and file.endswith(suffixes):
            out.append(name)
    return sorted(out)


def assert_free_threaded(where: str) -> None:
    """Raise ``EnvError`` if the GIL is enabled in a free-threaded Python, or
    ``torch._dynamo`` is imported (``torch.optim`` was used; it enables the GIL
    on Linux), whatever the build. ``where`` names the point of the check in
    the message, which lists the C extensions imported since the previous
    check (all of them at the first).

    In a Python built with the GIL, the GIL itself is not a failure
    (``collect`` records it); ``torch._dynamo`` still is.
    """
    global _seen
    now = set(sys.modules)
    new, _seen = now - _seen, now
    reasons = []
    if _free_threaded_build() and _gil_enabled():
        reasons.append("the GIL is enabled")
    if "torch._dynamo" in now:
        reasons.append("torch._dynamo is imported (torch.optim?)")
    if reasons:
        exts = ", ".join(_extensions(new)) or "none"
        raise EnvError(
            f"{where}: {' and '.join(reasons)}; C extensions imported since the"
            f" previous check: {exts}"
        )


def configure_determinism(device: str) -> dict[str, Any]:
    """Make torch deterministic for a run on ``device`` and return the settings.

    One intra-op thread, no TF32 in matmuls and cuDNN, and the deterministic
    algorithms (on CUDA warn only: some kernels have no deterministic
    version). They are set by ``torch._C._set_deterministic_algorithms``, not
    ``torch.use_deterministic_algorithms``, which imports ``torch._dynamo``.

    Raises:
        EnvError: ``OMP_NUM_THREADS`` is not ``"1"`` (OpenMP threads of the
            hooks' worker threads would split the products and round
            differently), or on CUDA ``CUBLAS_WORKSPACE_CONFIG`` is not one of
            ``CUBLAS_WORKSPACE`` or the device is not visible. Nothing is
            changed then.
    """
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp != "1":
        raise EnvError(f"OMP_NUM_THREADS must be '1', not {omp!r}")
    cuda = device.startswith("cuda")
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cuda and cublas not in CUBLAS_WORKSPACE:
        raise EnvError(
            f"CUBLAS_WORKSPACE_CONFIG must be one of {CUBLAS_WORKSPACE} on CUDA,"
            f" not {cublas!r}"
        )
    if cuda:
        _check_visible(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    # Not torch.use_deterministic_algorithms: it imports torch._dynamo.
    torch._C._set_deterministic_algorithms(True, warn_only=cuda)
    return _determinism()


def _check_visible(device: str) -> None:
    """Raise ``EnvError`` unless the CUDA device ``device`` is visible."""
    if not torch.cuda.is_available():
        raise EnvError(f"{device}: CUDA is not visible")
    index = torch.device(device).index
    count = torch.cuda.device_count()
    if index is not None and index >= count:
        raise EnvError(f"{device}: CUDA sees {count} device(s)")


def _determinism() -> dict[str, Any]:
    return {
        "num_threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _affinity() -> list[int] | None:
    get = getattr(os, "sched_getaffinity", None)  # Linux only
    return None if get is None else sorted(get(0))


def collect(job: Mapping[str, Any], *, attempt: int) -> dict[str, Any]:
    """The environment of attempt ``attempt`` of ``job`` (a ``job.json``),
    for ``provenance/attempt-<n>.json``.

    ``job["config"]["device"]`` (``"cpu"`` if absent) says whether to name the
    GPU; ``job["config"]["data"]["tours"]["data_version"]`` (M15's prepared
    data) is copied, ``None`` for a run without it.
    """
    config = job.get("config", {})
    device = str(config.get("device", "cpu"))
    tours = (config.get("data") or {}).get("tours") or {}
    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.device(device))
    return {
        "run_id": job.get("run_id"),
        "attempt": attempt,
        "host": os.environ.get("FLEET_HOST") or socket.gethostname(),
        "started": datetime.datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "free_threaded_build": _free_threaded_build(),
        "gil_enabled": _gil_enabled(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu": gpu,
        "cpu": _cpu_model(),
        "affinity": _affinity(),
        "env": {k: os.environ.get(k) for k in _ENV},
        "determinism": _determinism(),
        "data_version": tours.get("data_version"),
    }
