import json
import os
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest
import torch

from exp.lib.provenance import EnvError, collect, configure_determinism

ROOT = Path(__file__).resolve().parents[2]
FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _python(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter of this venv at the repository root,
    with ``OMP_NUM_THREADS=1`` and without ``PYTHON_GIL`` unless ``env`` says
    otherwise (the environment the worker gives a run)."""
    base = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    base["OMP_NUM_THREADS"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(not FREE_THREADED, reason="needs a free-threaded Python")
class TestAssertFreeThreaded:
    def test_torch_optim_raises(self):
        """Making a ``torch.optim`` optimizer imports ``torch._dynamo`` (which
        enables the GIL on Linux): the check after it raises ``EnvError``."""
        out = _python(
            """
            import torch
            from exp.lib.provenance import EnvError, assert_free_threaded
            assert_free_threaded("import")
            torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
            try:
                assert_free_threaded("build")
            except EnvError as e:
                print("EnvError:", e)
            """
        )
        assert out.returncode == 0, out.stderr
        assert "EnvError: build:" in out.stdout
        assert "torch._dynamo" in out.stdout

    def test_the_message_lists_the_extensions_imported_since(self):
        """An extension imported between two checks is named; one imported
        before the first is not."""
        out = _python(
            """
            import torch
            from exp.lib.provenance import EnvError, assert_free_threaded
            assert_free_threaded("import")
            import mujou  # mujou._native, a C extension (free-threaded)
            import torch._dynamo
            try:
                assert_free_threaded("build")
            except EnvError as e:
                print(e)
            """
        )
        assert out.returncode == 0, out.stderr
        assert "mujou._native" in out.stdout
        assert "torch._C" not in out.stdout

    def test_the_first_check_names_the_extensions_imported_before_it(self):
        """An extension imported before ``provenance`` (by the program) is
        named by the first check, which is where the GIL would show."""
        out = _python(
            """
            import mujou, torch._dynamo
            from exp.lib.provenance import EnvError, assert_free_threaded
            try:
                assert_free_threaded("import")
            except EnvError as e:
                print(e)
            """
        )
        assert out.returncode == 0, out.stderr
        assert "mujou._native" in out.stdout
        assert "torch._C" in out.stdout

    def test_configure_determinism_keeps_the_gil_disabled(self):
        out = _python(
            """
            import json, sys
            from exp.lib.provenance import (
                assert_free_threaded, collect, configure_determinism)
            settings = configure_determinism("cpu")
            assert_free_threaded("determinism")
            info = collect({"run_id": "r", "config": {"device": "cpu"}}, attempt=1)
            assert_free_threaded("collect")
            print(json.dumps({"settings": settings,
                              "gil": sys._is_gil_enabled(), "info": info}))
            """
        )
        assert out.returncode == 0, out.stderr
        got = json.loads(out.stdout.splitlines()[-1])
        assert got["gil"] is False
        assert got["info"]["gil_enabled"] is False
        assert got["settings"] == {
            "num_threads": 1,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "deterministic": True,
            "warn_only": False,
        }
        assert got["info"]["determinism"] == got["settings"]


class TestConfigureDeterminism:
    @pytest.mark.parametrize("omp", ["4", None])
    def test_omp_threads_other_than_one_raise(self, monkeypatch, omp):
        if omp is None:
            monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        else:
            monkeypatch.setenv("OMP_NUM_THREADS", omp)
        with pytest.raises(EnvError, match="OMP_NUM_THREADS"):
            configure_determinism("cpu")

    def test_cuda_needs_the_cublas_workspace(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        with pytest.raises(EnvError, match="CUBLAS_WORKSPACE_CONFIG"):
            configure_determinism("cuda:0")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is visible")
    def test_cuda_not_visible_raises(self, monkeypatch):
        """Exit code 4, so the worker cordons the host's GPUs."""
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with pytest.raises(EnvError, match="CUDA is not visible"):
            configure_determinism("cuda:0")

    @pytest.mark.cuda
    def test_a_cuda_index_beyond_the_devices_raises(self, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with pytest.raises(EnvError, match="CUDA sees"):
            configure_determinism(f"cuda:{torch.cuda.device_count()}")


class TestCollect:
    def test_it_records_the_job_and_the_environment(self, monkeypatch):
        monkeypatch.setenv("FLEET_HOST", "std-09")
        monkeypatch.setenv("PYTHON_GIL", "0")
        tours = {"data_version": "abc", "held_out": "v"}
        config = {"device": "cpu", "data": {"balls": None, "tours": tours}}
        info = collect({"run_id": "m14-s1-x-s0", "config": config}, attempt=2)
        assert info["run_id"] == "m14-s1-x-s0"
        assert info["attempt"] == 2
        assert info["host"] == "std-09"
        assert info["device"] == "cpu"
        assert info["gpu"] is None
        assert info["data_version"] == "abc"
        assert info["env"]["PYTHON_GIL"] == "0"
        json.dumps(info)  # provenance/attempt-<n>.json

    def test_a_run_without_tours_has_no_data_version(self):
        config = {"device": "cpu", "data": {"balls": {}, "tours": None}}
        assert collect({"config": config}, attempt=1)["data_version"] is None
