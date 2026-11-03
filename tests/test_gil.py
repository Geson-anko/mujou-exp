"""The process of a run stays free-threaded: hydra's compose, mujou and the
steps of a network do not bring the GIL back, and ``torch.optim`` is caught.

Each check runs in a fresh interpreter: once the GIL is enabled in a process,
it stays so, and would leak into the other tests.
"""

import json
import os
import subprocess
import sys
import sysconfig
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="needs a free-threaded Python",
)

_RUN = """
import json, sys

import torch
from exp.lib.provenance import EnvError, assert_free_threaded, configure_determinism
from exp.video.compose import compose

import mujou.pc as pc
import mujou.testing
from mujou import TensorSpec

assert_free_threaded("import")
configure_determinism("cpu")
conf = compose(["+experiment=m14/s0", "seed=1"])
tau = pc.TimeConstants(h=conf.time.h, lam=conf.time.lam, r=conf.time.r)
torch.manual_seed(conf.seed)
net = pc.Chain(
    torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh()),
    torch.nn.Linear(3, 1),
    x=TensorSpec((2,)), y=TensorSpec((1,)), loss=pc.loss.MSELoss(), tau=tau,
    optimizer=pc.optim.SGD(tau=1.0), learn_interval=None,
)
yes = torch.tensor(True)
sample = pc.msg.Sample(x=torch.ones(2), y=torch.ones(1), mask=yes, reset=yes)
with mujou.testing.run(net, inputs={net.prev: sample}) as run:
    for _ in range(20):
        run.step(tau.period)
assert_free_threaded("build")
out = {"gil_after_steps": sys._is_gil_enabled()}
torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
try:
    assert_free_threaded("torch.optim")
    out["error"] = None
except EnvError as e:
    out["error"] = str(e)
print(json.dumps(out))
"""


def test_a_run_stays_free_threaded_until_torch_optim():
    """Compose a preset, build and step a small network: the GIL is still
    disabled and ``assert_free_threaded`` passes; making a ``torch.optim``
    optimizer then makes it raise ``EnvError``."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    env["OMP_NUM_THREADS"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_RUN)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.splitlines()[-1])
    assert got["gil_after_steps"] is False
    assert got["error"] is not None
    assert got["error"].startswith("torch.optim:")
