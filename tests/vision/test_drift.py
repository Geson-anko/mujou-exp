"""``exp.vision.drift``: the mean relative move of each module's weights
between checkpoints ``every`` seconds apart, by hand on tiny modules."""

import json
import math

import pytest
import torch
from torch import nn

from exp.lib import ckpt
from exp.vision.drift import drift, main, relative_move


def _save(run, t: float, w: float, b: float, s: float) -> None:
    lin = nn.Linear(1, 2)
    lin.weight.data = torch.tensor([[w], [0.0]])
    lin.bias.data = torch.tensor([b, 0.0])
    sigma = nn.Module()
    sigma.register_parameter("log_sigma", nn.Parameter(torch.tensor(s)))
    ckpt.save_eval(
        run,
        program="vae",
        group="g",
        t=t,
        frame=round(t / 0.1),
        modules={"decoder": lin, "likelihood": sigma},
    )


@pytest.fixture
def run(tmp_path):
    # (t, weight, bias, ln sigma); 240 is before start, 330 off the grid
    for t, w, b, s in [
        (240.0, 9.0, 9.0, 9.0),
        (300.0, 1.0, 1.0, 1.0),
        (330.0, 9.0, 9.0, 9.0),
        (360.0, 2.0, 1.0, 1.0),
        (420.0, 2.0, 3.0, 2.0),
    ]:
        _save(tmp_path, t, w, b, s)
    return tmp_path


def test_relative_move_by_hand():
    a = {"x": torch.tensor([3.0, 4.0]), "y": torch.tensor([0.0])}
    b = {"x": torch.tensor([3.0, 4.0]), "y": torch.tensor([5.0])}
    assert relative_move(a, b, ["x", "y"]) == pytest.approx(1.0)
    assert relative_move(a, b, ["x"]) == 0.0
    assert math.isnan(relative_move(a, b, []))


def test_drift_by_hand(run):
    """Pairs (300, 360) and (360, 420); the weight [[w], [0]] moves 1 then 0,
    of norms 1 then 2; with the bias [b, 0], |theta| is sqrt(2) then sqrt(5)."""
    d = drift(run)
    assert d["decoder"]["matrices"] == pytest.approx((1.0 / 1.0 + 0.0) / 2)
    assert d["decoder"]["all"] == pytest.approx(
        (1.0 / math.sqrt(2) + 2.0 / math.sqrt(5)) / 2
    )
    assert math.isnan(d["likelihood"]["matrices"])
    assert d["likelihood"]["all"] == pytest.approx((0.0 + 1.0) / 2)


def test_too_few_checkpoints_are_refused(run):
    with pytest.raises(ValueError):
        drift(run, start=400.0)


def test_main_prints_a_line_per_run(run, capsys):
    main([str(run)])
    row = json.loads(capsys.readouterr().out)
    assert row["run"] == run.name
    assert row["decoder"]["matrices"] == pytest.approx(0.5)
