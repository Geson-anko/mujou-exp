"""The digest of the initial weights (M17 diagnosis): equal for equal
tensors, different when one element, a name or a buffer differs."""

import torch
from torch import nn

from exp.video.diag.init_digest import digest


def _mods(seed: int) -> dict[str, nn.Linear | nn.LayerNorm]:
    torch.manual_seed(seed)
    return {"a": nn.Linear(3, 2), "b": nn.LayerNorm(2)}


def test_equal_weights_give_the_same_digest():
    assert digest(_mods(0)) == digest(_mods(0))


def test_one_element_changes_the_digest():
    m = _mods(0)
    before = digest(m)
    w = m["a"].weight
    with torch.no_grad():
        w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(9.0))
    assert digest(m) != before


def test_the_names_and_buffers_count():
    m = _mods(0)
    assert digest({"c": m["a"], "b": m["b"]}) != digest(m)
    bn = nn.BatchNorm1d(2)
    before = digest({"n": bn})
    mean = bn.running_mean
    assert mean is not None
    mean.add_(1.0)
    assert digest({"n": bn}) != before


def test_another_seed_gives_another_digest():
    assert digest(_mods(0)) != digest(_mods(1))
