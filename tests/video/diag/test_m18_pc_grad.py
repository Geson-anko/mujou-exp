"""The gradient check of M18 stage 4 (diagnosis): at the 15th step the PC-ALM
net's gradients are BP's to float32's rounding, one step earlier the check
finds enc1's off, and ``compare``'s numbers of hand-made gradients."""

import math
from dataclasses import replace

import pytest
import torch

from exp.video.diag.m18_pc_grad import bp_grads, compare, named, pc_grads
from exp.video.schema import AEConf, OptimConf, RunConf
from exp.video.vae.modules import build_vae
from tests.video.packed import packed_conf, stacks


def _conf(learn_interval: float) -> RunConf:
    """The real layout (4 encoder nodes, the latent, 2 decoder nodes) at a
    small width; ``pc_grads`` sets the batch to the stacks'."""
    conf = packed_conf(
        "pc", streams=1, learn_interval=learn_interval, token_dim=4, depth=4, nodes=4
    )
    return replace(
        conf,
        program="vae",
        ae=AEConf(decoder_depth=2, sigma0_sq=0.05),
        optim_vision=OptimConf(kind="sgd", lr=0.1, warmup=0.0),
    )


def _rows(learn_interval: float) -> list[dict]:
    conf = _conf(learn_interval)
    mods = build_vae(conf, seed=conf.seed)
    x = stacks(3)
    g = torch.Generator().manual_seed(5)
    u = torch.rand(x.shape, generator=g)
    eps = torch.randn(3, *mods.code_shape, generator=g)
    rows = compare(bp_grads(mods, x, u, eps), pc_grads(mods, conf, x, u, eps))
    assert [r["name"] for r in rows] == list(named(mods))
    return rows


class TestCheck:
    def test_at_the_fifteenth_step_every_tensor_is_bps(self):
        """29 ms: every tensor with a gradient within float32's rounding of
        BP's (a relative error of a few 1e-7 on this net)."""
        rows = _rows(29e-3)
        assert all(r["norm"] > 0 for r in rows)
        assert max(r["rel"] for r in rows) < 1e-5
        assert min(r["cos"] for r in rows) > 1 - 1e-9

    def test_one_step_earlier_it_finds_enc1(self):
        """27 ms: enc1's tensors (``encoder.stages.0``) are off, the rest
        are BP's."""
        for r in _rows(27e-3):
            if r["name"].startswith("encoder.stages.0."):
                assert r["rel"] > 0.1, r["name"]
            else:
                assert r["rel"] < 1e-5, r["name"]

    def test_no_learning_within_the_frame_is_an_error(self):
        conf = _conf(math.inf)
        mods = build_vae(conf, seed=conf.seed)
        x = stacks(1)
        eps = torch.zeros(1, *mods.code_shape)
        with pytest.raises(RuntimeError, match="no step"):
            pc_grads(mods, conf, x, torch.zeros(x.shape), eps)


class TestCompare:
    def test_the_numbers_of_hand_made_gradients(self):
        """BP (3, 4) against (3, 4.1): norm 5, difference 0.1, rel 0.02,
        cos (9 + 16.4) / (5 * |(3, 4.1)|); a zero gradient has no rel."""
        bp = {"a": torch.tensor([3.0, 4.0]), "z": torch.zeros(2)}
        pc = {"a": torch.tensor([3.0, 4.1]), "z": torch.tensor([0.0, 1e-9])}
        a, z = compare(bp, pc)
        assert a["norm"] == pytest.approx(5.0)
        assert a["diff"] == pytest.approx(0.1, rel=1e-6)
        assert a["rel"] == pytest.approx(0.02, rel=1e-6)
        want = (9 + 4 * 4.1) / (5 * math.hypot(3, 4.1))
        assert a["cos"] == pytest.approx(want, rel=1e-6)
        assert z["rel"] is None and z["cos"] is None
        assert z["diff"] == pytest.approx(1e-9)
