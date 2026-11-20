"""``exp.vision.iid``: the negative ELBO against ``torch.distributions``, the
learning rate's schedule by hand, the units drawn, and a tiny run read by
``stream_eval`` (tiny synthetic Walking Tours of ``test_eval``: videos A, B,
C of 20 frames, H held out; stacks of 2 frames of 16 px)."""

import json
import math

import pytest
import torch
import yaml
from torch.distributions import Normal, kl_divergence

from exp.lib import ckpt
from exp.vision.compose import to_dict
from exp.vision.eval import stacks_at
from exp.vision.iid import Units, _optimizer, lr_at, main, neg_elbo
from exp.vision.models import build_vae, mup_lr
from exp.vision.schema import (
    DataConf,
    MeasureConf,
    ModelConf,
    OptimConf,
    RunConf,
    ToursConf,
)
from exp.vision.stream_eval import evaluate_stream
from tests.vision.test_eval import SIZE, N, random_tours


def conf_of(**model) -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=N)),
        model=ModelConf(latent=4, width=64, **model),
        optim=OptimConf(kind="rmsprop"),  # muP's checks; iid's AdamW ignores it
        measure=MeasureConf(
            duration=3.0, ckpt_times=[0.0, 3.0], clips=3, windows=1, posthoc_seeds=1
        ),
    )


@pytest.fixture
def root(tmp_path):
    return random_tours(tmp_path / "d")


def test_neg_elbo_is_the_gaussian_log_likelihood_and_kl():
    conf = conf_of()
    mods = build_vae(conf, seed=0)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(256, (3, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
    u = torch.rand(x.shape, generator=g)
    eps = torch.randn(3, 4, generator=g)
    with torch.no_grad():
        mods.likelihood.log_sigma.fill_(-1.3)
        got = neg_elbo(mods, x, u, eps)
        mu, logvar = mods.posterior(x)
        q = Normal(mu, (0.5 * logvar).exp())
        y_hat = mods.decoder(mu + q.scale * eps).reshape(3, -1)
        y = ((x.float() + u) / 256).reshape(3, -1)
        nll = -Normal(y_hat, math.exp(-1.3)).log_prob(y).sum(1)
        kl = kl_divergence(q, Normal(0.0, 1.0)).sum(1)
    assert float(got["nll"]) == pytest.approx(float(nll.mean()), rel=1e-5)
    assert float(got["kl"]) == pytest.approx(float(kl.mean()), rel=1e-5)
    assert float(got["loss"]) == pytest.approx(float((nll + kl).mean()), rel=1e-5)
    sq = (y - y_hat).square().sum(1).mean()
    assert float(got["sq"]) == pytest.approx(float(sq), rel=1e-5)


def test_lr_rises_linearly_then_decays_by_a_cosine_to_zero():
    kw = {"lr": 2.0, "warmup": 4, "steps": 14}
    assert [lr_at(s, **kw) for s in range(4)] == [0.5, 1.0, 1.5, 2.0]
    assert lr_at(4, **kw) == pytest.approx(2.0)
    assert lr_at(9, **kw) == pytest.approx(1.0)  # half way through the decay
    assert lr_at(14, **kw) == pytest.approx(0.0)


def test_mup_groups_scale_the_learning_rate_by_mup_lr():
    conf = conf_of(param="mup", scale_decoder=True)
    mods = build_vae(conf, seed=0)
    opt = _optimizer(mods, conf, 1e-3)
    mult = mup_lr(mods, conf.model)
    seen = 0
    for g in opt.param_groups:
        for p in g["params"]:
            assert g["lr"] == pytest.approx(1e-3 * mult[p])
            seen += 1
    assert seen == len(mods.parameters())
    assert {g["mult"] for g in opt.param_groups} == {1.0, 4.0}  # width 64 = 256 / 4


@pytest.mark.parametrize(("data", "count"), [("span", 30), ("all", 60)])
def test_units_draw_kept_stacks_of_their_units(root, data, count):
    conf = conf_of()
    units = Units(conf, root, data, seed=0)
    assert units.count == count
    _, kept = stacks_at(conf, root, range(count))
    drawn = units.draw(40)
    assert drawn.shape == (40, N, 3, SIZE, SIZE)
    # every draw is one of the kept stacks of the units (not the first of a video)
    flat = kept.flatten(1)
    assert all((flat == x.flatten()).all(1).any() for x in drawn)
    if data == "span":  # units 0-29: A and half of B, never C
        _, c = stacks_at(conf, root, range(41, 60))
        assert not any((c.flatten(1) == x.flatten()).all(1).any() for x in drawn)


def test_a_tiny_run_is_read_by_stream_eval(root, tmp_path, monkeypatch):
    run = tmp_path / "run"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(to_dict(conf_of())))
    # batch 2 of frames of 0.1 s: a step is 0.2 s, so 0.6 s and 1.2 s are not
    # stamps of CKPT_T; 60 s is step 300
    monkeypatch.setattr("exp.vision.iid.CKPT_T", (0.0, 0.4, 0.5, 1.0))
    argv = ["--config", str(cfg), "--data", "span", "--lr", "1e-3", "--steps", "7"]
    argv += ["--batch", "2", "--warmup", "2", "--log-every", "3", "--device", "cpu"]
    argv += ["--root", str(root), "--out", str(run)]
    main(argv)
    times = sorted(float(p.stem) for p in (run / "ckpt").glob("*.pt"))
    assert times == [0.0, 0.4, 1.0, 1.4]  # 0.5 s is no step; the last is step 7
    head = ckpt.load_eval(
        ckpt.eval_path(run, 1.4), build_vae(conf_of(), seed=0).modules()
    )
    assert head["frame"] == 14
    rows = [json.loads(x) for x in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [3, 6, 7]
    assert all(math.isfinite(r["loss"]) for r in rows)
    out = evaluate_stream(
        run,
        lambda c: build_vae(c, seed=c.seed),
        root=root,
        fit_steps=2,
        spacing={
            "past_every": 0.5,
            "future_every": 1.0,
            "future": 2.0,
            "probe_every": 0.1,
            "block": 0.5,
        },
    )
    summ = [json.loads(x) for x in (out / "summary.jsonl").read_text().splitlines()]
    assert [r["t"] for r in summ] == times
    assert all(math.isfinite(r["span"]) for r in summ)
