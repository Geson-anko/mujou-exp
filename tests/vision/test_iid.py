"""``exp.vision.iid``: the negative ELBO against ``torch.distributions``, the
learning rate's schedule by hand, the units drawn, and a tiny run read by
``stream_eval`` (tiny synthetic Walking Tours of ``test_eval``: videos A, B,
C of 20 frames, H held out; stacks of 2 frames of 16 px). V18d's lags
against their log-uniform law, and its clips on videos whose frames carry
their video and their number (plans/v18d.md, O12), and the foundation's
stacks of spaced frames on the same videos (plans/foundation.md, 3: one
frame apart is ``draw``, the frames $s$ apart in one training video, the
last frame uniform where the stack fits)."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import Tensor
from torch.distributions import Normal, kl_divergence

from exp.lib import ckpt
from exp.vision.compose import to_dict
from exp.vision.eval import stacks_at
from exp.vision.iid import Units, _optimizer, draw_lags, lr_at, main, neg_elbo
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
from tests.vision.test_eval import SIZE, N, ramp, random_tours, write_tours


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


def chi2(counts: Tensor, probs: Tensor) -> tuple[float, int]:
    """Pearson's statistic of ``counts`` against ``probs`` and its degrees."""
    want = probs * counts.sum()
    return float(((counts - want).square() / want).sum()), len(counts) - 1


def under_chi2(counts: Tensor, probs: Tensor) -> bool:
    """The statistic below its mean plus 5 of its sd (the dof's chi-square)."""
    stat, dof = chi2(counts.double(), probs.double())
    return stat < dof + 5 * math.sqrt(2 * dof)


class TestDrawLags:
    @pytest.mark.parametrize(("lo", "hi"), [(8, 512), (1, 20)])
    def test_the_rounded_log_uniform_law(self, lo, hi):
        # P(n = k): the uniform's mass of [log(k - 1/2), log(k + 1/2)] in [lo, hi]
        n = draw_lags(400_000, lo, hi, torch.Generator().manual_seed(0))
        assert int(n.min()) == lo and int(n.max()) == hi
        k = torch.arange(lo, hi + 1, dtype=torch.float64)
        top, bottom = (k + 0.5).clamp(max=hi), (k - 0.5).clamp(min=lo)
        probs = (top.log() - bottom.log()) / math.log(hi / lo)
        counts = torch.bincount(n - lo, minlength=hi - lo + 1)
        assert under_chi2(counts, probs)


LENGTHS = {"A": 60, "B": 40, "C": 90, "H": 30}  # H held out
CODES = {"A": 10, "B": 20, "C": 30, "H": 40}


@pytest.fixture
def numbered(tmp_path) -> Path:
    """Videos whose frame j is of the color (its video's code, j, 0)."""
    return write_tours(
        tmp_path / "n",
        {v: ramp((CODES[v], 0, 0), (0, 1, 0), m) for v, m in LENGTHS.items()},
    )


def frames_of(x: Tensor) -> tuple[Tensor, Tensor]:
    """The video codes and the frame numbers ``(..., N)`` of stacks ``(...,
    N, 3, S, S)`` of ``numbered``."""
    return x[..., 0, 0, 0].long(), x[..., 1, 0, 0].long()


class TestClips:
    K = 4

    def test_the_k_stacks_are_n_apart_in_one_video(self, numbered):
        units = Units(conf_of(), numbered, "all", seed=0)
        n = draw_lags(3000, 1, 9, torch.Generator().manual_seed(1))
        x, ends = units.draw_clips(n, self.K)
        assert x.shape == (3000, self.K, N, 3, SIZE, SIZE)
        video, frame = frames_of(x)  # (B, K, N)
        assert (video == video[:, :1, :1]).all()  # one video a clip
        assert not (video == CODES["H"]).any()  # never the held-out
        last = frame[:, :, -1]  # (B, K): the last frames
        assert torch.equal(last, ends)
        steps = torch.arange(self.K)[None] * n[:, None]
        assert torch.equal(last, last[:, :1] + steps)  # tau + i n
        assert torch.equal(frame[:, :, 0], last - (N - 1))  # whole, oldest first
        assert int(frame.min()) >= 0
        length = {CODES[v]: m for v, m in LENGTHS.items()}
        assert all(int(e) < length[int(v)] for e, v in zip(last[:, -1], video[:, 0, 0]))

    def test_the_first_is_uniform_on_the_units_that_fit(self, numbered):
        # at n = 5: the first stacks of video v end at its frames 1..L_v - 16
        units = Units(conf_of(), numbered, "all", seed=2)
        x, _ = units.draw_clips(torch.full((6000,), 5), self.K)
        video, frame = frames_of(x[:, 0])
        fit = [
            (c, j)
            for v, c in CODES.items()
            if v != "H"
            for j in range(N - 1, LENGTHS[v] - 3 * 5)
        ]
        index = {u: i for i, u in enumerate(fit)}
        got = torch.tensor(
            [index[(int(v), int(f))] for v, f in zip(video[:, -1], frame[:, -1])]
        )
        counts = torch.bincount(got, minlength=len(fit))
        assert under_chi2(counts, torch.full((len(fit),), 1 / len(fit)))

    def test_one_stack_a_clip_is_draws(self, numbered):
        a = Units(conf_of(), numbered, "all", seed=6)
        b = Units(conf_of(), numbered, "all", seed=6)
        x, _ = a.draw_clips(torch.full((500,), 37), 1)
        assert torch.equal(x[:, 0], b.draw(500))

    def test_perm_draws_each_role_alone_from_its_units(self, numbered):
        # video C (90 frames) at n = 5: role i ends uniformly at 1 + 5i ..
        # 74 + 5i, whatever the others (ST's law of each role)
        units = Units(conf_of(), numbered, "all", seed=3)
        n = torch.full((20_000,), 5)
        x, ends = units.draw_clips(n, self.K, torch.Generator().manual_seed(4))
        video, frame = frames_of(x)
        assert (video == video[:, :1, :1]).all()  # the video's
        assert torch.equal(frame[:, :, -1], ends)
        assert torch.equal(frame[:, :, 0], ends - (N - 1))  # whole
        c = video[:, 0, 0] == CODES["C"]
        fit = LENGTHS["C"] - 3 * 5 - (N - 1)
        for i in range(self.K):
            role = ends[c, i] - (N - 1) - 5 * i
            assert int(role.min()) >= 0 and int(role.max()) < fit
            counts = torch.bincount(role, minlength=fit)
            assert under_chi2(counts, torch.full((fit,), 1 / fit))
        gap = (ends[c, 1] - ends[c, 0]).double()
        assert float(gap.std()) > 10  # not n apart: independent
        r = torch.corrcoef(ends[c][:, [0, 3]].double().T)[0, 1]
        assert abs(float(r)) < 0.05

    def test_a_clip_of_no_video_or_the_span_units_are_refused(self, numbered):
        units = Units(conf_of(), numbered, "all", seed=0)
        with pytest.raises(ValueError, match="fits in no video"):
            units.draw_clips(torch.tensor([30]), self.K)
        span = Units(conf_of(), numbered, "span", seed=0)
        with pytest.raises(ValueError, match="'all'"):
            span.draw_clips(torch.tensor([2]), self.K)


def eight_of() -> RunConf:
    """``conf_of`` of stacks of 8 frames (the foundation's)."""
    conf = conf_of()
    tours = replace(conf.data.tours, frames=8)
    return replace(conf, data=replace(conf.data, tours=tours))


class TestSpaced:
    def test_one_frame_apart_is_draw_to_the_bit(self, numbered):
        a = Units(eight_of(), numbered, "all", seed=3)
        b = Units(eight_of(), numbered, "all", seed=3)
        assert torch.equal(
            a.draw_spaced(torch.ones(400, dtype=torch.long)), b.draw(400)
        )

    def test_the_frames_are_s_apart_in_one_training_video(self, numbered):
        units = Units(eight_of(), numbered, "all", seed=4)
        s = draw_lags(3000, 1, 5, torch.Generator().manual_seed(1))
        x = units.draw_spaced(s)
        assert x.shape == (3000, 8, 3, SIZE, SIZE)
        video, frame = frames_of(x)  # (B, 8)
        assert (video == video[:, :1]).all()  # one video a stack
        assert not (video == CODES["H"]).any()  # never the held-out
        steps = torch.arange(8)[None] * s[:, None]
        assert torch.equal(frame, frame[:, :1] + steps)  # tau + i s, oldest first
        assert int(frame.min()) >= 0
        length = {CODES[v]: m for v, m in LENGTHS.items()}
        assert all(int(f) < length[int(v)] for f, v in zip(frame[:, -1], video[:, 0]))

    def test_the_last_frame_is_uniform_where_the_stack_fits(self, numbered):
        # s = 5: a stack of video v ends at its frames 35..L_v - 1
        units = Units(eight_of(), numbered, "all", seed=5)
        x = units.draw_spaced(torch.full((6000,), 5))
        video, frame = frames_of(x)
        fit = [
            (c, j) for v, c in CODES.items() if v != "H" for j in range(35, LENGTHS[v])
        ]
        index = {u: i for i, u in enumerate(fit)}
        got = torch.tensor(
            [index[(int(v), int(f))] for v, f in zip(video[:, -1], frame[:, -1])]
        )
        counts = torch.bincount(got, minlength=len(fit))
        assert under_chi2(counts, torch.full((len(fit),), 1 / len(fit)))

    def test_bad_spacings_or_the_span_units_are_refused(self, numbered):
        units = Units(eight_of(), numbered, "all", seed=0)
        with pytest.raises(ValueError, match="at least 1"):
            units.draw_spaced(torch.tensor([0]))
        with pytest.raises(ValueError, match="fits in no video"):
            units.draw_spaced(torch.tensor([13]))  # 7 x 13 = 91 >= 90
        span = Units(eight_of(), numbered, "span", seed=0)
        with pytest.raises(ValueError, match="'all'"):
            span.draw_spaced(torch.tensor([1]))
