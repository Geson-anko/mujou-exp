r"""``exp.vision.fill_eval`` (V18c's E1, E1e, E1g, E2).

The oracles: on synthetic sets with stub JEPAs, a predictor that outputs the
target side's tokens has ``gain`` 1 and one that outputs the fit set's mean
token 0; the readouts and the stages ``oracle``, ``mean``, ``interp`` do not
depend on the predictor (they are fit without it); a K3 encoder (every token
the same global code) has ``swap`` exactly 1/2 even where the two quarters
differ in distribution (a bright top-left, a dark bottom-right), ``mean``
exactly 1/2 for any encoder, while patch tokens choose their own quarter;
the constant prediction identifies at exactly $1/n$; arm M's ``init_m``
yardstick is its run's step 0. The block colors by hand; the
targets' spaces centred and of unit scale; the cache's key changes with the
sets. On the synthetic Walking Tours of ``test_eval``: the sets' places are
their frames, and tiny runs, the yardsticks and E1g after ``gen_eval`` run
end to end (a cheap feature in place of Inception's).
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from exp.lib import ckpt
from exp.lib.png import read_png
from exp.lib.tours.wide import finalize as wide_finalize
from exp.lib.video_io import INDEX, U8Reader, U8Writer
from exp.vision import fill_eval, gen_eval
from exp.vision.compose import to_dict
from exp.vision.fill_eval import (
    BANDS,
    E1_MASKS,
    FillConf,
    FillSets,
    Space,
    block_colors,
    build_targets,
    e1_rows,
    evaluate_gen,
    evaluate_run,
    evaluate_targets,
    evaluate_yardsticks,
    fill_sets,
    gain,
    targets_key,
)
from exp.vision.gen_eval import hard_sets
from exp.vision.jepa import JEPAModules, build_jepa, take
from exp.vision.jepa_eval import W_SEED, RandProjCode
from exp.vision.models import build_vae
from tests.vision.test_eval import SIZE, N
from tests.vision.test_gen_eval import TINY as TINY_GEN
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

TINY = FillConf(
    n_cand=8,
    grid_step=1,
    hard_reach=2,
    hard_stride=2,
    target_dim=4,
    readout_dim=8,
    color_block=4,
    outer=4,
    picture_rows=4,
    picture_scale=1,
)
BAND = 8  # the synthetic wide data's band


def features(x: Tensor) -> Tensor:
    """A cheap feature of u8 images of any size: the 4 x 4 area means."""
    return F.adaptive_avg_pool2d(x.float() / 255, 4).flatten(1)


# -- stub JEPAs ------------------------------------------------------------------------


class PatchTokens(nn.Module):
    """An encoder whose token is its patch's mean of each channel ``(B, T,
    C)``; it keeps the tokens of the whole images it saw last."""

    def __init__(self, patch: int = 4) -> None:
        super().__init__()
        self._p = patch
        self.full = torch.empty(0)

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        b, c, h, w = images.shape
        p = self._p
        t = images.reshape(b, c, h // p, p, w // p, p).mean((3, 5)).flatten(2)
        self.full = t.transpose(1, 2)
        return self.full if keep is None else take(self.full, keep)


class Global(nn.Module):
    """A K3 encoder: every token is the images' mean of each channel."""

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        t = 16 if keep is None else keep.shape[1]
        return images.mean((2, 3))[:, None].expand(-1, t, -1)


class Peek(nn.Module):
    """A predictor that outputs the target side's tokens of the hidden
    positions (from the encoder's whole images)."""

    def __init__(self, encoder: PatchTokens, proj: nn.Linear) -> None:
        super().__init__()
        self._enc, self._proj = [encoder], [proj]

    @override
    def forward(self, context: Tensor, keep: Tensor, hide: Tensor) -> Tensor:
        return self._proj[0](take(self._enc[0].full, hide))


class Const(nn.Module):
    """A predictor of a fixed token per position ``value`` ``(T, d)``."""

    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self._v = value

    @override
    def forward(self, context: Tensor, keep: Tensor, hide: Tensor) -> Tensor:
        return self._v[hide]


def identity(c: int) -> nn.Linear:
    proj = nn.Linear(c, c)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(c))
        proj.bias.zero_()
    return proj


def mods_of(encoder: nn.Module, predictor: nn.Module | None = None) -> JEPAModules:
    proj = identity(3 * N)
    if predictor is None:
        assert isinstance(encoder, PatchTokens)
        predictor = Peek(encoder, proj)
    return JEPAModules(encoder, proj, predictor)


# -- synthetic sets -----------------------------------------------------------------------


def stacks(b: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, (b, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)


def skewed(b: int, seed: int) -> Tensor:
    """Random stacks whose top-left quarter is bright (192 to 255) and
    bottom-right dark (0 to 63): quarters of different distributions."""
    h = SIZE // 2
    x = stacks(b, seed)
    x[..., :h, :h] = x[..., :h, :h] // 4 + 192
    x[..., h:, h:] = x[..., h:, h:] // 4
    return x


def synthetic(cand: Tensor) -> FillSets:
    n = len(cand)
    idx = torch.arange(n)
    hard_q, among = hard_sets(idx, n, 1, 1)
    return FillSets(
        fit=stacks(48, 1),
        calib=stacks(16, 2),
        grid_last=cand[:, -1],
        cand_stacks=cand,
        cand=idx,
        hard_q=hard_q,
        hard_among=among,
        ident={},
    )


@pytest.fixture(scope="module")
def plain():
    sets = synthetic(stacks(12, 3))
    return sets, build_targets(sets, features, replace(TINY, readout_dim=1024))


def run_e1(mods: JEPAModules, sets, targets, k_run: int = 4) -> list[dict]:
    conf = replace(TINY, readout_dim=1024)  # no PCA: the inputs are at most 72 wide
    return e1_rows(
        mods, sets, targets, conf, expand=1, k_run=k_run, dev=torch.device("cpu")
    )


def by(rows: list[dict], **labels) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in labels.items())]


METRICS = ("r2", "id_easy", "id_hard")


class TestE1:
    def test_a_perfect_prediction_gains_1(self, plain):
        sets, targets = plain
        rows = run_e1(mods_of(PatchTokens()), sets, targets)
        for kind in ("incep", "color"):
            for m in (*E1_MASKS, "Q", "H"):
                (o,) = by(rows, target=kind, mask=m, stage="oracle")
                (p,) = by(rows, target=kind, mask=m, stage="pred")
                (u,) = by(rows, target=kind, mask=m, stage="mean")
                assert gain(p["r2"], o["r2"], u["r2"]) == pytest.approx(1, abs=1e-9)
                assert p["hits_easy"] == o["hits_easy"]

    def test_the_mean_token_gains_0_and_identifies_at_chance(self, plain):
        sets, targets = plain
        enc = PatchTokens()
        with torch.no_grad():
            mean = identity(3 * N)(
                enc(stacks(48, 1).reshape(48, N * 3, SIZE, SIZE).float() / 255)
            ).mean(0)
        rows = run_e1(mods_of(enc, Const(mean)), sets, targets)
        n = len(sets.cand)
        for kind in ("incep", "color"):
            for m in E1_MASKS:
                (o,) = by(rows, target=kind, mask=m, stage="oracle")
                (p,) = by(rows, target=kind, mask=m, stage="pred")
                (u,) = by(rows, target=kind, mask=m, stage="mean")
                # float32 tokens: the two means differ by their rounding
                assert gain(p["r2"], o["r2"], u["r2"]) == pytest.approx(0, abs=1e-5)
                assert u["id_easy"] == p["id_easy"] == pytest.approx(1 / n)

    def test_the_readouts_never_see_the_predictor(self, plain):
        sets, targets = plain
        enc = PatchTokens()
        a = run_e1(mods_of(enc), sets, targets)
        b = run_e1(mods_of(enc, Const(torch.zeros(16, 3 * N))), sets, targets)
        for stage in ("oracle", "mean", "interp"):
            assert by(a, stage=stage) == by(b, stage=stage)
        assert by(a, stage="pred") != by(b, stage="pred")

    def test_k3_swaps_at_exactly_one_half_and_patch_tokens_do_not(self):
        # the quarters differ in distribution: a choice between the two
        # quarters' targets would find a K3 code's own one by the offset
        sets = synthetic(skewed(12, 4))
        targets = build_targets(sets, features, replace(TINY, readout_dim=1024))
        k3 = run_e1(mods_of(Global(), Const(torch.zeros(16, 3 * N))), sets, targets)
        swaps = by(k3, mask="S")
        assert {(r["target"], r["stage"]) for r in swaps} == {
            (k, s)
            for k in ("incep", "color")
            for s in ("oracle", "pred", "mean", "shuf")
        }
        assert all(r["swap"] == 0.5 for r in swaps)
        patch = run_e1(mods_of(PatchTokens()), sets, targets)
        (r,) = by(patch, mask="S", target="color", stage="oracle")
        assert r["swap"] > 0.9  # its own quarter's colors, read linearly
        # the mean token: no direction, whatever the encoder (not rounding's)
        assert all(r["swap"] == 0.5 for r in by(patch, mask="S", stage="mean"))


class TestEncoding:
    def test_codes_of_the_sets_keep_no_graph(self):
        # outside no_grad, as E2's and arm M's PCA of E1g call it
        mods = build_jepa(conf_of(), seed=0)
        x = stacks(3, 5)
        cpu = torch.device("cpu")
        assert torch.is_grad_enabled()
        for fn in (mods.mu, RandProjCode(mods).mu):
            assert not fill_eval._apply(fn, x, cpu).requires_grad


class TestTargets:
    def test_block_colors_by_hand(self):
        x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float64).reshape(2, 3, 4, 4)
        pix = torch.zeros(4, 4, dtype=torch.bool)
        pix[:, 2:] = True  # the right half: blocks (0, 1) and (1, 1)
        got = block_colors(x.to(torch.uint8), pix, 2)
        want = (
            torch.stack(
                [x[:, :, 0:2, 2:4].mean((2, 3)), x[:, :, 2:4, 2:4].mean((2, 3))], 2
            ).flatten(1)
            / 255
        )
        torch.testing.assert_close(got, want)

    def test_a_space_centres_and_scales_the_fit_set(self):
        g = torch.Generator().manual_seed(0)
        raw = torch.randn(50, 7, generator=g, dtype=torch.float64) * 3 + 1
        for dim in (None, 4):
            u = Space.fit(raw, dim)(raw)
            torch.testing.assert_close(
                u.mean(0), torch.zeros(u.shape[1], dtype=torch.float64)
            )
            assert float(u.var(0).mean()) == pytest.approx(1)

    def test_the_key_names_the_sets(self, plain):
        sets, _ = plain
        a = replace(sets, ident={"fit_ks": [1, 2, 3]})
        b = replace(sets, ident={"fit_ks": [1, 2, 4]})
        assert targets_key(a, TINY) != targets_key(b, TINY)
        assert targets_key(a, TINY) == targets_key(replace(a), TINY)


# -- end to end on the synthetic Walking Tours --------------------------------------------


@pytest.fixture
def root(tmp_path) -> Path:
    from tests.vision.test_eval import random_tours

    return random_tours(tmp_path / "d")


def wide_of(root: Path, out: Path) -> Path:
    """Wide data of ``root``'s videos: random bands of ``BAND`` beside each
    frame."""
    g = torch.Generator().manual_seed(9)
    for d in sorted(p for p in root.iterdir() if (p / INDEX).is_file()):
        frames = U8Reader(d)
        meta = {
            "video": d.name,
            "data": root.name,
            "data_index_sha256": "0",
            "fps": 10.0,
            "geometry": {"band": BAND, "size": SIZE},
        }
        with U8Writer(out / d.name, (3, SIZE, SIZE + 2 * BAND), meta) as w:
            for f in frames[0 : len(frames)]:
                bands = torch.randint(
                    256, (2, 3, SIZE, BAND), generator=g, dtype=torch.uint8
                )
                w.append(torch.cat([bands[0], f, bands[1]], 2))
    return wide_finalize(out)


KW = {"fconf": TINY, "spacing": EVERY}


class TestSets:
    def test_the_places_are_the_sets_last_frames(self, root):
        sets = fill_sets(conf_of(), root, TINY, span=SPAN, spacing=EVERY)
        assert sets.places is not None
        for name in ("fit", "calib", "grid"):
            got = torch.stack([U8Reader(root / v)[f] for v, f in sets.places[name]])
            assert torch.equal(got, sets.last(name)), name
        assert torch.equal(sets.cand_stacks[:, -1], sets.grid_last[sets.cand])


class TestEvaluate:
    @pytest.mark.slow
    def test_targets_runs_yardsticks_and_e1g(self, root, tmp_path):
        cache = tmp_path / "cache"
        conf = conf_of()
        path = evaluate_targets(cache, conf=conf, root=root, features=features, **KW)
        first = torch.load(path / "targets.pt", weights_only=True)["values"]
        assert not any(k.startswith("B/") for k in first)
        wide = wide_of(root, tmp_path / "w")
        again = evaluate_targets(
            cache, conf=conf, root=root, features=features, wide=wide, **KW
        )
        assert again == path  # the same sets: E2's added, E1's kept
        values = torch.load(path / "targets.pt", weights_only=True)["values"]
        assert torch.equal(values["U/R/incep"]["grid"], first["U/R/incep"]["grid"])
        assert {f"B/{b}/color" for b in BANDS} <= set(values)
        # a run of patch 4 (V18's grid) and an arm M of patch 2 (spread by 2)
        run = tiny_run(root, tmp_path)
        out = evaluate_run(run, steps=(4,), root=root, cache=cache, **KW)
        names = sorted(p.name for p in (out / "rows").iterdir())
        assert names == ["000004-e1.json", "000004-e1e.json"]
        assert json.loads((out / "config.json").read_text())["seeds"]["w"] == W_SEED
        other = {"fconf": replace(TINY, readout_dim=4), "spacing": EVERY}
        with pytest.raises(ValueError, match="other settings"):  # rows all there
            evaluate_run(run, steps=(4,), root=root, cache=cache, **other)
        e1 = json.loads((out / "rows" / "000004-e1.json").read_text())
        assert {r["stage"] for r in e1} == {"oracle", "pred", "mean", "shuf", "interp"}
        assert all(math.isfinite(r["r2"]) for r in e1 if "r2" in r)
        assert {r["in_dist"] for r in e1 if r["mask"] == "Q_tl"} == {True}
        assert len(by(e1, mask="S")) == 8
        side = json.loads(
            (run / "posthoc_side" / "rows" / "000004-z_w.json").read_text()
        )
        assert {r["band"] for r in side} == {*BANDS, "side", "outer"}
        arm_m = tiny_run(
            root, tmp_path, "--patch", "2", "--sigreg-on", "mean", name="m"
        )
        e1e = json.loads((out / "rows" / "000004-e1e.json").read_text())
        assert {r["code"] for r in e1e} == {"h_V"} and len(e1e) == 2 * (9 + 2)
        evaluate_run(arm_m, steps=(4,), parts=("e1",), root=root, cache=cache, **KW)
        e1 = json.loads(
            (arm_m / "posthoc_fill" / "rows" / "000004-e1.json").read_text()
        )
        assert len(by(e1, mask="S")) == 8 and all(
            math.isfinite(r["swap"]) for r in by(e1, mask="S")
        )
        # the yardsticks
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        v16 = ckpt.save_eval(
            vae_run, program="vae", group="iid", t=1.0, frame=10,
            modules=build_vae(conf, seed=5).modules(),
        )  # fmt: skip
        yard = evaluate_yardsticks(
            tmp_path / "y", conf=conf, v16=v16, m_run=arm_m, root=root, cache=cache,
            codes=("pca", "v16", "init_m", "ext", "interp", "mean"),
            pca_components=4, zpca_dim=4, **KW,
        )  # fmt: skip
        e1e = json.loads((yard / "rows" / "yardstick-e1e.json").read_text())
        assert {r["code"] for r in e1e} == {"color_V", "pca_V", "ext"}
        rows = json.loads(
            (
                tmp_path / "y" / "posthoc_side" / "rows" / "yardstick-mean.json"
            ).read_text()
        )
        assert all(r["id_easy"] == pytest.approx(1 / TINY.n_cand) for r in rows)
        # init_m is arm M's own step 0 (the same fit stacks, PCA and seed)
        evaluate_run(
            arm_m, steps=(0,), parts=("e2",), root=root, cache=cache, zpca_dim=4,
            **KW,
        )  # fmt: skip
        side_dir = tmp_path / "y" / "posthoc_side" / "rows"
        yard_m = json.loads((side_dir / "yardstick-init_m.json").read_text())
        step0 = json.loads(
            (arm_m / "posthoc_side" / "rows" / "000000-z.json").read_text()
        )
        drop = ("run", "step", "t", "code")
        assert [{k: v for k, v in r.items() if k not in drop} for r in step0] == [
            {k: v for k, v in r.items() if k not in drop} for r in yard_m
        ]
        # E1g after G's decoders of z
        gen_eval.evaluate_run(
            run, steps=(4,), codes=("z",), root=root, features=features,
            gconf=replace(TINY_GEN, n_cand=TINY.n_cand), spacing=EVERY,
        )  # fmt: skip
        out = evaluate_gen(
            run, steps=(4,), root=root, cache=cache, features=features, **KW
        )
        g = json.loads((out / "rows" / "000004-e1g.json").read_text())
        assert {(r["mask"], r["stage"]) for r in g} == {
            (m, s) for m in E1_MASKS for s in ("full", "oracle", "pred", "mean")
        }
        assert all(0 <= r["id_easy"] <= 1 and 0 <= r["vis_easy"] <= 1 for r in g)
        image = read_png(out / "g-000004-R.png")
        assert image.shape[:2] == (TINY.picture_rows * (SIZE + 2), 5 * (SIZE + 2))

    def test_runs_without_targets_are_refused(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        with pytest.raises(FileNotFoundError, match="fill_eval targets"):
            evaluate_run(run, steps=(4,), root=root, cache=tmp_path / "none", **KW)
