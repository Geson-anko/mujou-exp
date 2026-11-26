r"""``exp.vision.width_guard`` (the guard of the linear horizon,
``plans/jepa.md``).

The oracles: every anchor's candidates lie in its block and its triple has no
cut; a decoder whose velocity is 0 draws the noise itself, so every
condition and every code draws the same images (the noise is the anchors'
and the samples', nothing else); the identification by hand on features
placed by hand (its own, a neighbour, the pick's hub); the midpoint and the
extrapolation commute with the standardization (an affine map); the
cheats: strength 0 the identity, a drop idempotent, the past mean by hand;
the anchors' paired differences of 0 have no spread; and the yard's,
the cheats' and a run's jobs end to end on tiny runs (a cheap feature in
place of Inception's).
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, override

import pytest
import torch
from torch import Tensor, nn

from exp.vision.eval import Standardizer
from exp.vision.geom_eval import _sets_at
from exp.vision.results.v18c.judge_width import cheat_stage, guard_stage
from exp.vision.width_eval import evaluate_run as width_run
from exp.vision.width_guard import (
    Guard,
    GuardConf,
    _identify,
    cheat_codes,
    diff_se,
    evaluate_cheat,
    evaluate_run,
    evaluate_yard,
    guard_anchors,
    load_decoder,
    save_decoder,
)
from tests.lib.test_flow_decoder import SMALL
from tests.vision.test_fill_eval import TINY as FILL_TINY
from tests.vision.test_gen_eval import TINY as GEN_TINY, features
from tests.vision.test_geom_eval import SPLICED, TINY, tours_with_zurich
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY
from tests.vision.test_width_eval import WTINY, tiny_sets

GTINY = GuardConf(lags=(1, 2), anchors=6, samples=2, reach=2, picture_every=2,
                  k=2, cheat_windows=(1, 2), cheat_drops=(1,), cheat_lag=1,
                  cheat_count=2, cheat_length=8)  # fmt: skip


class Still(nn.Module):
    """A velocity of 0: Euler's method keeps the noise."""

    @override
    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        return torch.zeros_like(x)


class TestAnchors:
    def test_the_candidates_lie_in_the_block_and_the_triple_has_no_cut(self):
        _, ws = tiny_sets()
        for lag in (1, 2):
            t = guard_anchors(ws, lag, 6, 2)
            assert len(t) == 6
            lo, hi = t - 2 * lag, t + 3 * lag
            assert bool((ws.frame_block[lo] == ws.frame_block[hi + ws.span]).all())
            _, inside = ws.centres(lag)
            assert bool(torch.isin(t, inside).all())


def tiny_guard() -> Guard:
    _, ws = tiny_sets()
    return Guard(SPLICED, ws, features=features, local=None, conf=GTINY, gen=GEN_TINY,
                 dev=torch.device("cpu"))  # fmt: skip


class TestDrawings:
    def test_the_noise_alone_whatever_the_condition_and_the_code(self):
        g = tiny_guard()
        lg = g.lags[0]
        a = g._draw(Still(), torch.randn(len(lg.t), 3, dtype=torch.float64), lg.noise)
        b = g._draw(Still(), torch.randn(len(lg.t), 5, dtype=torch.float64), lg.noise)
        assert torch.equal(a, b)
        want = ((lg.noise.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
        assert torch.equal(a, want)

    def test_the_rows_of_a_still_decoder_read_alike(self):
        g = tiny_guard()
        cond = torch.randn(g._ws.n, 3, dtype=torch.float64)
        rows = g.code_rows(Still(), cond, cond[:20], {"code": "x"})
        for r in rows:
            shares = {v["share"] for v in r["read"]["ext"].values()}
            assert len(shares) == 1  # every condition the same drawings


class TestIdentification:
    def test_by_hand(self):
        cand = torch.eye(3, dtype=torch.float64)[None].expand(2, 3, 3) * 2
        center = torch.zeros(3, dtype=torch.float64)
        own = cand[:, 1:2]  # the middle one, the true
        hits, picks = _identify(own, cand, center)
        assert hits.tolist() == [1.0, 1.0] and picks.tolist() == [[1], [1]]
        side = torch.stack([cand[0, 0], cand[1, 2]])[:, None]
        hits, picks = _identify(side, cand, center)
        assert hits.tolist() == [0.0, 0.0] and picks.tolist() == [[0], [2]]


class TestAffine:
    def test_the_midpoint_and_extrapolation_commute_with_the_standardizer(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(50, 4, generator=g) * 3 + 1
        std = Standardizer(z)
        a, b = z[:10], z[10:20]
        torch.testing.assert_close(
            std(2 * a - b), 2 * std(a) - std(b), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(std((a + b) / 2), (std(a) + std(b)) / 2, rtol=1e-5,
                                   atol=1e-5)  # fmt: skip


class TestCheats:
    def test_strength_0_drop_and_past_mean(self):
        s = torch.tensor([0.1, 0.5, 0.3], dtype=torch.float64)
        x = torch.arange(12, dtype=torch.float64).reshape(4, 3)
        fns = cheat_codes(x, s, replace(GTINY, cheat_windows=(1,), cheat_drops=(2,)))
        assert torch.equal(fns["base"](x), x)
        d = fns["drop-2"](x)
        assert torch.equal(d[:, 1], torch.zeros(4, dtype=torch.float64))
        assert torch.equal(d[:, 2], torch.zeros(4, dtype=torch.float64))
        assert torch.equal(fns["drop-2"](d), d)  # idempotent
        m = fns["mean-1"](x)
        assert torch.equal(m[0], x[0])
        assert torch.equal(m[2], (x[1] + x[2]) / 2)


class TestSE:
    def test_no_difference_has_no_spread(self):
        a = torch.rand(
            40, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        blocks = torch.arange(40) // 4
        from exp.vision.geom_eval import GeomConf

        assert diff_se(a, a, blocks, replace(GeomConf(), boot=50)) == 0.0


def write_uncond(path: Path, dim: int) -> None:
    torch.manual_seed(0)
    from exp.lib.flow_decoder import FlowUNet

    net = FlowUNet(dim, **SMALL)
    save_decoder(path, net, None, replace(GEN_TINY, net=SMALL), dim)


class TestJobs:
    @pytest.mark.slow
    def test_the_yard_the_cheats_and_a_run_end_to_end(self, tmp_path):
        root = tours_with_zurich(tmp_path / "d")
        conf = replace(conf_of(), device="cpu")
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        dec = tmp_path / "gen"
        write_uncond(dec / "ckpt" / "dec-yardstick-uncond-d0.pt", 4)
        net, mean, sd = load_decoder(dec / "ckpt" / "dec-yardstick-uncond-d0.pt",
                                     torch.device("cpu"))  # fmt: skip
        assert mean is None and sd is None and not net.training
        kw: dict[str, Any] = dict(root=root, features=features, guard=GTINY, gen=GEN_TINY,
                                  gconf=TINY, wconf=WTINY, fconf=FILL_TINY, spacing=EVERY,
                                  span=SPAN)  # fmt: skip
        out = evaluate_yard(conf, sets_dir, tmp_path / "guard", fit={"pca": (0, 1)},
                            dec=dec, **kw)  # fmt: skip
        assert (out / "ckpt" / "dec-width-pca-d1.pt").exists()
        rows = json.loads((out / "pca" / "rows-d0.json").read_text())
        assert [r["lag"] for r in rows] == [1, 2]
        assert set(rows[0]["read"]["ext"]) == {"future", "now", "ext", "shuffle", "pos"}
        assert set(rows[0]["read"]["mid"]) == {"past", "now", "future", "mid"}
        assert (out / "pca" / "d0-1.png").exists()
        unc = json.loads((out / "uncond" / "rows-d0.json").read_text())
        assert "pos" not in unc[0]["read"]["ext"]
        ctl = json.loads((out / "controls.json").read_text())
        assert set(ctl[0]["read"]["ext"]) == {"feature_ext", "pixel_ext", "real_next"}
        # the decoders learned once: a second call loads them
        again = evaluate_yard(
            conf, sets_dir, tmp_path / "guard", fit={"pca": (0,)}, **kw
        )
        assert (
            json.loads((again / "pca" / "rows-d0.json").read_text())[0]["read"]
            == rows[0]["read"]
        )
        cheat = evaluate_cheat(conf, sets_dir, tmp_path / "guard" / "cheat", **kw)
        names = sorted(
            p.name for p in cheat.iterdir() if p.is_dir() and p.name != "ckpt"
        )
        assert names == ["base", "drop-1", "mean-1", "mean-2"]
        width = json.loads((cheat / "mean-2" / "width.json").read_text())
        assert {r["stratum"] for r in width if r["part"] == "lin"} == {"common"}
        # the description reads them
        got_g = guard_stage(tmp_path, TINY)
        assert got_g["sigma_dec"] is not None and set(got_g["codes"]) == {
            "pca",
            "uncond",
        }
        got_c = cheat_stage(tmp_path, WTINY, TINY, got_g["sigma_dec"])
        assert set(got_c["families"]) == {"mean", "drop"}
        # a run: its width first, then its guard (a row of no lag where none wins)
        run = tiny_run(root, tmp_path, name="run")
        width_run(run, steps=(0,), sets_dir=sets_dir, root=root, device="cpu", gconf=TINY,
                  wconf=WTINY, fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        got = evaluate_run(run, steps=(0,), sets_dir=sets_dir, device="cpu", **kw)
        rows = json.loads((got / "rows" / "000000.json").read_text())
        assert rows and rows[0]["part"] in ("no_lag", "guard")
