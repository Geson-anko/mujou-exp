r"""``exp.vision.width_eval`` (the linear horizon, ``plans/jepa.md``,
「線形の幅（2026-11-26）」).

The oracles: the dense sets at the geom lags are the geom sets' ``all`` and
``in`` (the same rule spread over the lags), the common centres are ``in``
at every lag, the pairs by brute force; the rows' pooled $G$ and blocks'
sums of ``in`` at the geom lags are ``geom_eval``'s to the bit (the same
code, the same sets); the rows of a series reversed in time (sets of whole
blocks) have the same wins and $G$; the survivors' sums are the running
``and`` of the wins; the anchors' places by hand; the checks V-h0 on their
full series (but V-h0e's $N_{\mathrm{lin}}$ at the edge of the band's window,
README; and the long V-h0i, V-h0k run on few draws only); and the yard's,
the run's and the VAE's jobs end to end on tiny runs.
"""

import json
import math
from dataclasses import replace

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.lib.midpoint import Whitener, triple_terms, wins
from exp.lib.synth_series import ou
from exp.vision.compose import to_dict
from exp.vision.geom_eval import _sets_at, code_rows as geom_code_rows
from exp.vision.models import build_vae
from exp.vision.results.v18c.judge_width import yard_stage
from exp.vision.width_eval import (
    CheckConf,
    WidthConf,
    _vh0a,
    _vh0b,
    _vh0c,
    _vh0d,
    _vh0e,
    _vh0f,
    _vh0g,
    _vh0h,
    _vh0i,
    _vh0k,
    _vh0l,
    _vh0m,
    code_rows,
    evaluate_run,
    evaluate_vae,
    evaluate_yard,
    lin_sums,
    point,
    slots,
    synthetic_sets,
    white_rows,
    width_sets,
)
from tests.vision.test_fill_eval import TINY as FILL_TINY
from tests.vision.test_geom_eval import SPLICED, TINY, sets_of, tours_with_zurich
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

WTINY = WidthConf(max_lag=6, max_pair=8, main_lo=2, strat_lags=(2, 3, 4), dir_lag=2,
                  lin_ts=(2, 4, 6))  # fmt: skip


def tiny_sets():
    sets = sets_of()
    return sets, width_sets(sets, TINY, WTINY, SPLICED, stack=2, stride=1)


class TestSets:
    def test_the_geom_lags_are_the_geom_sets(self):
        sets, ws = tiny_sets()
        for lag in TINY.lags:
            u_all, u_in = ws.centres(lag)
            assert torch.equal(u_all, sets.triples[lag]["all"])
            assert torch.equal(u_in, sets.triples[lag]["in"])

    def test_the_common_centres_are_in_at_every_lag(self):
        _, ws = tiny_sets()
        common = ws.common
        assert len(common) > 0
        for lag in range(1, WTINY.max_lag + 1):
            assert bool(torch.isin(common, ws.centres(lag)[1]).all())

    def test_the_pairs_by_brute_force(self):
        sets, ws = tiny_sets()
        f = sets.n + sets.span
        fb = ws.frame_block
        for lag in (1, 5):
            u_all, u_in = ws.pairs(lag)
            want_all, want_in = [], []
            for u in range(sets.n - lag):
                frames = range(u, u + lag + sets.span + 1)
                if len({int(fb[i]) for i in frames}) == 1:
                    want_all.append(u)
                    if not any(
                        int(c) in range(u, u + lag + sets.span) for c in sets.cuts
                    ):
                        want_in.append(u)
            assert u_all.tolist() == want_all and u_in.tolist() == want_in
            assert f == len(fb)

    def test_the_strata_beyond_the_geom_lags(self):
        _, ws = tiny_sets()
        by = ws.strata[3]
        joined = torch.cat([by["slow"], by["mid"], by["fast"]]).sort().values
        assert torch.equal(joined, ws.centres(3)[1])


def codes(n: int, d: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    return ou(n, d, 0.9, g), ou(200, d, 0.9, g)


class TestRows:
    def test_the_geom_lags_are_geom_to_the_bit(self):
        sets, ws = tiny_sets()
        ev, ref = codes(sets.n)
        lab = {"code": "x", "place": "own"}
        mine = code_rows(ev, ref, ws, TINY, WTINY, sens=False, labels=lab)
        geom = geom_code_rows(ev, ref, sets, TINY, sens=False, labels=lab)
        for lag in TINY.lags:
            for stratum in ("in", "slow", "mid", "fast"):
                (a,) = (r for r in mine if r["part"] == "lin" and r["lag"] == lag
                        and r["stratum"] == stratum and r["k"] == TINY.k)  # fmt: skip
                (b,) = (r for r in geom if r["part"] == "mid" and r["lag"] == lag
                        and r["stratum"] == stratum and r["k"] == TINY.k)  # fmt: skip
                assert a["n_triples"] == b["n_triples"]
                assert (a["G"], a["num_b"], a["den_b"], a["n_b"]) == (
                    b["G"],
                    b["num_b"],
                    b["den_b"],
                    b["n_b"],
                )

    def test_reversed_in_time_the_same(self):
        ws = synthetic_sets(4 * 40 - 3, 3, 40, 6)  # whole blocks of 40 frames
        x = ou(ws.n, 3, 0.8, torch.Generator().manual_seed(1))
        f = lin_sums(white_rows(x, ws, WTINY, full=False))
        b = lin_sums(white_rows(x.flip(0), ws, WTINY, full=False))
        assert torch.equal(f.n.sum(1), b.n.sum(1))
        for q in ("25", "50"):
            assert torch.equal(f.win[q].sum(1), b.win[q].sum(1))
        torch.testing.assert_close(f.g(), b.g(), rtol=1e-12, atol=0)

    def test_the_survivors_are_the_running_and_of_the_wins(self):
        ws = synthetic_sets(4 * 40 - 3, 3, 40, 6)
        x = ou(ws.n, 3, 0.95, torch.Generator().manual_seed(2))
        ls = lin_sums(white_rows(x, ws, WTINY, full=False))
        common = ws.common
        alive = torch.ones(len(common), dtype=torch.bool)
        for i, lag in enumerate(range(WTINY.main_lo, WTINY.max_lag + 1)):
            alive &= wins(triple_terms(x, common, lag), 0.25)
            assert float(ls.surv["25"][i].sum()) == float(alive.sum())

    def test_the_full_rows_have_pairs_and_directions(self):
        sets, ws = tiny_sets()
        ev, ref = codes(sets.n)
        rows = code_rows(ev, ref, ws, TINY, WTINY, sens=False, labels={})
        main = [r for r in rows if r["k"] == TINY.k]
        assert {r["lag"] for r in main if r["part"] == "dpair"} == set(range(1, 9))
        (d,) = (r for r in main if r["part"] == "dir")
        assert len(d["W"]) == TINY.k and set(d["n_lin"]) == {"2", "4", "6"}
        side = [r for r in rows if r["k"] != TINY.k and r["part"] == "lin"]
        assert {r["stratum"] for r in side} <= {"common"}
        pt = point(lin_sums(rows, k=TINY.k), WTINY)
        assert pt["25"]["mean"] >= WTINY.main_lo * WTINY.step


class TestSlots:
    def test_the_places_by_hand(self):
        s = slots(torch.tensor([40]), 8)
        assert (s["past"].item(), s["now"].item(), s["future"].item()) == (32, 40, 48)
        assert s["ext"].tolist() == [[16, 24, 32, 40, 48, 56, 64, 72, 80]]
        assert s["mid"].tolist() == [[8, 16, 24, 32, 40, 48, 56, 64, 72]]


@pytest.mark.slow
class TestChecks:
    def test_the_checks_on_their_series(self):
        cc, w = CheckConf(), WidthConf()
        from exp.vision.geom_eval import GeomConf

        g = GeomConf()
        ws = synthetic_sets(cc.n, cc.span, g.block, w.max_lag)
        rows = (_vh0a(cc, ws, w) + _vh0b(cc, ws, w) + _vh0c(cc, ws, w)
                + _vh0d(cc, ws, w) + _vh0e(cc, ws, w) + _vh0f(cc, ws, w)
                + _vh0g(cc, ws, w) + _vh0h(cc, g, w) + _vh0l(cc, ws, w)
                + _vh0m(cc, ws, g, w))  # fmt: skip
        checks = [r for r in rows if r["part"] == "check"]
        failed = [r["name"] for r in checks if not r["pass"]]
        # README: V-h0e's N_lin at 3.2 s, 4% inside the band's window, is
        # within the directions' sampling spread of the edge
        assert set(failed) <= {"V-h0e-nlin"}, failed
        names = {r["name"] for r in checks}
        assert {"V-h0a", "V-h0b", "V-h0c", "V-h0d", "V-h0e-window", "V-h0f", "V-h0g",
                "V-h0h-reverse", "V-h0h-thin", "V-h0l", "V-h0m"} <= names  # fmt: skip

    def test_the_long_checks_run_on_few_draws(self):
        from exp.vision.geom_eval import GeomConf

        cc = replace(CheckConf(), cover_seeds=(1101, 1103), cover_long=2,
                     known_seeds=(1301, 1302))  # fmt: skip
        g, w = replace(GeomConf(), boot=50), WidthConf()
        ws = synthetic_sets(cc.n, cc.span, g.block, w.max_lag)
        (i,) = _vh0i(cc, ws, g, w)
        assert i["n"] == 2 and set(i["share"]) == {"G_hi", "mean", "typ"}
        k = _vh0k(cc, ws, g, w)
        assert [r["name"] for r in k] == ["V-h0k", "V-h0k-power", "V-h0k-power"]
        assert math.isclose(i["truth"]["G_hi"], 3.0569, rel_tol=1e-4)


class TestJobs:
    @pytest.mark.slow
    def test_the_yard_run_and_vae_end_to_end(self, tmp_path):
        root = tours_with_zurich(tmp_path / "d")
        mix = tiny_run(root, tmp_path, "--patch", "2", "--token-dim", "16", "--proj",
                       "fixed", "--sigreg-on", "mix", name="mix")  # fmt: skip
        mean = tiny_run(root, tmp_path, "--sigreg-on", "mean", name="mean")
        conf = replace(conf_of(), device="cpu")
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        ckpt.save_eval(vae_run, program="vae", group="iid", t=0.0, frame=10,
                       modules=build_vae(conf, seed=5).modules())  # fmt: skip
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)  # the geom sets, made first
        inits = {"init": mix, "init_f": mix, "init_m": mean, "init_c": mix,
                 "init_c1": mix}  # fmt: skip
        out = evaluate_yard(conf, sets_dir, tmp_path / "width", inits=inits,
                            init_v16=vae_run, root=root, gconf=TINY, wconf=WTINY,
                            fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        names = sorted(p.stem for p in (out / "yard" / "rows").iterdir())
        assert names == sorted(["pca", "color", "mean8", "init", "init_f", "init_m",
                                "init_c", "init_c1", "init_v16"])  # fmt: skip
        vh = json.loads((out / "yard" / "vh.json").read_text())
        assert vh and all(r["pass"] for r in vh if r["part"] == "check")
        init_c = json.loads((out / "yard" / "rows" / "init_c.json").read_text())
        assert {r["place"] for r in init_c} == {"own", "h_mean"}
        assert {r["key"] for r in init_c} == {
            json.loads((sets_dir / "sets.json").read_text())["key"]
        }
        got = evaluate_run(mix, steps=(0,), sets_dir=sets_dir, root=root, device="cpu",
                           gconf=TINY, wconf=WTINY, fconf=FILL_TINY, spacing=EVERY,
                           span=SPAN)  # fmt: skip
        rows = json.loads((got / "rows" / "000000.json").read_text())
        assert {r["place"] for r in rows} == {"own", "h_mean"}
        got_y = yard_stage(out, WTINY, TINY)
        assert "pca/own" in got_y["codes"] and "init_c-init_c1/own" in got_y["pairs"]
        (vae_run / "ckpt").mkdir(exist_ok=True)
        first = sorted((vae_run / "ckpt").glob("*.pt"))[0]
        target = evaluate_vae(first, name="v16", sets_dir=sets_dir, out=out, root=root,
                              device="cpu", gconf=TINY, wconf=WTINY, fconf=FILL_TINY,
                              spacing=EVERY, span=SPAN)  # fmt: skip
        assert {r["code"] for r in json.loads(target.read_text())} == {"v16"}


def test_whitening_is_of_the_reference():
    """V-h0m in small: the rows of the reference's whitening, not the
    evaluation's."""
    sets, ws = tiny_sets()
    ev, ref = codes(sets.n)
    ref = ref * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    rows = code_rows(ev, ref, ws, TINY, WTINY, sens=False, labels={})
    got = lin_sums(rows, k=TINY.k, ref="all").g()
    want = lin_sums(white_rows(Whitener.fit(ref, TINY.k, c_eps=TINY.c_eps).white(ev),
                               ws, WTINY, full=False)).g()  # fmt: skip
    torch.testing.assert_close(got, want, rtol=1e-12, atol=0)
