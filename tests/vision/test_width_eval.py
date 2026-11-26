r"""``exp.vision.width_eval`` (the linear horizon, ``plans/jepa.md``,
「線形の幅（2026-11-26）」).

The oracles: the dense sets at the geom lags are the geom sets' ``all`` and
``in`` (the same rule spread over the lags), the common centres are ``in``
at every lag, the pairs by brute force; the rows' pooled $G$ and blocks'
sums of ``in`` at the geom lags are ``geom_eval``'s to the bit (the same
code, the same sets); the rows of a series reversed in time (sets of whole
blocks) have the same wins and $G$; the survivors' sums are the running
first runs of the wins; the shifted blocks' sums hold the same triples
(their totals) in each centre's own shifted block; the guard's conditions
and targets by hand; a band of no noise has the same triple terms at every
time (of no phase) and its pooled $G = s/2$; a rotation's period on the
grid is refused; the anchors' places by hand; the checks V-h0 on their
full series (the long V-h0i, V-h0k run on few draws only), V-h0e's
$N_{\mathrm{lin}}$ truth of hand-made directions (a $T$ whose closed margin
to 1/4 is within $z$ spreads left out); the checks run apart (any split of
the units and of the seeds) merged to the rows run at once, to the byte, and
a truth of V-h0k alone that of the one stream; and the yard's, the run's
and the VAE's jobs end to end on tiny runs (the yard of a few codes the
same files to the byte).
"""

import json
import math
from dataclasses import replace

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.lib.linwidth import survival
from exp.lib.midpoint import Whitener, block_sums, triple_terms, wins
from exp.lib.synth_series import ou
from exp.vision.compose import to_dict
from exp.vision.geom_eval import GeomConf, _sets_at, code_rows as geom_code_rows
from exp.vision.models import build_vae
from exp.vision.results.v18c.judge_width import yard_stage
from exp.vision.width_eval import (
    TARGETS,
    CheckConf,
    WidthConf,
    _band_columns_g,
    _band_noise,
    _band_truth,
    _gen,
    _rotation_truth,
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
    _vh0k_truth,
    _vh0l,
    _vh0m,
    check_parts,
    check_rows,
    check_units,
    code_rows,
    conditions,
    evaluate_checks,
    evaluate_run,
    evaluate_vae,
    evaluate_yard,
    lin_sums,
    merge_checks,
    merge_checks_into,
    nlin_truth,
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

    def test_the_survivors_are_the_first_runs_of_the_wins(self):
        ws = synthetic_sets(4 * 40 - 3, 3, 40, 6)
        x = ou(ws.n, 3, 0.95, torch.Generator().manual_seed(2))
        ls = lin_sums(white_rows(x, ws, WTINY, full=False))
        common = ws.common
        lags = range(WTINY.main_lo, WTINY.max_lag + 1)
        won = torch.stack([wins(triple_terms(x, common, lag), 0.25) for lag in lags], 1)
        ends = []  # by hand: the end of each moment's first run of wins
        for row in won.tolist():
            if True not in row:
                continue
            e = row.index(True)
            while e + 1 < len(row) and row[e + 1]:
                e += 1
            ends.append(e)
        for i in range(len(lags)):
            assert float(ls.surv["25"][i].sum()) == sum(e >= i for e in ends)
        assert torch.equal(survival(won).sum(0).double(), ls.surv["25"].sum(1))

    def test_the_shifted_blocks_hold_the_same_triples(self):
        ws = synthetic_sets(4 * 40 - 3, 3, 40, 6)
        # blocks of 40 from the frame 20, the first 20 frames joined to the first
        assert ws.blocks_shift[:60].tolist() == [0] * 60
        assert ws.blocks_shift[60:100].tolist() == [1] * 40
        x = ou(ws.n, 3, 0.9, torch.Generator().manual_seed(3))
        rows = white_rows(x, ws, WTINY, full=False)
        ls, sh = lin_sums(rows), lin_sums(rows, shift=True)
        torch.testing.assert_close(ls.num.sum(1), sh.num.sum(1), rtol=1e-12, atol=0)
        assert torch.equal(ls.n.sum(1), sh.n.sum(1))
        assert torch.equal(ls.win["25"].sum(1), sh.win["25"].sum(1))
        c = ws.common
        t = triple_terms(x, c, 3)
        want = block_sums(t.num, ws.blocks_shift[c], ws.n_blocks_shift)
        torch.testing.assert_close(sh.num[2], want, rtol=1e-12, atol=0)

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

    def test_the_conditions_and_their_targets_by_hand(self):
        # a code of the time and its square: ext at t + n, mid at t
        u = torch.arange(100, dtype=torch.float64)
        c = torch.stack([u, u * u], 1)
        s = slots(torch.tensor([40]), 8)
        got = conditions(c, s)
        assert got["ext"].tolist() == [[48.0, 2 * 1600.0 - 1024.0]]
        assert got["mid"].tolist() == [[40.0, (1024.0 + 2304.0) / 2]]
        assert (got["past"][0, 0], got["now"][0, 0], got["future"][0, 0]) == (
            32,
            40,
            48,
        )
        for read in ("ext", "mid"):
            assert s[read][0, 4] == s[TARGETS[read]][0]


class TestSynthetic:
    def test_a_band_of_no_noise_is_of_no_phase(self):
        ws = synthetic_sets(4 * 40 - 3, 3, 40, 6)
        x = _band_noise(ws.n, 8, 37.3, 0.0, torch.Generator().manual_seed(4))
        for lag in (1, 3, 6):
            t = triple_terms(x, ws.common, lag)
            rho = 1 - 2 * t.num / t.den
            torch.testing.assert_close(rho, rho[:1].expand_as(rho), rtol=0, atol=1e-10)
            s = 1 - math.cos(2 * math.pi * lag / 37.3)
            g = float(t.num.sum() / t.den.sum())
            assert g == pytest.approx(s / 2, rel=1e-10)

    def test_a_period_on_the_grid_is_refused(self):
        with pytest.raises(ValueError, match="on the grid"):
            _rotation_truth(torch.tensor([123.0], dtype=torch.float64), 246)
        cc = CheckConf()
        for period in (cc.rotation_period, *cc.chain_periods, cc.thin_period):
            for lag in range(1, 257):
                for half in (1 / 6, 1 / 4):
                    _rotation_truth(torch.tensor([period], dtype=torch.float64), lag,
                                    half)  # fmt: skip


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
        assert not failed, failed
        # the user's decision: 3.2 s, 4% inside the band's window, left out
        nlin = [r for r in checks if r["name"] == "V-h0e-nlin"]
        assert all(r["excluded"] == [32] for r in nlin), nlin
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
        assert i["n"] == 2 and set(i["share"]) == {"G_hi", "mean1", "mean", "typ"}
        k = _vh0k(cc, ws, g, w)
        assert [r["name"] for r in k] == ["V-h0k", "V-h0k-power", "V-h0k-power"]
        assert k[0]["true_ratio"] == 0.0
        assert all(math.isfinite(r["true_ratio"]) for r in k)
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
        # a few codes apart (fleet): their files the same to the byte
        few = evaluate_yard(conf, sets_dir, tmp_path / "few", inits=inits,
                            init_v16=vae_run, root=root, gconf=TINY, wconf=WTINY,
                            fconf=FILL_TINY, spacing=EVERY, span=SPAN,
                            codes=("init_c", "pca", "init_v16"))  # fmt: skip
        got_few = sorted(p.name for p in (few / "yard" / "rows").iterdir())
        assert got_few == ["init_c.json", "init_v16.json", "pca.json"]
        for name in ("rows/init_c.json", "rows/init_v16.json", "rows/pca.json",
                     "vh.json"):  # fmt: skip
            assert (few / "yard" / name).read_bytes() == (
                out / "yard" / name
            ).read_bytes()
        color = evaluate_yard(conf, sets_dir, tmp_path / "color", inits={},
                              init_v16=vae_run, root=root, gconf=TINY, wconf=WTINY,
                              fconf=FILL_TINY, spacing=EVERY, span=SPAN,
                              codes=("color",))  # fmt: skip
        assert not (color / "yard" / "vh.json").exists()
        assert (color / "yard" / "rows" / "color.json").read_bytes() == (
            out / "yard" / "rows" / "color.json"
        ).read_bytes()
        with pytest.raises(ValueError, match="no yardsticks"):
            evaluate_yard(conf, sets_dir, tmp_path / "x", inits=inits, init_v16=vae_run,
                          codes=("pcaa",))  # fmt: skip
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


class TestNlinTruth:
    def test_a_t_within_the_spread_of_the_edge_is_left_out(self):
        cc = CheckConf()
        w = WidthConf(main_lo=8, max_lag=64, lin_ts=(8, 16, 32, 64))
        lags = list(range(8, 65))
        closed = {t: _band_columns_g(cc, t) for t in lags}
        # 5 band directions about the closed G, of the spread 0.0158 (z =
        # 3.54 of 5: 32 s's margin 0.0122 is within, 16 s's 0.186 is not),
        # 2 white ones at 3/4
        off = [-0.02, -0.01, 0.0, 0.01, 0.02]
        band = [[closed[t] + o for t in lags] for o in off]
        white = [[0.75] * len(lags)] * 2
        got = nlin_truth(cc, w, {"g": white[:1] + band + white[1:]}, 5)
        assert got == {"inside": [8, 16], "beyond": [64], "excluded": [32]}
        assert closed[32] == pytest.approx(0.2378, abs=1e-4)
        # with no spread every T is in or beyond
        flat = [[closed[t] for t in lags]] * 5
        got = nlin_truth(cc, w, {"g": white + flat}, 5)
        assert got == {"inside": [8, 16, 32], "beyond": [64], "excluded": []}


# a small V-h0 (any of its pass or fail; the merge is of the rows alone)
CG = replace(GeomConf(), block=160, boot=40)
CW = WidthConf(max_lag=24, main_lo=4, strat_lags=(), dir_lag=4, lin_ts=(4, 8, 16, 24),
               max_pair=8)  # fmt: skip
CC = replace(CheckConf(), n=6 * 160 - 7, d=8, thin_n=6 * 160 - 7,
             cover_seeds=(1101, 1104), cover_long=2, known_seeds=(1301, 1303),
             white_dirs=(1, 2), white_dirs_half=(3,), leak_dims=4)  # fmt: skip


class TestChecksApart:
    def test_any_split_merges_to_the_rows_at_once(self):
        whole = json.dumps(check_rows(CC, CG, CW))
        units = check_units(CC)
        short = [u for u in units if u not in ("V-h0i", "V-h0k")]
        splits = [
            [check_parts(CC, CG, CW, units)],
            [check_parts(CC, CG, CW, short[:5]), check_parts(CC, CG, CW, short[5:]),
             check_parts(CC, CG, CW, ["V-h0i", "V-h0k"], (1101, 1102)),
             check_parts(CC, CG, CW, ["V-h0k", "V-h0i"], (1102, 1400))],
            [check_parts(CC, CG, CW, [u], (s, s + 1)) for u in units
             for s in (1101, 1102, 1103, 1301, 1302)
             if u in ("V-h0i", "V-h0k") or s == 1101],
        ]  # fmt: skip
        for parts in splits:
            rows = json.loads(json.dumps([r for p in reversed(parts) for r in p]))
            assert json.dumps(merge_checks(rows, CC)) == whole

    def test_a_missing_or_twice_given_part_is_refused(self):
        units = check_units(CC)
        rows = check_parts(CC, CG, CW, units, (1101, 1400))
        with pytest.raises(ValueError, match="missing"):
            merge_checks(rows[:-1], CC)
        with pytest.raises(ValueError, match="twice"):
            merge_checks(rows + rows[-1:], CC)
        with pytest.raises(ValueError, match="no check units"):
            check_parts(CC, CG, CW, ["V-h0z"])

    def test_the_files_apart_merge_to_the_byte(self, tmp_path):
        evaluate_checks(tmp_path / "once", CC, CG, CW)
        evaluate_checks(tmp_path / "p0", CC, CG, CW, seeds=(1101, 1103))
        evaluate_checks(tmp_path / "p1", CC, CG, CW, units=("V-h0i", "V-h0k"),
                        seeds=(1103, 1400))  # fmt: skip
        merge_checks_into(tmp_path / "merged", [tmp_path / "p1", tmp_path / "p0"],
                          CC, CG, CW)  # fmt: skip
        once = (tmp_path / "once" / "rows.json").read_bytes()
        assert (tmp_path / "merged" / "rows.json").read_bytes() == once
        with pytest.raises(ValueError, match="other settings"):
            merge_checks_into(tmp_path / "m2", [tmp_path / "p0"], replace(CC, seed=7),
                              CG, CW)  # fmt: skip

    def test_a_truth_of_v_h0k_alone_is_of_the_one_stream(self):
        ws = synthetic_sets(CC.n, CC.span, CG.block, CW.max_lag)
        g = _gen(CC.known_truth_seed)
        stream = [_band_truth(CC, ws, CW, CC.band_period * r, g)["25"]
                  for r in CC.known_rs]  # fmt: skip
        for i in range(len(CC.known_rs)):
            alone = _vh0k_truth(CC, ws, CW, i)
            assert alone["mean"] == stream[i]["mean"]
            assert alone["mean1"] == stream[i]["mean1"]
