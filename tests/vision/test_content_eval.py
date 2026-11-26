r"""``exp.vision.content_eval`` (the guard rebuilt on the observation's
content, ``plans/jepa.md``).

The oracles: the positive process's two codes give each other back
($\cos\theta = \cos5\theta\cos4\theta + \sin5\theta\sin4\theta$) and each
triple's win is the closed one of the planes' angles ($\rho = \sum_k (1 -
\cos\alpha_k)\cos\alpha_k / \sum_k (1 - \cos\alpha_k)$, $\alpha_k =
2\pi k \Delta / T$: the slow code's window ends at $T/6$, the fast one's
between $T/30$ and $T/24$); the negative process's cheats (the drop a
projection, the past mean by hand); the calibration's noise in closed form
and against the least squares of a long realization (the conditional mean is
linear there: $\psi = -\psi_*$; of ``calib_noise_d`` $\psi^d = -\psi_*$,
and of ``calib_noise`` the closed $\psi^d$ of ``change_explained``); a random orthogonal matrix is orthogonal;
the rows of a code by hand on the least squares: the observation itself
explains all ($E = E^d = 1$, the ceiling 0, the copy and the extrapolation
their errors by hand), a constant code nothing ($E = E^d = 0$); the checks
apart merge to the rows of one run, byte for byte; and the real data's job
end to end on a tiny run (one code alone the same file to the byte), read
by the judge; and its alignment: a code that is the stack's last frame
itself (pooled as the observation) is read back by the least squares with
$E = E^d = 1$ on the training set, the held-out runs and Zurich.

v3's oracles: the parts' shares solved in closed form are the plan's
(0.1466, 0.1034; 0.1608, 0.0892) and sum to 1 and, of the change, to $r$;
the calibrations' noises (0.49336, 0.0023702) and the difference's and the
pair's $E^d$ of the former ($-0.509$, $+0.0060$) are the plan's and, of a
long realization (the conditional mean linear, so the least squares reach
the best), the empirical shares, $r$ and the least squares' $E$, $\psi$,
$\psi^d$ and $E^d$ are the closed forms'; the positive codes give each
other back and share the scene, the cheats keep it; the starts overlap as
registered and the validation runs lie the gap away; a readout that
memorizes the runs (a scene in the code, its detail independent in the
observation) loses on the validation runs to the least squares, its detail's
$E$ above 0 in sample and below out of it, and the choice keeps the least
squares, not worse on the record's runs and not past the best; the
observation itself is read back by the least squares at $E = E^d = 1$ on
every set, from the pair and from the level's difference (on the real data's
job too, of the stack's last frame); the checks apart merge to the rows of
one run, byte for byte; the real data's job end to end on a tiny run (one
code alone the same file and manifest), read by the judge.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    content_sums,
    fit_linear,
    kind_of,
    pair_inputs,
    pool_obs,
    predict_linear,
)
from exp.lib.linwidth import windows
from exp.vision import content_eval
from exp.vision.compose import to_dict
from exp.vision.content_eval import (
    ContentConf,
    ContentConf3,
    RunSet,
    SelectConf,
    Synth3Conf,
    SynthConf,
    _noisy3,
    calib3,
    calib3_d,
    calib_noise,
    calib_noise_d,
    change_explained,
    change_parts3,
    cheats3,
    check_parts,
    check_parts3,
    check_units,
    check_units3,
    chosen3,
    columns3,
    content_rows,
    diff_ed3,
    draw_starts,
    evaluate_content,
    evaluate_content3,
    far_runs,
    front3,
    front_of,
    merge_checks,
    merge_checks3,
    model_source,
    negative,
    negative_cheats,
    obs_acf3,
    orthogonal,
    pair3,
    pair_ed3,
    parts3,
    positive,
    process3,
    select_rows,
    series_acf,
    starts3,
    sums3,
)
from exp.vision.geom_eval import GeomConf, _sets_at
from exp.vision.models import build_vae
from exp.vision.results.v18c.judge_content import (
    PSI_STAR,
    checks_stage,
    models3_stage,
    real3_stage,
    real_stage,
)
from exp.vision.width_eval import (
    WidthConf,
    _common_sums,
    counts_of,
    synthetic_sets,
    white_rows,
)
from exp.vision.width_guard import _past_mean
from tests.vision.test_eval import SIZE, write_tours
from tests.vision.test_fill_eval import TINY as FILL_TINY
from tests.vision.test_gen_eval import features
from tests.vision.test_geom_eval import SPLICED, TINY, tours_with_zurich
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY
from tests.vision.test_width_eval import WTINY, tiny_sets
from tests.vision.test_width_guard import GTINY

CPU = torch.device("cpu")
HOLD = 1  # the held-out runs of the tiny data
F64 = torch.float64
CG = replace(GeomConf(), block=160, boot=40)
CW = WidthConf(max_lag=24, main_lo=4, strat_lags=(), dir_lag=4, lin_ts=(4, 8, 16, 24),
               max_pair=8)  # fmt: skip
SC = replace(SynthConf(), n=6 * 160 - 7, period=61.3, train_n=600, windows=(2, 4),
             long_parts=2, seeds=(2101, 2103), cal_seeds=(2301, 2303),
             readout_seeds=(1, 2, 3), mlp=MLPConf(hidden=8, steps=30, batch=32))  # fmt: skip


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestProcesses:
    def test_the_fast_code_gives_the_slow_back(self):
        _, c = positive(500, SC, _gen())
        s, f = c["slow"] / math.sqrt(2), c["fast"] / math.sqrt(2)
        cos = f[:, 2] * f[:, 0] + f[:, 3] * f[:, 1]  # cos(5t - 4t)
        sin = f[:, 3] * f[:, 0] - f[:, 2] * f[:, 1]
        assert torch.allclose(cos, s[:, 0], atol=1e-12)
        assert torch.allclose(sin, s[:, 1], atol=1e-12)
        assert c["slow"].square().mean().item() == pytest.approx(1.0, abs=0.02)

    def test_each_triples_win_is_the_closed_one(self):
        ws = synthetic_sets(SC.n, SC.span, CG.block, CW.max_lag)
        _, c = positive(SC.n, SC, _gen(1))
        lags = torch.arange(CW.main_lo, CW.max_lag + 1)
        for name, ks in (("slow", (1,)), ("fast", SC.fast_ks)):
            ls = _common_sums(c[name], ws, CW)
            alpha = (
                2
                * math.pi
                * lags.double()[:, None]
                * torch.tensor(ks, dtype=F64)
                / SC.period
            )
            w = 1 - alpha.cos()
            rho = (w * alpha.cos()).sum(1) / w.sum(1)
            assert ls.p("25").tolist() == (rho > 0.5).double().tolist()
        x = CW.seconds(lags)
        hi = windows(
            x, _common_sums(c["slow"], ws, CW).p("25"), 0.5, below=False
        ).typical
        assert hi is not None and abs(hi - SC.period / 6 * CW.step) < 2 * CW.step
        fast = windows(
            CW.seconds(torch.arange(1, CW.max_lag + 1)),
            torch.tensor([float(_rho_fast(n) > 0.5) for n in range(1, CW.max_lag + 1)]),
            0.5, below=False,
        ).typical  # fmt: skip
        assert fast is not None
        assert (
            SC.period / 30 * CW.step - CW.step
            < fast
            < SC.period / 24 * CW.step + CW.step
        )

    def test_the_negative_cheats(self):
        y, m = negative(50, SC, _gen(2))
        assert y.shape == (50, 4) and m.shape == (50, 4)
        ch = negative_cheats(m, SC)
        assert torch.equal(ch["drop"][:, :2], m[:, :2]) and not ch["drop"][:, 2:].any()
        u = 10
        assert torch.allclose(ch["mean-4"][u], m[u - 4 : u + 1].mean(0), atol=1e-12)
        assert torch.equal(ch["mean-2"], _past_mean(m, 2))

    def test_the_calibrations_noise(self):
        s2 = calib_noise(SC)
        rho = 2**-SC.psi_star
        assert (1 + SC.f_dims / (1 + s2)) / (1 + SC.f_dims) == pytest.approx(rho)
        # the least squares of a long realization (the conditional mean is linear)
        g = _gen(3)
        y, m = negative(200000, SC, g)
        noisy = m.clone()
        noisy[:, 2:] += math.sqrt(s2) * torch.randn(200000, 2, generator=g, dtype=F64)
        t = torch.arange(1, 200000)
        blocks = torch.zeros(len(t), dtype=torch.long)
        es = []
        for c in (m, noisy):
            b = fit_linear(c, y)
            es.append(
                content_sums(y, [predict_linear(b, c)], y.mean(0), t, blocks, 1).value()
            )
        assert math.log2(es[1] / es[0]) == pytest.approx(-SC.psi_star, abs=0.01)

    def test_the_change_calibrations_noise(self):
        s2, s2d = calib_noise(SC), calib_noise_d(SC)
        e0 = change_explained(SC, 0.0)
        assert math.log2(change_explained(SC, s2d) / e0) == pytest.approx(-SC.psi_star)
        assert 0 < s2d < s2
        # the least squares of a long realization: E^d of the level readout
        g = _gen(7)
        n = 400000
        y, m = negative(n, SC, g)
        t = torch.arange(1, n)
        blocks = torch.zeros(len(t), dtype=torch.long)
        eds = []
        for v in (0.0, s2d, s2):
            c = m.clone()
            c[:, 2:] += math.sqrt(v) * torch.randn(n, 2, generator=g, dtype=F64)
            b = fit_linear(c, y)
            eds.append(
                content_sums(y, [predict_linear(b, c).double()], y.mean(0), t, blocks, 1)
                .value(True)
            )  # fmt: skip
        assert eds[0] == pytest.approx(e0, abs=0.01)
        assert math.log2(eds[1] / eds[0]) == pytest.approx(-SC.psi_star, abs=0.02)
        want = math.log2(change_explained(SC, s2) / e0)
        assert math.log2(eds[2] / eds[0]) == pytest.approx(want, abs=0.03)

    def test_orthogonal(self):
        q = orthogonal(5, 3)
        assert torch.allclose(q.T @ q, torch.eye(5, dtype=F64), atol=1e-12)
        assert torch.equal(q, orthogonal(5, 3))


def _rho_fast(n: int) -> float:
    a = [2 * math.pi * k * n / SC.period for k in SC.fast_ks]
    w = [1 - math.cos(x) for x in a]
    return sum(wi * math.cos(x) for wi, x in zip(w, a, strict=True)) / sum(w)


class TestRows:
    def test_the_observation_itself_and_a_constant(self):
        _, ws = tiny_sets()
        g = _gen(4)
        y = torch.randn(ws.n, 3, generator=g)
        y_tr = torch.randn(200, 3, generator=g)
        conf = ContentConf(seeds=(1,), mlp=MLPConf(hidden=4, steps=2, batch=4),
                           ext_lags=(1, 2))  # fmt: skip
        rows = content_rows({"code": "y"}, y_tr, y_tr, y, y, ws, conf, CPU)
        lin = next(r for r in rows if r["part"] == "content" and r["readout"] == "lin")
        assert sum(lin["err_b"][0]) == pytest.approx(0, abs=1e-9)
        assert sum(lin["derr_b"][0]) == pytest.approx(0, abs=1e-9)
        ext = next(r for r in rows if r["part"] == "obs_ext" and r["readout"] == "lin")
        t = ws.common
        yd = y.double()
        for lag in (1, 2):
            e = ext["lags"][0][str(lag)]
            assert e["ceil"] == pytest.approx(0, abs=1e-8)
            assert e["copy"] == pytest.approx(
                float((yd[t + lag] - yd[t]).square().sum()), rel=1e-6
            )
            want = float((yd[t + lag] - 2 * yd[t] + yd[t - lag]).square().sum())
            assert e["ext"] == pytest.approx(want, rel=1e-6)
        const = content_rows({"code": "0"}, torch.zeros(200, 2), y_tr, torch.zeros(ws.n, 2),
                             y, ws, conf, CPU, ext=False)  # fmt: skip
        lin0 = next(r for r in const if r["readout"] == "lin")
        d = sum(lin0["dtot_b"])
        assert sum(lin0["derr_b"][0]) == pytest.approx(d, rel=1e-6)  # E^d = 0
        mean = y_tr.double().mean(0)
        tot = float((yd[t] - mean).square().sum())
        assert sum(lin0["err_b"][0]) == pytest.approx(tot, rel=1e-6)  # E = 0
        assert {r["part"] for r in const} == {"content"}

    def test_a_code_against_itself(self):
        _, ws = tiny_sets()
        g = _gen(5)
        y = torch.randn(ws.n, 3, generator=g)
        x = y + 0.1 * torch.randn(ws.n, 3, generator=g)
        conf = ContentConf(seeds=(1, 2), mlp=MLPConf(hidden=4, steps=2, batch=4))
        rows = content_rows({}, x, y, x, y, ws, conf, CPU, ext=False)
        from exp.lib.fidelity import ContentSums

        s = ContentSums.of_row(next(r for r in rows if r["readout"] == "lin"))
        nb = ws.n_blocks
        f = front_of((1.0, 2.0), s, s, counts_of(nb, TINY), counts_of(nb // 2, TINY),
                     (0.0, 0.0), 0.5, 0.1)  # fmt: skip
        assert f["psi"]["point"] == 0.0 and f["psi_d"]["point"] == 0.0


class TestChecksApart:
    def test_any_split_merges_to_the_rows_at_once(self):
        whole = check_parts(SC, CG, CW, check_units())
        merged = json.dumps(merge_checks(whole, SC, CG))
        split = (check_parts(SC, CG, CW, ["V-h4a2-long", "V-h4a1-long"])
                 + check_parts(SC, CG, CW, ["V-h4a3", "V-h4a1"], (2102, 2302))
                 + check_parts(SC, CG, CW, ["V-h4a2"], (2101, 2103))
                 + check_parts(SC, CG, CW, ["V-h4a1", "V-h4a3"], (2101, 2102))
                 + check_parts(SC, CG, CW, ["V-h4a3"], (2302, 2303)))  # fmt: skip
        assert json.dumps(merge_checks(list(reversed(split)), SC, CG)) == merged
        names = {r["name"] for r in json.loads(merged)}
        assert names == {"V-h4a-sigma", "V-h4a1", "V-h4a1-share", "V-h4a2", "V-h4a2-count",
                         "V-h4a3", "V-h4a3-power", "V-h4a3-d"}  # fmt: skip
        with pytest.raises(ValueError, match="missing"):
            merge_checks(split[1:], SC, CG)
        with pytest.raises(ValueError, match="twice"):
            merge_checks(split + split[:1], SC, CG)
        with pytest.raises(ValueError, match="no check units"):
            check_parts(SC, CG, CW, ["V-h4z"])
        got = checks_stage(json.loads(merged))
        assert set(got["checks"]) == {"V-h4a1", "V-h4a2", "V-h4a2-count", "V-h4a3"}


class TestReal:
    @pytest.mark.slow
    def test_end_to_end_and_the_judge(self, tmp_path):
        root = tours_with_zurich(tmp_path / "d")
        conf = replace(conf_of(), device="cpu")
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        cconf = ContentConf(obs=2, seeds=(1, 2, 3), mlp=MLPConf(hidden=8, steps=20, batch=8),
                            ext_lags=(1, 2), holdout_count=HOLD)  # fmt: skip
        kw: dict[str, Any] = dict(root=root, guard=GTINY, cconf=cconf, gconf=TINY, wconf=WTINY,
                                  fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        width = tmp_path / "width"
        codes = ("base", "rot", "mean-1", "mean-2", "drop-1", "color", "mean8")
        out = evaluate_content(conf, sets_dir, width / "content", codes=codes, local=True,
                               local_features=features, **kw)  # fmt: skip
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == sorted(codes)
        rows = json.loads((out / "base" / "content.json").read_text())
        got = {(r["part"], r["obs"], r["readout"], r.get("set")) for r in rows}
        assert got == {("content", "pix", "mlp", None), ("content", "pix", "lin", None),
                       ("obs_ext", "pix", "mlp", None), ("obs_ext", "pix", "lin", None),
                       ("content", "local", "mlp", None), ("content", "local", "lin", None),
                       ("fit", "pix", "mlp", "train"), ("fit", "pix", "lin", "train"),
                       ("fit", "pix", "mlp", "holdout"), ("fit", "pix", "lin", "holdout")}  # fmt: skip
        alone = evaluate_content(conf, sets_dir, tmp_path / "alone", codes=("mean-2",),
                                 local=True, local_features=features, **kw)  # fmt: skip
        for name in ("mean-2/content.json",):
            assert (alone / name).read_bytes() == (out / name).read_bytes()
        with pytest.raises(ValueError, match="no codes"):
            evaluate_content(conf, sets_dir, tmp_path / "x", codes=("mean-3",), **kw)
        with pytest.raises(ValueError, match="need their runs"):
            evaluate_content(conf, sets_dir, tmp_path / "x", codes=("init",), **kw)
        # the cheats' widths (any rows of the tiny sets) and the judge
        _, ws = tiny_sets()
        g = _gen(6)
        base = torch.randn(ws.n, 2, generator=g, dtype=F64)
        for name, x in (("base", base), ("mean-1", _past_mean(base, 1)),
                        ("mean-2", _past_mean(base, 2))):  # fmt: skip
            path = width / "guard" / "cheat" / name / "width.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(white_rows(x, ws, WTINY, full=False)))
        res = real_stage(width, WTINY, TINY)
        assert set(res["b1"]["fronts"]) == {"mean-1", "mean-2"}
        assert res["b2"]["codes"] == ["drop-1"]
        assert set(res["b3"]["comparable"]) >= {"base", "color", "mean8"}
        assert res["b3"]["comparable"]["init"] is None and not res["b3"]["pass"]
        assert len(res["sigma_seed"]) == 2
        assert set(res["pair_sigma"]) == set(codes) - {"base"}
        assert len(res["codes"]["base"]["rho1"]["pix"]) == 3

    @pytest.mark.slow
    def test_the_frame_itself_is_read_back_aligned(self, tmp_path, monkeypatch):
        # the code: the stack's last frame pooled as the observation, so the
        # least squares reads it back exactly where the code and the
        # observation are of the same stack (else E, E^d < 1)
        root = tours_with_zurich(tmp_path / "d")
        conf = replace(conf_of(), device="cpu")
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        obs = 2

        class Frame:
            def raw(self, x: torch.Tensor) -> torch.Tensor:
                return pool_obs(x[:, -1].cpu(), obs).double()

            def white(self, raw: torch.Tensor) -> torch.Tensor:
                return raw.double()

            def mu(self, x: torch.Tensor) -> torch.Tensor:
                return self.raw(x).float()

        monkeypatch.setattr(content_eval, "_yard_white", lambda *a, **k: Frame())
        # enough training stacks that the least squares of 12 pixels is determined
        guard = replace(GTINY, cheat_count=4, cheat_length=16)
        cconf = ContentConf(obs=obs, seeds=(1,), mlp=MLPConf(hidden=4, steps=2, batch=4),
                            ext_lags=(1,), holdout_count=HOLD)  # fmt: skip
        out = evaluate_content(conf, sets_dir, tmp_path / "w", codes=("color",), root=root,
                               guard=guard, cconf=cconf, gconf=TINY, wconf=WTINY,
                               fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        rows = json.loads((out / "color" / "content.json").read_text())
        lin = {r.get("set", "zurich"): ContentSums.of_row(r)
               for r in rows if r["readout"] == "lin" and r["part"] in ("content", "fit")}  # fmt: skip

        assert set(lin) == {"zurich", "train", "holdout"}
        for s in lin.values():
            assert s.value() == pytest.approx(1, abs=1e-6)
            assert s.value(True) == pytest.approx(1, abs=1e-6)


# -- v3 -------------------------------------------------------------------------------------

SEL = SelectConf(widths=(4, 8), every=10, mlp=MLPConf(hidden=8, steps=30, batch=32),
                 seeds=(1, 2, 3), v2_width=8, se_draws=50)  # fmt: skip
SC3 = replace(Synth3Conf(), n=6 * 160 - 7, period=61.3, total=6000, run=64, train_count=16,
              record_count=4, val_count=6, gap=32, windows=(2, 4), ell=20.0, f_tau=6.0,
              long_parts=2, seeds=(2101, 2103), cal_seeds=(2301, 2303), select=SEL)  # fmt: skip
LONG = replace(Synth3Conf(), ell=20.0)  # ergodic within a test's series


class TestParts3:
    def test_the_registered_shares(self):
        sc = Synth3Conf()
        neg, pos = parts3(sc, False), parts3(sc, True)
        assert neg["g"][0] == pytest.approx(0.1466, abs=5e-5)
        assert neg["n"][0] == pytest.approx(0.1034, abs=5e-5)
        assert pos["g"][0] == pytest.approx(0.1608, abs=5e-5)
        assert pos["n"][0] == pytest.approx(0.0892, abs=5e-5)
        for p in (neg, pos):
            assert sum(s for s, _ in p.values()) == pytest.approx(1.0)
            assert sum(s * f for s, f in p.values()) == pytest.approx(sc.r)
        # the change the codes hold: about 0.088 (negative) and 0.016 (positive)
        assert pair_ed3(sc, 0.0) == pytest.approx(0.088, abs=5e-4)
        c = change_parts3(sc, True)
        held = (c["k1"] + c["k4"] + c["k5"] + c["s"]) / sum(c.values())
        assert held == pytest.approx(0.016, abs=5e-4)

    def test_the_registered_calibrations(self):
        sc = Synth3Conf()
        s2, s2d = calib3(sc), calib3_d(sc)
        assert s2 == pytest.approx(0.49336, abs=5e-5)
        assert s2d == pytest.approx(0.0023702, abs=5e-7)
        assert diff_ed3(sc, s2) == pytest.approx(-0.509, abs=5e-4)
        assert pair_ed3(sc, s2) == pytest.approx(0.0060, abs=5e-5)
        assert math.log2(pair_ed3(sc, s2d) / pair_ed3(sc, 0.0)) == pytest.approx(-0.1)
        # the level's: kept - f sigma2 / (1 + sigma2) is 2^-0.1 of kept
        f = parts3(sc, False)["f"][0]
        assert (sc.kept - f * s2 / (1 + s2)) / sc.kept == pytest.approx(2**-0.1)


class TestAcf:
    def test_the_pooled_autocorrelation_by_hand(self):
        # two series of 2 columns; pairs only within a series; the mean of all
        g = _gen(3)
        ys = [torch.randn(7, 2, generator=g, dtype=F64),
              torch.randn(4, 2, generator=g, dtype=F64)]  # fmt: skip
        got = series_acf(ys, [0, 2, 5])
        m = torch.cat(ys).mean(0)
        var = (torch.cat(ys) - m).square().sum(1).mean()
        for i, lag in enumerate((0, 2, 5)):
            pr = [((y[: len(y) - lag] - m) * (y[lag:] - m)).sum(1)
                  for y in ys if lag < len(y)]  # fmt: skip
            want = torch.cat(pr).mean() / var
            assert got["rho"][i] == pytest.approx(float(want), rel=1e-12)
        assert got["rho"][0] == pytest.approx(1.0) and got["pairs"] == [11, 7, 2]

    @pytest.mark.slow
    def test_the_synthetic_observation_against_its_closed_form(self):
        # a long series (ell 2 s, so the scene averages out) of each sign
        lags = [1, 10, 40]
        for pos in (False, True):
            y, _ = process3(300000, LONG, pos, _gen(4))
            got = series_acf([y], lags)["rho"]
            want = obs_acf3(LONG, pos, lags)
            assert got == pytest.approx(want, abs=0.03)
        assert obs_acf3(LONG, False, [0]) == pytest.approx([1.0])
        # a step's change is 2 (1 - rho(1)) of the level's: r
        for pos in (False, True):
            assert obs_acf3(LONG, pos, [1])[0] == pytest.approx(1 - LONG.r / 2)


def _lstsq3(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    """$E$, the level's difference's $E^d$ and the pair's $E^d$ of the least
    squares in sample."""
    n = len(y)
    t = torch.arange(1, n)
    blocks = torch.zeros(n - 1, dtype=torch.long)
    lv = predict_linear(fit_linear(x, y), x).double()
    xp = pair_inputs(x[None])[0]
    d = torch.cat([torch.zeros_like(y[:1]), y[1:] - y[:-1]])
    cp = predict_linear(fit_linear(xp[1:], d[1:]), xp).double()
    a = content_sums(y, [lv], y.mean(0), t, blocks, 1)
    b = content_sums(y, [lv], y.mean(0), t, blocks, 1, dpreds=[cp])
    return a.value(), a.value(True), b.value(True)


class TestProcess3:
    @pytest.mark.slow
    def test_a_long_series_against_the_closed_forms(self):
        g = _gen(1)
        n = 300000
        y, c = process3(n, LONG, False, g)
        cols = columns3(LONG, False)
        p = parts3(LONG, False)
        v = y.var(0)
        eps = p["eps"][0] / y.shape[1]  # each column's white noise
        for name, key in (("s", "s"), ("n", "n"), ("g", "g")):
            want = p[key][0] + eps * len(cols[name])
            assert float(v[cols[name]].sum()) == pytest.approx(want, rel=0.06)
        d = y[1:] - y[:-1]
        r = float(d.square().sum(1).mean() / (y - y.mean(0)).square().sum(1).mean())
        assert r == pytest.approx(LONG.r, abs=0.002)
        m = c["mixed"]
        assert m.shape[1] == 2 + LONG.f_dims + LONG.scene
        assert torch.allclose(m.var(0), torch.ones(m.shape[1], dtype=F64), atol=0.05)
        # the scene's columns are the code's: y_s = sqrt(share / 16) s
        s_code = m[:, 2 + LONG.f_dims :]
        s_obs = y[:, cols["s"]]
        resid = s_obs - math.sqrt(p["s"][0] / LONG.scene) * s_code
        assert float(resid.var(0).max()) == pytest.approx(eps, rel=0.05)
        # the least squares reach the closed forms (the conditional mean is
        # linear); of the same observation, so the ratios are precise
        s2, s2d = calib3(LONG), calib3_d(LONG)
        e0, _, p0 = _lstsq3(m, y)
        e1, d1, p1 = _lstsq3(_noisy3(m, s2, LONG, g), y)
        _, _, p2 = _lstsq3(_noisy3(m, s2d, LONG, g), y)
        assert e0 == pytest.approx(LONG.kept, abs=0.005)
        assert p0 == pytest.approx(pair_ed3(LONG, 0.0), abs=0.002)
        assert math.log2(e1 / e0) == pytest.approx(-LONG.psi_star, abs=0.01)
        assert math.log2(p2 / p0) == pytest.approx(-LONG.psi_star, abs=0.02)
        assert d1 == pytest.approx(diff_ed3(LONG, s2), abs=0.03)
        assert p1 == pytest.approx(pair_ed3(LONG, s2), abs=0.002)

    def test_the_positive_codes_and_the_cheats(self):
        y, c = process3(500, SC3, True, _gen(2))
        assert y.shape == (500, 6 + 48)
        assert c["slow"].shape == (500, 2 + 16) and c["fast"].shape == (500, 4 + 16)
        assert torch.equal(c["slow"][:, 2:], c["fast"][:, 4:])  # the same scene
        f = c["fast"] / math.sqrt(2)
        cos = f[:, 2] * f[:, 0] + f[:, 3] * f[:, 1]  # cos(5t - 4t)
        assert torch.allclose(cos, c["slow"][:, 0] / math.sqrt(2), atol=1e-12)
        _, cn = process3(300, SC3, False, _gen(3))
        m = cn["mixed"]
        ch = cheats3(SC3)
        drop = ch["drop"](m)
        assert not drop[:, 2:4].any() and torch.equal(drop[:, :2], m[:, :2])
        assert torch.equal(drop[:, 4:], m[:, 4:])  # the scene is kept
        runs = torch.stack([m[:64], m[64:128]])
        assert torch.equal(ch["mean-4"](runs)[1], _past_mean(m[64:128], 4))


class TestDraws3:
    def test_draw_starts(self):
        tr = draw_starts(1000, 50, 30, 1)
        assert len(tr) == 30 and all(0 <= k <= 950 for k in tr)
        assert tr == draw_starts(1000, 50, 30, 1)
        assert any(
            abs(a - b) < 50 for a in tr for b in tr if a != b
        )  # overlaps allowed
        rec = draw_starts(1000, 50, 3, 2, tr[:5], tries=10000)
        assert all(abs(k - j) >= 50 for k in rec for j in tr[:5])
        assert all(abs(a - b) >= 50 for a in rec for b in rec if a != b)
        with pytest.raises(ValueError, match="runs in"):
            draw_starts(100, 50, 5, 1, [], tries=200)

    def test_the_registered_runs(self):
        sc = Synth3Conf()
        st = starts3(sc)
        assert [len(st[k]) for k in ("train", "record", "val")] == [512, 64, 128]
        for v in st["val"]:
            assert all(
                abs(v - j) >= sc.run + sc.gap for j in st["train"] + st["record"]
            )
            assert v + sc.run <= sc.total
        assert all(abs(a - b) >= sc.run for a in st["val"] for b in st["val"] if a != b)
        assert all(abs(a - j) >= sc.run for a in st["record"] for j in st["train"])

    def test_the_far_runs_and_their_sums_by_hand(self):
        # of the starts 0, 10, 25, 41 those at least 8 from 15 and 48
        assert far_runs([0, 10, 25, 41], [15, 48], 8) == [0, 2]
        s = ContentSums(torch.tensor([[1.0, 2, 3], [4, 5, 6]], dtype=F64),
                        torch.tensor([7.0, 8, 9], dtype=F64),
                        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=F64),
                        torch.tensor([1.0, 2, 3], dtype=F64))  # fmt: skip
        t = s.take([2, 0])
        assert t.err.tolist() == [[3, 1], [6, 4]] and t.tot.tolist() == [9, 7]
        assert t.value() == pytest.approx(1 - (4.5 + 2.5) / 16)
        assert t.value(True) == pytest.approx(1 - (0.45 + 0.25) / 4)


def _memorize_sets(seed: int, n_runs: int, length: int = 32
                   ) -> tuple[RunSet, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Runs whose code is $(z_t, s_r)$ and observation $(z_t A, n_r)$:
    $z$ white, the scene $s_r$ and the detail $n_r$ constant in a run and
    independent. The best readout is $(z A, 0)$ (``A`` the identity on 2
    columns, $n$ of variance 0.25 in 4)."""
    g = _gen(seed)
    z = torch.randn(n_runs, length, 2, generator=g, dtype=F64)
    s = torch.randn(n_runs, 1, 8, generator=g, dtype=F64).expand(-1, length, -1)
    dn = 0.5 * torch.randn(n_runs, 1, 4, generator=g, dtype=F64).expand(-1, length, -1)
    return RunSet(torch.cat([z, s], -1), torch.cat([z, dn], -1)), z, dn


class TestSelect:
    def test_the_choice_does_not_keep_a_readout_that_memorized_the_runs(self):
        # v2's readout learns n_r from s_r on the training runs and errs on
        # others (V-h4b's failure, V-h5a0's mechanism); the choice by the
        # validation runs keeps the least squares, near E* on the record's
        train, _, _ = _memorize_sets(1, 48)
        val, _, _ = _memorize_sets(2, 32)
        rec, _, dn = _memorize_sets(3, 32)
        ws = synthetic_sets(6 * 160 - 7, 7, 160, 24)
        ev, _, _ = _memorize_sets(4, 31, 32)
        x_ev, y_ev = ev.flat()
        sc = SelectConf(alphas=(0.0, 0.1), widths=(64,), every=1000,
                        mlp=MLPConf(hidden=64, steps=3000, batch=256, lr=3e-3), seeds=(1, 2),
                        v2_width=64, se_draws=50)  # fmt: skip
        rows = select_rows({}, train, val, rec, [(x_ev[: ws.n], y_ev[: ws.n])], ws, sc, CPU,
                           components={"n": [2, 3, 4, 5]})  # fmt: skip
        v2_val, lin_val = (
            sums3(rows, "v2", "val").value(),
            sums3(rows, "lin", "val").value(),
        )
        assert v2_val < lin_val
        # V-h5a0 (ii): on n's columns (best 0) v2's MLP fits the training
        # runs better than the least squares and errs more off them
        n_e = {(ro, st): sums3(rows, ro, st, "n").value()
               for ro in ("v2", "lin") for st in ("train", "val")}  # fmt: skip
        assert n_e["v2", "train"] > n_e["lin", "train"]
        assert n_e["v2", "val"] < n_e["lin", "val"]
        chosen = chosen3(rows)
        assert kind_of(chosen["level"]) == "lin"
        sel_rec = sums3(rows, "sel", "record").value()
        assert sel_rec >= sums3(rows, "lin", "record").value() - 0.02
        # nothing beats the best readout (z A, 0) on the same record runs
        # (E's anchors past each run's first) beyond the runs' chance
        y_rec = rec.y[:, 1:].flatten(0, 1)
        mean = train.y.flatten(0, 1).mean(0)
        best = 1 - float(dn[:, 1:].square().sum() / (y_rec - mean).square().sum())
        assert sel_rec <= best + 0.02
        lv = next(r for r in rows if r["part"] == "select" and r["target"] == "level")
        assert lv["selected"] == min(lv["scores"], key=lambda k: lv["scores"][k])
        assert kind_of(lv["runner"]) == "mlp" and lv["gap"] > 0 and lv["gap_se"] > 0
        ch = next(r for r in rows if r["part"] == "select" and r["target"] == "change")
        assert "diff" in ch["scores"] and len(ch["scores"]) == 2 + 3 + 1

    def test_the_observation_itself_is_read_back(self):
        # the code is the observation: the least squares give E = E^d = 1 on
        # every set, by the pair's readout and by the level's difference
        g = _gen(5)
        ws = synthetic_sets(6 * 160 - 7, 7, 160, 24)

        def runs(n: int) -> RunSet:
            y = torch.randn(n, 16, 3, generator=g, dtype=F64).cumsum(1)
            return RunSet(y, y)

        y_ev = torch.randn(ws.n, 3, generator=g, dtype=F64).cumsum(0) / 10
        rows = select_rows({"code": "y"}, runs(8), runs(4), runs(4), [(y_ev, y_ev)], ws,
                           SEL, CPU)  # fmt: skip
        for st in ("train", "val", "record", "eval"):
            for ro in ("lin", "lin-diff"):
                s = sums3(rows, ro, st)
                assert s.value() == pytest.approx(1, abs=1e-6)
                assert s.value(True) == pytest.approx(1, abs=1e-6)
        assert {r["readout"] for r in rows if r["part"] == "sums"} == {
            "sel", "lin", "lin-diff", "v2"}  # fmt: skip
        with pytest.raises(KeyError):
            sums3(rows, "sel", "zurich")


class TestChecks3Apart:
    def test_any_split_merges_to_the_rows_at_once(self):
        whole = check_parts3(SC3, CG, CW, check_units3())
        merged = json.dumps(merge_checks3(whole, SC3, CG))
        split = (check_parts3(SC3, CG, CW, ["V-h5a2-long", "V-h5a1-long"])
                 + check_parts3(SC3, CG, CW, ["V-h5a3", "V-h5a1"], (2102, 2302))
                 + check_parts3(SC3, CG, CW, ["V-h5a2"], (2101, 2103))
                 + check_parts3(SC3, CG, CW, ["V-h5a1", "V-h5a3"], (2101, 2102))
                 + check_parts3(SC3, CG, CW, ["V-h5a3"], (2302, 2303)))  # fmt: skip
        assert json.dumps(merge_checks3(list(reversed(split)), SC3, CG)) == merged
        names = {r["name"] for r in json.loads(merged)}
        assert names == {"V-h5a-sigma", "V-h5a0", "V-h5a0-d", "V-h5a4", "V-h5a4-far", "V-h5a1",
                         "V-h5a1-share", "V-h5a2", "V-h5a2-count", "V-h5a3", "V-h5a3-d",
                         "V-h5a3-power", "V-h5a3-chosen", "V-h5a-long-chosen"}  # fmt: skip
        with pytest.raises(ValueError, match="missing"):
            merge_checks3(split[1:], SC3, CG)
        with pytest.raises(ValueError, match="twice"):
            merge_checks3(split + split[:1], SC3, CG)
        with pytest.raises(ValueError, match="no check units"):
            check_parts3(SC3, CG, CW, ["V-h5z"])
        got = checks_stage(json.loads(merged), "V-h5a")
        assert set(got["checks"]) == {"V-h5a0", "V-h5a0-d", "V-h5a4", "V-h5a1", "V-h5a2",
                                      "V-h5a2-count", "V-h5a3", "V-h5a3-d"}  # fmt: skip
        long2 = next(r for r in whole if r.get("unit") == "V-h5a2-long")["value"]
        assert set(long2["content"]) == {"mixed", "drop", "mean-2", "mean-4", "mixed-rot",
                                         "noisy", "noisy-d"}  # fmt: skip
        draw = next(r for r in whole if r.get("unit") == "V-h5a2")["value"]
        assert {r["part"] for r in draw["content"]["mixed"]} == {"select", "sums"}

    def test_the_pairs_contrast(self):
        _, ws = tiny_sets()
        g = _gen(8)
        y = torch.randn(ws.n, 3, generator=g, dtype=F64)
        nb = ws.n_blocks
        t, blocks = ws.common, ws.blocks[ws.common - ws.max_lag]
        s = content_sums(y, [0.5 * y], y.mean(0), t, blocks, nb)
        c = (counts_of(nb, TINY), counts_of(nb // 2, TINY))
        p = pair3(s, s, *c, (0.01, 0.02), 0.1)
        rho = 2**-0.1
        assert p["D"]["point"] == pytest.approx((1 - rho) * s.value())
        assert p["sigma"] == [0.01, 0.02] and p["psi"][0] == 0.0
        f = front3((1.0, 2.0), s, s, *c, (0.0, 0.0), 0.5, 0.1)
        assert f["verdict"] in ("kept", "unresolved", "incomparable")
        assert front3((0.1, 2.0), s, s, *c, (0.0, 0.0), 0.5, 0.1)["verdict"] == "width"


C3 = ContentConf3(obs=2, select=SEL, ext_lags=(1, 2), record_count=HOLD, val_count=1,
                  val_gap=2)  # fmt: skip


class TestReal3:
    @pytest.mark.slow
    def test_end_to_end_and_the_judge(self, tmp_path):
        root = tours_with_zurich(tmp_path / "d")
        conf = replace(conf_of(), device="cpu")
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        kw: dict[str, Any] = dict(root=root, guard=GTINY, cconf=C3, gconf=TINY, wconf=WTINY,
                                  fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        width = tmp_path / "width"
        codes = ("base", "rot", "mean-1", "mean-2", "drop-1", "color", "mean8")
        out = evaluate_content3(conf, sets_dir, width / "content3", codes=codes, **kw)
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == sorted(codes)
        rows = json.loads((out / "base" / "content3.json").read_text())
        got = {(r["part"], r.get("readout"), r.get("set")) for r in rows}
        want = {("sums", ro, st) for ro in ("sel", "lin", "lin-diff", "v2")
                for st in ("train", "val", "record", "eval")}  # fmt: skip
        assert got == want | {("select", None, None), ("obs_ext", "sel", None)}
        man = json.loads((out / "intervals.json").read_text())
        starts = {r["set"]: r["starts"] for r in man}
        length = GTINY.cheat_length
        assert len(starts["val"]) == 1 and len(starts["record"]) == HOLD
        for v in starts["val"]:
            assert all(abs(v - j) >= length + C3.val_gap
                       for j in starts["train"] + starts["record"])  # fmt: skip
        alone = evaluate_content3(
            conf, sets_dir, tmp_path / "alone", codes=("mean-2",), **kw
        )
        assert (alone / "mean-2" / "content3.json").read_bytes() == \
            (out / "mean-2" / "content3.json").read_bytes()  # fmt: skip
        assert (alone / "intervals.json").read_bytes() == (
            out / "intervals.json"
        ).read_bytes()
        with pytest.raises(ValueError, match="no codes"):
            evaluate_content3(conf, sets_dir, tmp_path / "x", codes=("mean-3",), **kw)
        # the cheats' widths (any rows of the tiny sets) and the judge
        _, ws = tiny_sets()
        g = _gen(6)
        base = torch.randn(ws.n, 2, generator=g, dtype=F64)
        for name, x in (("base", base), ("mean-1", _past_mean(base, 1)),
                        ("mean-2", _past_mean(base, 2))):  # fmt: skip
            path = width / "guard" / "cheat" / name / "width.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(white_rows(x, ws, WTINY, full=False)))
        res = real3_stage(width, WTINY, TINY)
        assert set(res["b1"]["fronts"]) == {"mean-1", "mean-2"}
        assert res["b2"]["codes"] == ["base", "drop-1"]
        assert set(res["b3"]["pairs"]) == {"rot-base", "base-rot"}
        assert res["b3"]["comparable"]["init"] is None and not res["b3"]["pass"]
        assert set(res["b4"]["codes"]) == set(codes) and len(res["b4"]["margins"]) == 2
        far = far_runs(starts["record"], starts["train"], length + C3.val_gap)
        assert res["b4_far"]["runs"] == len(far)
        assert set(res["b4_far"]) == ({"runs"} | set(codes) if far else {"runs"})
        assert len(res["sigma_rot"]) == 2
        assert set(res["codes"]["base"]["chosen"]) == {"level", "change"}

    @pytest.mark.slow
    def test_a_models_step_0_is_its_yardstick(self, tmp_path):
        # a learned code (``--model``) of a run's step 0 is the yardstick
        # ``init_m`` of the run (zpca), of a VAE's t = 0 checkpoint
        # ``init_v16``: the same rows but the code's name
        root = tours_with_zurich(tmp_path / "d")
        mean = tiny_run(root, tmp_path, "--sigreg-on", "mean", name="mean")
        conf = replace(conf_of(), device="cpu")
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        ckpt.save_eval(vae_run, program="vae", group="iid", t=0.0, frame=10,
                       modules=build_vae(conf, seed=5).modules())  # fmt: skip
        first = sorted((vae_run / "ckpt").glob("*.pt"))[0]
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        kw: dict[str, Any] = dict(root=root, guard=GTINY, cconf=C3, gconf=TINY, wconf=WTINY,
                                  fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        width = tmp_path / "width"
        yard = evaluate_content3(conf, sets_dir, width / "content3",
                                 codes=("base", "rot", "init_m", "init_v16"),
                                 inits={"init_m": mean}, init_v16=vae_run, **kw)  # fmt: skip
        models = {"m0": (mean, 0), "m4": (mean, 4), "v16": (first, None)}
        out = evaluate_content3(
            conf, sets_dir, width / "content3m", models=models, **kw
        )
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
            "m0",
            "m4",
            "v16",
        ]

        def unnamed(path):
            return [{k: v for k, v in r.items() if k != "code"}
                    for r in json.loads(path.read_text())]  # fmt: skip

        for model, code in (("m0", "init_m"), ("v16", "init_v16")):
            assert unnamed(out / model / "content3.json") == unnamed(
                yard / code / "content3.json"
            )
        assert unnamed(out / "m4" / "content3.json") != unnamed(
            out / "m0" / "content3.json"
        )
        assert (out / "intervals.json").read_bytes() == (
            yard / "intervals.json"
        ).read_bytes()
        # the record's reading: v16 (a model of judge_geom.VAES) against its
        # own step 0 (the same code): D = (1 - 2^-psi*) E
        got = models3_stage(width, tmp_path / "no-geom", TINY)
        assert set(got["models"]) == {"v16"}
        v16 = got["models"]["v16"]
        assert v16["init"] == "init_v16" and set(v16["pairs"]) == {"pca", "init_v16"}
        e = v16["pairs"]["init_v16"]["E"]
        assert e[0] == e[1]
        assert v16["pairs"]["init_v16"]["D"]["point"] == pytest.approx(
            (1 - 2**-PSI_STAR) * e[0]
        )
        with pytest.raises(ValueError, match="named as codes"):
            evaluate_content3(conf, sets_dir, tmp_path / "x", models={"color": (mean, 0)},
                              **kw)  # fmt: skip
        with pytest.raises(ValueError, match="no checkpoint of step 3"):
            evaluate_content3(conf, sets_dir, tmp_path / "x", models={"m3": (mean, 3)},
                              **kw)  # fmt: skip

    def test_the_model_source(self):
        assert model_source("a=/r/x@60000") == ("a", (Path("/r/x"), 60000))
        assert model_source("v16=/r/ckpt/1.0.pt") == (
            "v16",
            (Path("/r/ckpt/1.0.pt"), None),
        )
        with pytest.raises(ValueError, match="NAME="):
            model_source("/r/x@6000")

    @pytest.mark.slow
    def test_the_frame_itself_is_read_back_aligned(self, tmp_path, monkeypatch):
        # the code: the stack's last frame pooled as the observation; the
        # least squares read it back (E = 1) and its change from the pair
        # and from the level's difference (E^d = 1) on every set
        # videos long enough for the runs of 16 stacks of the three sets
        g = _gen(9)
        frames = {v: torch.randint(256, (80, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
                  for v in ("A", "B", "C")} | {"H": SPLICED}  # fmt: skip
        root = write_tours(tmp_path / "d", frames)
        conf = replace(conf_of(), device="cpu")
        sets_dir = tmp_path / "tlin"
        _sets_at(sets_dir, conf, root, TINY)
        obs = 2

        class Frame:
            def raw(self, x: torch.Tensor) -> torch.Tensor:
                return pool_obs(x[:, -1].cpu(), obs).double()

            def white(self, raw: torch.Tensor) -> torch.Tensor:
                return raw.double()

            def mu(self, x: torch.Tensor) -> torch.Tensor:
                return self.raw(x).float()

        monkeypatch.setattr(content_eval, "_yard_white", lambda *a, **k: Frame())
        guard = replace(GTINY, cheat_count=4, cheat_length=16)
        cconf = replace(C3, val_gap=0)
        out = evaluate_content3(conf, sets_dir, tmp_path / "w", codes=("color",), root=root,
                                guard=guard, cconf=cconf, gconf=TINY, wconf=WTINY,
                                fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        rows = json.loads((out / "color" / "content3.json").read_text())
        for st in ("train", "val", "record", "eval"):
            for ro in ("lin", "lin-diff"):
                s = sums3(rows, ro, st)
                assert s.value() == pytest.approx(1, abs=1e-5)
                assert s.value(True) == pytest.approx(1, abs=1e-5)
