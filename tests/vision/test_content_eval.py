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
"""

import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    content_sums,
    fit_linear,
    pool_obs,
    predict_linear,
)
from exp.lib.linwidth import windows
from exp.vision import content_eval
from exp.vision.content_eval import (
    ContentConf,
    SynthConf,
    calib_noise,
    calib_noise_d,
    change_explained,
    check_parts,
    check_units,
    content_rows,
    evaluate_content,
    front_of,
    merge_checks,
    negative,
    negative_cheats,
    orthogonal,
    positive,
)
from exp.vision.geom_eval import GeomConf, _sets_at
from exp.vision.results.v18c.judge_content import checks_stage, real_stage
from exp.vision.width_eval import (
    WidthConf,
    _common_sums,
    counts_of,
    synthetic_sets,
    white_rows,
)
from exp.vision.width_guard import _past_mean
from tests.vision.test_fill_eval import TINY as FILL_TINY
from tests.vision.test_gen_eval import features
from tests.vision.test_geom_eval import TINY, tours_with_zurich
from tests.vision.test_jepa_eval import SPAN, conf_of
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
