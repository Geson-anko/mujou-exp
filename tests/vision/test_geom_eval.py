r"""``exp.vision.geom_eval`` (V18c's time linearity and information width,
``plans/jepa.md``).

The oracles: the cut rule finds exactly the splice of two unrelated drifting
videos (a drift's neighbours differ far less than frames 20 apart, and the
splice far more); the triples are every centre whose frames lie in one block
(by brute force) and the strata partition them, ``slow`` below ``mid``
below ``fast`` (the displacement by brute force); the blocks' sums add up
to the pooled $G$, and the resampled $s$ of unit counts is the estimate; a
resample of no triple is dropped; a code not scoreable at a $k$ has only
its spectrum; a synthetic control's population rows are of the identity
metric; the near neighbours follow the given times; the spread's diagnosis
by hand; the VAE's ``own`` of the token path is its $\mu$; the JEPA's
``own`` is its $z$ and ``h_mean`` ``TokenCode``'s; the pairwise mean of
copies is the value to the bit, and the replicated K3's ``h_mean`` is the
JEPA's to the bit and its $z$ of rank at most $d$; the checks V-d0a to
V-d0f pass on short series; and the yard's and the run's jobs end to end on
tiny runs (the step 0 checkpoints, V16's of $t = 0$, sets of other settings
refused).
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import Tensor

from exp.lib import ckpt
from exp.lib.midpoint import Whitener, triple_terms
from exp.lib.synth_series import ou
from exp.vision.compose import to_dict
from exp.vision.geom_eval import (
    STRATA,
    CheckConf,
    GeomConf,
    JepaPlaces,
    K3Places,
    VaeTokenCode,
    _first_vae,
    _sets_at,
    _tree_mean,
    block_triples,
    check_rows,
    code_rows,
    cut_rule,
    evaluate_run,
    evaluate_yard,
    geom_sets,
    population_whitener,
    ratio_se,
    s_resamples,
    spread_diagnosis,
    twonn_rows,
)
from exp.vision.jepa import Mix, build_jepa
from exp.vision.jepa_diag import run_ckpts
from exp.vision.jepa_eval import TokenCode
from exp.vision.models import build_vae
from exp.vision.results.v18c.judge_geom import yard_stage
from tests.vision.test_eval import SIZE, write_tours
from tests.vision.test_fill_eval import TINY as FILL_TINY
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

CPU = torch.device("cpu")
TINY = GeomConf(
    lags=(2, 4),
    block=40,
    k=2,
    k_side=(1, 64),
    k_sens=(3,),
    cut_gap=20,
    cut_pairs=500,
    grid_step=2,
    spacings=(2, 4),
    near=4,
    boot=50,
    sub_draws=3,
    deltas=(0.01, 0.02),
    res_dims=(2, 3),
    res_phis=(0.0, 0.5),
    res_draws=2,
    res_points=40,
    pca_dim=4,
)


def drifting(n: int, seed: int) -> Tensor:
    """``n`` frames u8 ``(n, 3, SIZE, SIZE)`` drifting by 0.2 of a random
    pattern a frame from a random base."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, SIZE, SIZE, generator=g) * 100 + 70
    pattern = torch.randn(3, SIZE, SIZE, generator=g)
    t = torch.arange(n)[:, None, None, None]
    return (base + 0.2 * t * pattern).clamp(0, 255).round().to(torch.uint8)


SPLICED = torch.cat([drifting(60, 1), drifting(60, 2)])  # unrelated at 59 | 60


def sets_of(frames: Tensor = SPLICED, stack: int = 2):
    return geom_sets(
        frames, stack=stack, stride=1, gconf=TINY, dev=CPU, ident={"v": "t"}
    )


class TestSets:
    def test_the_cut_is_the_splice_alone(self):
        d, theta = cut_rule(SPLICED, gap=20, pairs=500, q=0.01, seed=1, dev=CPU)
        assert (d >= theta).nonzero().flatten().tolist() == [59]
        assert sets_of().cuts.tolist() == [59]

    def test_triples_lie_in_one_block(self):
        blocks = torch.arange(100) // 30
        for lag in (1, 3, 7):
            got = block_triples(blocks, 96, 4, lag)
            want = [
                u
                for u in range(lag, 96 - lag)
                if blocks[u - lag] == blocks[u + lag + 4]
            ]
            assert got.tolist() == want

    def test_strata_partition_the_triples(self):
        sets = sets_of()
        for lag, by in sets.triples.items():
            assert (
                sorted(torch.cat([by["in"], by["cross"]]).tolist())
                == by["all"].tolist()
            )
            parts = torch.cat([by["slow"], by["mid"], by["fast"]])
            assert sorted(parts.tolist()) == by["in"].tolist()
            sizes = [len(by[s]) for s in ("slow", "mid", "fast")]
            assert max(sizes) - min(sizes) <= 1
            # a triple crosses the cut iff its frames u - n .. u + n + 1 hold 59 and 60
            u = by["all"]
            crossed = (u - lag <= 59) & (u + lag + 1 >= 60)
            assert torch.equal(torch.isin(u, by["cross"]), crossed)
            assert len(set(by["all"].tolist())) == len(by["all"])

    def test_slow_moves_less_than_fast(self):
        # M by brute force from the frames: stack u is frames u, u + 1 (x / 255)
        sets = sets_of()
        x = SPLICED.double() / 255

        def m_of(u: int, lag: int) -> float:
            def stack(v: int) -> Tensor:
                return x[v : v + 2]

            a = (stack(u) - stack(u - lag)).square().sum()
            b = (stack(u + lag) - stack(u)).square().sum()
            return float(a + b) / 2

        def below(a: float, b: float) -> bool:  # the rank is of float32's sums
            return a <= b * (1 + 1e-6)

        for lag, by in sets.triples.items():
            m = {s: [m_of(int(u), lag) for u in by[s]] for s in ("slow", "mid", "fast")}
            lo, hi = sets.edges[lag]
            assert below(max(m["slow"]), lo) and below(lo, min(m["mid"]))
            assert below(max(m["mid"]), hi) and below(hi, min(m["fast"]))
            assert max(m["slow"]) < max(m["mid"]) < max(m["fast"])


def synthetic_pair(n: int, d: int = 4) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(5)
    return ou(n, d, 0.8, g).float(), ou(300, d, 0.8, g).float()


class TestCodeRows:
    def test_rows_add_up_to_the_pooled_values(self):
        sets = sets_of()
        ev, rf = synthetic_pair(sets.n)
        rows = code_rows(ev, rf, sets, TINY, sens=False, labels={"code": "x"})
        w = Whitener.fit(rf, 2).white(ev)
        (spread_row,) = (r for r in rows if r["part"] == "spread" and r["k"] == 2)
        for r in (r for r in rows if r["part"] == "mid" and r["k"] == 2):
            centers = sets.triples[r["lag"]][r["stratum"]]
            t = triple_terms(w, centers, r["lag"])
            assert r["G"] == pytest.approx(float(t.num.sum() / t.den.sum()), rel=1e-12)
            assert sum(r["num_b"]) == pytest.approx(float(t.num.sum()), rel=1e-12)
            assert sum(r["n_b"]) == r["n_triples"] == len(centers)
            ones = torch.ones(1, sets.n_blocks, dtype=torch.float64)
            den_b, n_b, w_b, ww_b, m_b = (
                torch.tensor(x, dtype=torch.float64)
                for x in (r["den_b"], r["n_b"], spread_row["w_b"],
                          spread_row["ww_b"], spread_row["m_b"])
            )  # fmt: skip
            s = s_resamples(den_b, n_b, w_b, ww_b, m_b, ones)
            assert float(s[0]) == pytest.approx(r["s"], rel=1e-10)

    def test_a_k_beyond_the_code_has_its_spectrum_only(self):
        sets = sets_of()
        rows = code_rows(*synthetic_pair(sets.n), sets, TINY, sens=False, labels={})
        wide = [r for r in rows if r["k"] == 64]
        assert [r["part"] for r in wide] == ["spec"] and not wide[0]["scoreable"]

    def test_the_twonn_rows(self):
        sets = sets_of()
        rows = code_rows(*synthetic_pair(sets.n), sets, TINY, sens=False, labels={})
        tw = [r for r in rows if r["part"] == "twonn" and r["k"] == 2]
        assert len(tw) == 2 * (2 + 4)  # two metrics, the offsets of 2 and 4
        assert all(
            "near_share" in r for r in tw if r["spacing"] == 2 and r["d"] is not None
        )
        subs = [r for r in rows if r["part"] == "twonn_sub"]
        assert {r["k"] for r in subs} == {2} and all(len(r["d"]) == 3 for r in subs)

    def test_a_synthetic_control_has_its_populations_rows(self):
        # population I: w = z / sqrt(1 + eps), so G equals the identity metric's
        sets = sets_of()
        ev, rf = synthetic_pair(sets.n)
        rows = code_rows(ev, rf, sets, TINY, sens=False, labels={}, population=True)
        pop = [r for r in rows if r["ref"] == "population"]
        assert {r["k"] for r in pop} == {4} and pop[0]["part"] == "spec"
        z = ev.double()
        for r in (r for r in pop if r["part"] == "mid" and r["n_triples"]):
            t = triple_terms(z, sets.triples[r["lag"]][r["stratum"]], r["lag"])
            assert r["G"] == pytest.approx(float(t.num.sum() / t.den.sum()), rel=1e-12)
        assert {r["part"] for r in pop} >= {"spread", "twonn", "twonn_sub"}
        torch.testing.assert_close(
            population_whitener(4, 1e-6).white(z), z / math.sqrt(1 + 1e-6),
            rtol=0, atol=0,
        )  # fmt: skip

    def test_the_near_neighbours_follow_the_given_times(self):
        # times far apart: no neighbour is near, and d_far is d
        sets = sets_of()
        points = synthetic_pair(sets.n)[0].double()[sets.grid]
        far = torch.arange(len(points), dtype=torch.float64) * 1000
        (row, *_) = twonn_rows(points, sets, TINY, sub=False, times=far)
        assert row["near_share"] == 0 and row["d_far"] == row["d"]
        (row, *_) = twonn_rows(points, sets, TINY, sub=False, times=far * 0)
        assert row["near_share"] == 1 and row["d_far"] is None

    def test_sensitivity_adds_the_whitenings_of_v_d3(self):
        sets = sets_of()
        rows = code_rows(*synthetic_pair(sets.n), sets, TINY, sens=True, labels={})
        got = {(r["k"], r["eps"], r["ref"]) for r in rows if r["part"] == "spec"}
        assert got == {(2, 1e-6, "all"), (1, 1e-6, "all"), (64, 1e-6, "all"),
                       (3, 1e-6, "all"), (2, 1e-3, "all"), (2, 1e-6, "front"),
                       (2, 1e-6, "back")}  # fmt: skip


class TestSpread:
    def test_the_diagnosis_by_hand(self):
        # rows +-a e1, +-b e2: mean 0, covariance diag(a^2, b^2) / 2
        a, b = 3.0, 1.0
        w = torch.tensor([[a, 0], [-a, 0], [0, b], [0, -b]], dtype=torch.float64)
        got = spread_diagnosis(w, top=1)
        assert got["V_over_k"] == pytest.approx((a * a + b * b) / 4, rel=1e-15)
        assert got["var_max"] == pytest.approx(a * a / 2, rel=1e-15)
        assert got["top_share"] == pytest.approx(a * a / (a * a + b * b), rel=1e-15)


class TestBootstrap:
    def test_a_resample_of_no_triple_is_dropped(self):
        # one block of two holds the triples: half the draws have none
        num = torch.tensor([[2.0, 0.0]], dtype=torch.float64)
        den = torch.tensor([[4.0, 0.0]], dtype=torch.float64)
        g = replace(TINY, boot=200)
        assert ratio_se(num, den, torch.tensor([1.0]), g) == 0.0  # G = 1/2 in each
        assert math.isnan(ratio_se(num * 0, den * 0, torch.tensor([1.0]), g))


def stacks_u8(n: int) -> Tensor:
    g = torch.Generator().manual_seed(n)
    return torch.randint(256, (n, 2, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)


class TestPlaces:
    def test_the_vaes_own_is_its_mu(self):
        mods = build_vae(conf_of(), seed=0)
        x = stacks_u8(3)
        with torch.no_grad():
            got = VaeTokenCode(mods).places(x)
            torch.testing.assert_close(got["own"], mods.mu(x), rtol=0, atol=0)
        assert got["h_mean"].shape == (3, conf_of().model.width)

    def test_the_jepas_own_and_h_mean(self):
        mods = build_jepa(conf_of(), seed=0, token_dim=4, latent_dim=32)
        x = stacks_u8(3)
        with torch.no_grad():
            got = JepaPlaces(mods).places(x)
            assert torch.equal(got["own"], mods.mu(x))
            torch.testing.assert_close(got["h_mean"].float(), TokenCode(mods, "h_mean").mu(x),
                                       rtol=1e-6, atol=1e-6)  # fmt: skip

    def test_the_k3_h_mean_is_the_h_mean_to_the_bit(self):
        mods = build_jepa(conf_of(), seed=0, token_dim=4, latent_dim=32)
        x = stacks_u8(16)
        with torch.no_grad():
            k3, plain = K3Places(mods).places(x), JepaPlaces(mods).places(x)
        assert torch.equal(k3["h_mean"], plain["h_mean"])
        # z = W (1_T kron I_d) tbar: in the span of the d columns of W (1 kron I)
        assert isinstance(mods.mix, Mix)
        d = 4
        basis = mods.mix.weight.double().reshape(32, -1, d).sum(1)  # (D, d)
        z = k3["own"].double()
        resid = z - z @ torch.linalg.pinv(basis).T @ basis.T
        assert float(resid.norm()) <= 1e-5 * float(z.norm())


class TestTreeMean:
    @pytest.mark.parametrize("t", [1, 2, 64, 256])
    def test_copies_give_the_value_to_the_bit(self, t):
        v = torch.randn(50, 7, generator=torch.Generator().manual_seed(t)).double()
        v = v / 3  # a full float64 mantissa
        assert torch.equal(_tree_mean(v[:, None].expand(-1, t, -1)), v)

    def test_t_must_be_a_power_of_2(self):
        with pytest.raises(ValueError, match="power of 2"):
            _tree_mean(torch.zeros(1, 3, 2, dtype=torch.float64))


class TestChecks:
    @pytest.mark.slow
    def test_v_d0a_to_v_d0f_pass_on_short_series(self):
        cc = CheckConf(n=3000, draws=32, rotation_length=2800)
        rows = check_rows(cc, GeomConf(lags=(8, 16, 32, 64)))
        failed = [r for r in rows if r["part"] == "check" and not r["pass"]]
        assert not failed, failed[:3]
        names = {r["name"] for r in rows if r["part"] == "check"}
        assert {"V-d0a-line", "V-d0a-rotation", "V-d0b", "V-d0c-affine", "V-d0d",
                "V-d0e", "V-d0f-hand", "V-d0f-cheb1", "V-d0f-gauss4"} <= names  # fmt: skip
        assert sum(r["name"] == "V-d0b" for r in rows) == 12 * 4 * 2


def tours_with_zurich(root: Path) -> Path:
    g = torch.Generator().manual_seed(0)
    frames = {v: torch.randint(256, (20, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
              for v in ("A", "B", "C")}  # fmt: skip
    frames["H"] = SPLICED
    return write_tours(root, frames)


class TestYard:
    @pytest.mark.slow
    def test_the_yard_end_to_end(self, tmp_path):
        root = tours_with_zurich(tmp_path / "d")
        mix = tiny_run(root, tmp_path, "--patch", "2", "--token-dim", "16", "--proj",
                       "fixed", "--sigreg-on", "mix", name="mix")  # fmt: skip
        mean = tiny_run(root, tmp_path, "--sigreg-on", "mean", name="mean")
        conf = conf_of()
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        ckpt.save_eval(vae_run, program="vae", group="iid", t=0.0, frame=10,
                       modules=build_vae(conf, seed=5).modules())  # fmt: skip
        inits = {
            "init": mix,
            "init_f": mix,
            "init_m": mean,
            "init_c": mix,
            "init_c1": mix,
        }
        same = {"init_c": [mix, mix], "init_c1": [mix]}
        out = evaluate_yard(
            replace(conf, device="cpu"), tmp_path / "geom", inits=inits, init_v16=vae_run,
            same=same, root=root, gconf=TINY, fconf=FILL_TINY, spacing=EVERY, span=SPAN,
        )  # fmt: skip
        sets = json.loads((out / "sets.json").read_text())
        assert sets["cuts"] == [59]
        names = sorted(p.stem for p in (out / "yard" / "rows").iterdir())
        want = ["cheb_a", "cheb_b", "cheb_c", "color", "init", "init_c", "init_c1",
                "init_f", "init_m", "init_v16", "k3_hmean", "k3_z", "mean8",
                *(f"ou_mix_{c}" for c in TINY.mix_cs), "pca", "pca_perm"]  # fmt: skip
        assert names == sorted(want)
        vd = json.loads((out / "yard" / "vd.json").read_text())
        checks = {r["name"] for r in vd if r["part"] == "check"}
        assert {
            "V-d0h-k3_hmean",
            "V-d0h-k3_z",
            "V-d0h-reverse",
            "V-d1",
            "V-d2",
        } <= checks
        for r in vd:
            if r["name"] in ("V-d0h-k3_hmean", "V-d0h-reverse"):
                assert r["pass"], r
        same_rows = [r for r in vd if r["name"] == "step0_same"]
        assert [(r["init"], r["same"]) for r in same_rows] == [
            ("init_c", True),
            ("init_c1", True),
        ]
        init_c = json.loads((out / "yard" / "rows" / "init_c.json").read_text())
        assert {r["place"] for r in init_c} == {"own", "h_mean"}
        assert {(r["source"], r["t"]) for r in init_c} == {
            (str(run_ckpts(mix)[0]), float(run_ckpts(mix)[0].stem))
        }
        v16 = json.loads((out / "yard" / "rows" / "init_v16.json").read_text())
        assert {r["t"] for r in v16} == {0.0}
        mix1 = json.loads((out / "yard" / "rows" / "ou_mix_1.json").read_text())
        assert {r["ref"] for r in mix1} == {"all", "population"}
        mids = [r for r in init_c if r["part"] == "mid"]
        assert {r["stratum"] for r in mids} == set(STRATA)
        assert all(math.isfinite(r["G"]) for r in mids if r["n_triples"])
        # the description's yard stage reads it all
        got = yard_stage(out, TINY)
        assert set(got["curves"]["own"]) >= {"pca", "init_c"}
        assert "init_v16" in got["curves"]["h_mean"]
        assert got["controls"]["pca_perm"]["theory"][2]["G"] == 0.75
        assert got["controls"]["ou_mix_1"]["main"] == "population"
        assert got["controls"]["pca_perm"]["main"] == "ref-k2"
        assert [r["same"] for r in got["step0_same"]] == [True, True]
        # the sets of other settings are not used again
        tours = replace(conf.data.tours, stride=conf.data.tours.stride + 1)
        other = replace(conf, device="cpu", data=replace(conf.data, tours=tours))
        with pytest.raises(ValueError, match="other settings"):
            _sets_at(out, other, root, TINY)
        # the run job: the K3's rows of a run with W next to its own
        geom = evaluate_run(mix, steps=(0,), sets_dir=out, root=root, device="cpu",
                            gconf=TINY, fconf=FILL_TINY, spacing=EVERY, span=SPAN)  # fmt: skip
        k3 = json.loads((geom / "k3" / "000000.json").read_text())
        assert {(r["code"], r["place"]) for r in k3} == {
            ("k3_z", "own"),
            ("k3_hmean", "h_mean"),
        }
        assert (geom / "rows" / "000000.json").exists()

    def test_v16_must_start_untrained(self, tmp_path):
        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "600.0.pt").touch()
        with pytest.raises(ValueError, match="not of t = 0"):
            _first_vae(tmp_path)
        (tmp_path / "ckpt" / "0.0.pt").touch()
        assert _first_vae(tmp_path).name == "0.0.pt"
