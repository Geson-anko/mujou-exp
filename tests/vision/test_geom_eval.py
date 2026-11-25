r"""``exp.vision.geom_eval`` (V18c's time linearity and information width,
``plans/jepa.md``).

The oracles: the cut rule finds exactly the splice of two unrelated drifting
videos (a drift's neighbours differ far less than frames 20 apart, and the
splice far more); the triples are every centre whose frames lie in one block
(by brute force) and the strata partition them, ``slow`` below ``mid``
below ``fast``; the blocks' sums add up to the pooled $G$, and the
resampled $s$ of unit counts is the estimate; a code not scoreable at a $k$
has only its spectrum; the VAE's ``own`` of the token path is its $\mu$; the
JEPA's ``own`` is its $z$ and ``h_mean`` ``TokenCode``'s; the replicated
K3's ``h_mean`` is the JEPA's to the bit and its $z$ of rank at most $d$;
the checks V-d0a to V-d0f pass on short series; and the yard's job end to
end on tiny runs.
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
    block_triples,
    check_rows,
    code_rows,
    cut_rule,
    evaluate_yard,
    geom_sets,
    s_resamples,
)
from exp.vision.jepa import Mix, build_jepa
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
        sets = sets_of()
        lo, hi = sets.edges[2]
        assert lo <= hi


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

    def test_sensitivity_adds_the_whitenings_of_v_d3(self):
        sets = sets_of()
        rows = code_rows(*synthetic_pair(sets.n), sets, TINY, sens=True, labels={})
        got = {(r["k"], r["eps"], r["ref"]) for r in rows if r["part"] == "spec"}
        assert got == {(2, 1e-6, "all"), (1, 1e-6, "all"), (64, 1e-6, "all"),
                       (3, 1e-6, "all"), (2, 1e-3, "all"), (2, 1e-6, "front"),
                       (2, 1e-6, "back")}  # fmt: skip


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
        out = evaluate_yard(
            replace(conf, device="cpu"), tmp_path / "geom", inits=inits, init_v16=vae_run,
            root=root, gconf=TINY, fconf=FILL_TINY, spacing=EVERY, span=SPAN,
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
        init_c = json.loads((out / "yard" / "rows" / "init_c.json").read_text())
        assert {r["place"] for r in init_c} == {"own", "h_mean"}
        mids = [r for r in init_c if r["part"] == "mid"]
        assert {r["stratum"] for r in mids} == set(STRATA)
        assert all(math.isfinite(r["G"]) for r in mids if r["n_triples"])
        # the description's yard stage reads it all
        got = yard_stage(out, TINY)
        assert set(got["curves"]["own"]) >= {"pca", "init_c"}
        assert "init_v16" in got["curves"]["h_mean"]
        assert got["controls"]["pca_perm"]["lags"][2]["G_theory"] == 0.75
