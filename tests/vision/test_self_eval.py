r"""``exp.vision.self_eval`` (V18c's identification of the fill in its own
space, ``plans/jepa.md``).

The oracles: Codex's counterexample to the separate centring (a perfect
prediction is found at exactly 1 with the reference's fixed centres, at 0
and 5/6 with each side's own); a position-only part (a large power of two
added to every vector, the reference's too) changes no value; the closed
form of V-s6 against a Monte Carlo of the noise; the bands' ratios average
to the whole by their variance, 0 for the real and 1 for the reference's
mean on the reference; the locality's $a_U$, $a_V$ of tokens built to move
by $\alpha \delta$ inside and $\beta \delta$ outside (Codex's seam code),
of a local code (1, 0), of a mirror code ($a_U$ 0), of a K3 ($L$ 0) and of
a position-only code (none). On synthetic sets: ``pred`` and
``pool`` are their definitions computed apart (V-s3 checks only the scores'
path of ``pool``); a tiny JEPA passes the implementation checks V-s0, V-s1,
V-s3 (with $L$) and V-s6, ``const`` ties all (V-s2), ``oracle`` is found at
1; the local colour code and its mirror pass V-s4 and V-s5, and the
mirror's ``swap`` is the local code's per query; and the run's job end to
end.
"""

import json
import math
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor

from exp.lib.metrics import pair_wins
from exp.lib.regions import EVAL_MASKS
from exp.vision.gen_eval import hard_sets
from exp.vision.jepa import build_jepa, expand_masks, predict
from exp.vision.jepa_eval import ColorCode
from exp.vision.models import early_fusion
from exp.vision.self_eval import (
    MASKS,
    MirrorColorCode,
    SelfConf,
    SelfSets,
    _mask_stages,
    band_ratios,
    check_rows,
    evaluate_checks,
    evaluate_run,
    expected_pair_wins,
    local_sums,
    local_terms,
    quarter_cells,
    stage_scores,
    step_rows,
)
from tests.vision.test_eval import SIZE, N
from tests.vision.test_fill_eval import TINY, by
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

CPU = torch.device("cpu")


def picks(g: int, n: int, reach: int = 1) -> dict[str, Any]:
    cand = torch.arange(n) * (g - 1) // (n - 1)
    hard_q, among = hard_sets(cand, g, reach, 1)
    return {"cand": cand, "hard_q": hard_q, "hard_among": among}


class TestStageScores:
    def test_codexs_counterexample_of_the_centring(self):
        # the grid (1, 0), (-1, 0), (0, 1), (0, -1), the queries its 0 and 2
        real = torch.tensor([[1.0, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64)
        cand = torch.tensor([0, 2])
        query = real[cand]
        zero = torch.zeros(2, dtype=torch.float64)
        among = torch.tensor([[0, 1, 2, 3], [2, 0, 1, 3]])  # its own first
        got = stage_scores(
            query, zero, real, zero, cand=cand, hard_q=torch.arange(2), hard_among=among
        )
        assert got[("hard", "cos")]["A"] == 1.0 and got[("hard", "cos")]["top1"] == 1.0
        # each side's own mean: the queries' is (1/2, 1/2)
        own = stage_scores(
            query, query.mean(0), real, zero,
            cand=cand, hard_q=torch.arange(2), hard_among=among,
        )  # fmt: skip
        assert own[("hard", "cos")]["top1"] == 0.0
        assert own[("hard", "cos")]["A"] == pytest.approx(5 / 6, abs=1e-15)

    def test_a_position_only_part_changes_no_value(self):
        g = torch.Generator().manual_seed(0)
        real = torch.randint(-50, 50, (12, 6), generator=g).double()
        query = real[[0, 3, 6, 9, 11]] + torch.randint(-9, 9, (5, 6), generator=g)
        ref_q = torch.randint(-50, 50, (8, 6), generator=g).double()  # 8: exact means
        ref_r = torch.randint(-50, 50, (8, 6), generator=g).double()
        pos = 2.0**20 * torch.randint(1, 9, (6,), generator=g).double()
        kw = picks(12, 5)

        def scores(shift: Tensor) -> dict:
            return stage_scores(
                query + shift, (ref_q + shift).mean(0), real + shift,
                (ref_r + shift).mean(0), **kw,
            )  # fmt: skip

        plain, moved = scores(torch.zeros(6, dtype=torch.float64)), scores(pos)
        assert plain.keys() == moved.keys()
        for key in plain:
            assert plain[key]["A_i"] == moved[key]["A_i"], key
            assert plain[key]["hit_i"] == moved[key]["hit_i"], key

    def test_a_constant_ties_with_every_candidate(self):
        g = torch.Generator().manual_seed(1)
        real = torch.randn(12, 6, generator=g, dtype=torch.float64)
        query = torch.full((5, 6), 3.0, dtype=torch.float64)
        got = stage_scores(query, query[0], real, real.mean(0), **picks(12, 5))
        for s in ("easy", "hard", "hard_c"):
            row = got[(s, "cos")]
            assert row["A"] == 0.5 and row["top1"] == 0.0
            assert row["n_tie"] == len(row["A_i"])


class TestExpectedPairWins:
    @pytest.mark.parametrize("hard", [False, True])
    def test_the_closed_form_is_the_noises_average(self, hard):
        g = torch.Generator().manual_seed(2)
        real = torch.randn(10, 3, generator=g, dtype=torch.float64)
        kw: dict = {}
        if hard:
            kw = {"truth": torch.tensor([2, 5, 7]), "among": None}
            kw["among"] = torch.stack([torch.arange(i - 2, i + 3) for i in (2, 5, 7)])
        truth = kw.get("truth", torch.arange(10))
        sigma = 0.7
        want = expected_pair_wins(real, sigma, **kw)
        draws = 10_000
        noise = torch.randn(draws, len(truth), 3, generator=g, dtype=torch.float64)
        got = [
            pair_wins(real[truth] + sigma * noise[i], real, **kw)[0]
            for i in range(draws)
        ]
        mean = sum(got) / draws
        sd = math.sqrt(sum((v - mean) ** 2 for v in got) / (draws - 1))
        assert abs(mean - want) < 4 * sd / math.sqrt(draws)
        assert 0.5 < want < 1.0

    def test_it_falls_from_1_to_one_half(self):
        g = torch.Generator().manual_seed(3)
        real = torch.randn(10, 4, generator=g, dtype=torch.float64)
        values = [expected_pair_wins(real, s) for s in (1e-6, 0.3, 1.0, 3.0, 1e6)]
        assert values[0] == pytest.approx(1.0)
        assert values[-1] == pytest.approx(0.5, abs=1e-5)
        assert all(a > b for a, b in zip(values, values[1:], strict=False))


class TestBands:
    def setup_method(self):
        g = torch.Generator().manual_seed(4)
        self.ref = torch.randn(
            200, 20, generator=g, dtype=torch.float64
        ) * torch.linspace(3, 0.1, 20, dtype=torch.float64)
        self.mean = self.ref.mean(0)
        cov = (self.ref - self.mean).T @ (self.ref - self.mean)
        self.basis = torch.linalg.eigh(cov)[1].flip(1)
        self.edges = (4, 16, 64, 256)
        self.real = torch.randn(50, 20, generator=g, dtype=torch.float64)
        self.pred = self.real + 0.5 * torch.randn(
            50, 20, generator=g, dtype=torch.float64
        )

    def test_the_bands_average_to_the_whole_by_their_variance(self):
        got = band_ratios(self.pred, self.real, self.mean, self.basis, self.edges)
        assert len(got["ratio"]) == len(got["share"]) == 3  # 1-4, 5-16, 17-20
        whole = sum(r * s for r, s in zip(got["ratio"], got["share"], strict=True))
        assert whole == pytest.approx(got["total"], abs=1e-9)
        assert sum(got["share"]) == pytest.approx(1.0, abs=1e-12)
        err = (self.pred - self.real).square().sum()
        assert got["total"] == pytest.approx(
            float(err / (self.real - self.mean).square().sum()), abs=1e-12
        )

    def test_the_real_is_0_and_the_mean_1_on_the_reference(self):
        zero = band_ratios(self.real, self.real, self.mean, self.basis, self.edges)
        assert zero["ratio"] == [0.0, 0.0, 0.0] and zero["total"] == 0.0
        mean = self.mean.expand(200, -1)
        one = band_ratios(mean, self.ref, self.mean, self.basis, self.edges)
        assert one["ratio"] == pytest.approx([1.0] * 3, abs=1e-12)


class TestLocality:
    def tokens(self, b: int = 6, t: int = 16, d: int = 3, seed: int = 5):
        g = torch.Generator().manual_seed(seed)
        t_m = torch.randn(b, t, d, generator=g, dtype=torch.float64)
        t_j = torch.randn(t, d, generator=g, dtype=torch.float64)
        return t_m, t_j

    @pytest.mark.parametrize(("alpha", "beta"), [(1.0, 0.0), (0.3, 0.2), (-1.0, 0.0)])
    def test_codexs_seam_code(self, alpha, beta):
        t_m, t_j = self.tokens()
        inside, seam_u, seam_v = quarter_cells("tl", 1)
        delta = t_m - t_j
        alpha_t = torch.tensor(alpha, dtype=torch.float64)
        move = torch.where(inside[:, None], alpha_t, beta)
        got = local_terms(
            local_sums(t_j + move * delta, t_j, t_m), inside, seam_u, seam_v
        )
        assert got["a_U"] == pytest.approx(alpha, abs=1e-12)
        assert got["a_V"] == pytest.approx(beta, abs=1e-12)
        assert got["L"] == pytest.approx(alpha - beta, abs=1e-12)
        assert got["rho_V"] == pytest.approx(beta**2, abs=1e-12)
        assert got["eta_U"] == pytest.approx((alpha - 1) ** 2, abs=1e-12)
        for part in ("a_U_seam", "a_U_in"):
            assert got[part] == pytest.approx(alpha, abs=1e-12)
        for part in ("a_V_seam", "a_V_far"):
            assert got[part] == pytest.approx(beta, abs=1e-12)

    def test_a_k3_has_l_0_and_a_position_only_code_none(self):
        t_m, t_j = self.tokens()
        inside, seam_u, seam_v = quarter_cells("br", 1)
        u = t_m.mean(1, keepdim=True).expand_as(t_m)  # the K3 of the candidates
        k3_tr = 0.5 * (u + t_j.mean(0))  # any one value of every position
        got = local_terms(
            local_sums(k3_tr, t_j.mean(0).expand_as(t_j), u), inside, seam_u, seam_v
        )
        assert got["L"] is not None and abs(got["L"]) <= 1e-12
        same = t_j.expand_as(t_m)
        none = local_terms(local_sums(same, t_j, same), inside, seam_u, seam_v)
        assert none["a_U"] is None and none["a_V"] is None and none["L"] is None

    def test_the_cells_of_a_quarter(self):
        inside, seam_u, seam_v = quarter_cells("tl", 1)
        assert inside.tolist() == EVAL_MASKS["Q_tl"].tolist()
        assert seam_u.nonzero().flatten().tolist() == [1, 4, 5]  # not the corner 0
        assert seam_v.nonzero().flatten().tolist() == [2, 6, 8, 9]
        fine = quarter_cells("tl", 2)[0]
        assert fine.tolist() == expand_masks(EVAL_MASKS["Q_tl"][None], 2)[0].tolist()


class TestMirror:
    def test_it_is_the_colour_code_turned_half_a_turn(self):
        g = torch.Generator().manual_seed(6)
        x = torch.randint(256, (2, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
        local = ColorCode(4).tokens(x).reshape(2, 4, 4, -1)
        mirror = MirrorColorCode(4).tokens(x).reshape(2, 4, 4, -1)
        for a in range(4):
            for b in range(4):
                assert torch.equal(mirror[:, a, b], local[:, 3 - a, 3 - b])


# -- on synthetic sets --------------------------------------------------------------------


def synthetic_sets(grid: int = 20, n_cand: int = 8, refs: int = 16) -> SelfSets:
    g = torch.Generator().manual_seed(7)

    def stacks(b: int) -> Tensor:
        return torch.randint(256, (b, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)

    p = picks(grid, n_cand)
    return SelfSets(
        ref=stacks(refs),
        grid=stacks(grid),
        cand=p["cand"],
        hard_q=p["hard_q"],
        hard_among=p["hard_among"],
        contexts=torch.tensor([0, 5]),
        ident={},
    )


SCONF = SelfConf(contexts=2)  # V-s6's registered 16 draws and tolerance


@pytest.fixture(scope="module")
def tiny_rows():
    mods = build_jepa(conf_of(), seed=0, token_dim=4)
    return step_rows(
        mods, synthetic_sets(), SCONF, expand=1, k_run=8, transplant=True, dev=CPU
    )


class TestMaskStages:
    """``pred`` and ``pool`` against their definitions computed apart:
    ``jepa.predict``'s outputs at $Q_q$ in order of position, and the mean
    over the visible positions of ``proj(encoder(x, keep))``."""

    @pytest.mark.parametrize(("q", "k"), [("tl", 4), ("br", 8), ("tr", 12)])
    def test_pred_and_pool_are_their_definitions(self, q, k):
        mods = build_jepa(conf_of(), seed=0, token_dim=4)
        x = synthetic_sets().grid[:5]
        mask = MASKS[(q, k)]
        q_pos = EVAL_MASKS[f"Q_{q}"].nonzero().flatten()
        u = torch.zeros(len(x), 4, dtype=torch.float64)
        got = _mask_stages(mods, x, u, mask, q_pos, None, CPU)
        images = early_fusion(x)
        with torch.no_grad():
            z_hat, hide = predict(mods, images, mask[None].expand(len(x), -1))
            keep = (~mask).nonzero().flatten()[None].expand(len(x), -1)
            ctx = mods.proj(mods.encoder(images, keep)).double()
        full = torch.zeros(len(x), len(mask), 4, dtype=torch.float64)
        full.scatter_(1, hide[..., None].expand(-1, -1, 4), z_hat.double())
        assert torch.equal(got["pred"], full[:, q_pos].flatten(1))
        pool = ctx.mean(1, keepdim=True).expand(-1, len(q_pos), -1).flatten(1)
        torch.testing.assert_close(got["pool"], pool, rtol=0, atol=1e-12)
        assert set(got) == {"pred", "pool", "flat", "pred_k3", "flat_k3", "pool_k3"}


class TestStepRows:
    def test_the_implementation_checks_pass(self, tiny_rows):
        checks = {r["name"]: r for r in by(tiny_rows, part="check")}
        assert {"V-s0", "V-s1", "V-s2", "V-s3", "V-s3-L", "V-s6"} == set(checks)
        for name in checks:
            assert checks[name]["pass"], (name, checks[name])
        assert checks["V-s3-L"]["max_abs_L"] <= 1e-9
        assert all(math.isfinite(z) for z in checks["V-s6"]["z"])

    def test_the_oracle_is_found_and_the_constant_ties(self, tiny_rows):
        rows = by(tiny_rows, part="id", cand="nat", metric="cos", quarter="mean")
        for r in by(rows, stage="oracle"):
            if r["set"] != "hard_c":
                assert r["A"] == 1.0 and r["top1"] == 1.0
        for r in by(rows, stage="const"):
            assert r["A"] == 0.5 and r["top1"] == 0.0

    def test_the_rows_of_each_part(self, tiny_rows):
        ids = by(tiny_rows, part="id")
        assert {r["k"] for r in ids} == {4, 8, 12}
        assert {r["in_dist"] for r in ids if r["k"] == 8} == {True}
        assert {r["cand"] for r in ids} == {"nat", "fwd", "rev"}
        assert {r["j"] for r in by(ids, cand="fwd")} == {0, 1, "mean"}
        assert {r["stage"] for r in by(ids, cand="fwd")} == {
            "pred", "pool", "flat", "interp", "oracle_tr", "const",
        }  # fmt: skip
        assert {r["quarter"] for r in ids} == {"tl", "tr", "bl", "br", "mean"}
        swaps = by(tiny_rows, part="swap")
        assert {r["stage"] for r in swaps} == {"pred", "oracle"}
        assert all(0 <= r["swap"] <= 1 for r in swaps)
        local = by(tiny_rows, part="local")
        assert {(r["quarter"], r["j"]) for r in local} >= {("tl", 0), ("mean", "mean")}
        assert len(by(tiny_rows, part="spec")) == 4
        bands = by(tiny_rows, part="band")
        assert all(len(r["ratio"]) == 2 for r in bands)  # 16 dimensions: 1-4, 5-16
        comps = by(tiny_rows, part="comp")
        assert {r["comp_k"] for r in comps} == {4}  # 16 leaves nothing of 16

    def test_without_the_transplant_no_rows_of_it(self):
        mods = build_jepa(conf_of(), seed=0, token_dim=4)
        rows = step_rows(
            mods, synthetic_sets(), SCONF, expand=1, k_run=4, transplant=False, dev=CPU
        )
        assert not by(rows, part="local")
        assert {r["cand"] for r in by(rows, part="id")} == {"nat"}


class TestChecks:
    def test_the_local_code_and_its_mirror_pass(self):
        rows = check_rows(synthetic_sets(), SCONF, patch=4, dev=CPU)
        checks = {r["name"]: r for r in by(rows, part="check")}
        assert checks["V-s4"]["pass"], checks["V-s4"]
        assert checks["V-s5"]["pass"], checks["V-s5"]
        mirror = by(rows, part="local", code="mirror", quarter="mean", j="mean")[0]
        assert mirror["a_U"] == 0.0

    def test_the_mirrors_swap_is_the_local_codes(self):
        # the mirror's quarter A is the local code's B, its positions reversed:
        # <rev r_B - rev r_A, rev r_B> = <r_B - r_A, r_B>, both directions
        # averaged; swap is described only (no identity of the real's value)
        rows = check_rows(synthetic_sets(), SCONF, patch=4, dev=CPU)
        local = by(rows, part="swap", code="local")[0]
        mirror = by(rows, part="swap", code="mirror")[0]
        assert mirror["swap_i"] == local["swap_i"]
        assert "swap" not in by(rows, part="check", name="V-s5")[0]["each"]


# -- the jobs ----------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    from tests.vision.test_eval import random_tours

    return random_tours(tmp_path / "d")


class TestEvaluate:
    @pytest.mark.slow
    def test_a_tiny_run_and_the_checks(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        kw: dict[str, Any] = {"fconf": TINY, "sconf": SCONF, "root": root, "span": SPAN}
        out = evaluate_run(
            run, steps=(0, 4), transplant_steps=(4,), spacing=EVERY, **kw
        )
        names = sorted(p.name for p in (out / "rows").iterdir())
        assert names == ["000000.json", "000004-tr.json", "000004.json"]
        config = json.loads((out / "config.json").read_text())
        assert config["self"]["contexts"] == 2 and len(config["contexts"]) == 2
        rows = json.loads((out / "rows" / "000004.json").read_text())
        assert all(r["run"] == run.name and r["step"] == 4 for r in rows)
        with pytest.raises(ValueError, match="other settings"):
            evaluate_run(
                run, steps=(4,), spacing=EVERY,
                fconf=TINY, sconf=replace(SCONF, noise_draws=5), root=root, span=SPAN,
            )  # fmt: skip
        checks = evaluate_checks(conf_of(), tmp_path / "checks", spacing=EVERY, **kw)
        rows = json.loads((checks / "rows.json").read_text())
        assert {r["name"] for r in by(rows, part="check")} == {"V-s4", "V-s5"}
