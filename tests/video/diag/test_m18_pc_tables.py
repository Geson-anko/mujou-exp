"""The tables of results/m18.md's stage 4 (diagnosis): the verdict written
before the runs on hand-made values, the track of a pair against BP's range,
the summary of the gradient check (QP3, QP4) and QP6."""

import math

from exp.video.diag.m18_pc_tables import QP6_REL, grad_summary, pair_track, qp6, verdict

SEEDS = (100, 0, 1, 2)
INIT = {s: {"recon_blur": 0.3, "dyn_1/copy_1": 0.6} for s in SEEDS}
BP = {
    100: {"recon_blur": 0.14, "dyn_1/copy_1": 0.300},
    0: {"recon_blur": 0.15, "dyn_1/copy_1": 0.304},
    1: {"recon_blur": 0.16, "dyn_1/copy_1": 0.302},
    2: {"recon_blur": 0.13, "dyn_1/copy_1": 0.297},
}  # means 0.145 and 0.30075, ranges 0.03 and 0.007


def _shift(d: float, e: float = 0.0) -> dict[int, dict[str, float]]:
    return {
        s: {"recon_blur": v["recon_blur"] + d, "dyn_1/copy_1": v["dyn_1/copy_1"] + e}
        for s, v in BP.items()
    }


class TestVerdict:
    def test_bp_itself_passes_with_a_zero_difference(self):
        v = verdict(BP, BP, INIT)
        assert v["u"] == dict.fromkeys(SEEDS, True)
        assert v["U"] is True and v["D"] is True and v["passed"] is True
        rb = v["diff"]["recon_blur"]
        assert math.isclose(rb["mean_bp"], 0.145)
        assert rb["diff"] == 0.0
        assert math.isclose(rb["range_bp"], 0.03)

    def test_the_mean_difference_against_the_range(self):
        """Shifted by 0.029 (< 0.03) it is within; by 0.031 it is not."""
        assert verdict(_shift(0.029), BP, INIT)["diff"]["recon_blur"]["within"]
        v = verdict(_shift(0.031), BP, INIT)
        assert v["diff"]["recon_blur"]["within"] is False
        assert v["D"] is False and v["passed"] is False
        assert v["U"] is True  # 0.031 more is still below 0.3

    def test_u_is_per_seed_against_its_own_init(self):
        init = INIT | {1: {"recon_blur": 0.1, "dyn_1/copy_1": 0.6}}
        v = verdict(BP, BP, init)
        assert v["u"][1] is False and v["u"][0] is True
        assert v["U"] is False and v["passed"] is False

    def test_a_missing_value_is_unknown(self):
        pc = {s: dict(v) for s, v in BP.items()}
        del pc[2]["dyn_1/copy_1"]
        v = verdict(pc, BP, INIT)
        assert v["u"][2] is None and v["U"] is None
        assert v["diff"]["dyn_1/copy_1"]["within"] is None
        assert v["diff"]["recon_blur"]["within"] is True
        assert v["passed"] is None

    def test_a_diverged_run_fails_even_of_seed_0(self):
        """The seed 0 is falsy: the diverged set must still count."""
        v = verdict(BP, BP, INIT, diverged=frozenset({0}))
        assert v["u"][0] is False and v["U"] is False
        assert v["diff"]["recon_blur"]["within"] is False
        assert v["passed"] is False


class TestPairTrack:
    def test_the_largest_difference_and_the_first_time_out_of_range(self):
        pc = {0.0: 1.0, 60.0: 1.1, 120.0: 1.5, 180.0: 1.2}
        bp = {0.0: 1.0, 60.0: 1.0, 120.0: 1.0, 180.0: 1.0}
        # BP's range: 0 at 0 s, 0.2 at 60 s, 0.2 at 120 s, 0.3 at 180 s
        others = {0.0: 1.0, 60.0: 1.2, 120.0: 1.2, 180.0: 1.3}
        tr = pair_track(pc, bp, [bp, others])
        assert math.isclose(tr["max_abs"], 0.5) and tr["t_max"] == 120.0
        assert tr["first_out"] == 120.0  # 0.1 < 0.2 at 60 s, 0.5 > 0.2 at 120 s

    def test_never_out_and_without_bp_seeds(self):
        pc, bp = {60.0: 2.0}, {60.0: 2.0}
        assert pair_track(pc, bp, [bp, {60.0: 2.5}])["first_out"] is None
        tr = pair_track(pc, bp, [])
        assert tr == {"max_abs": 0.0, "t_max": 60.0, "first_out": None}


def _point(fails: list[str], pc64_rel: float, nll: float, kl: float, exact: float):
    rows = [{"rel": 1e-6}, {"rel": None}]
    return {
        "t": 0.0,
        "failures": fails,
        "loss": {"exact": exact, "pc64": {"nll": nll, "kl": kl}},
        **{k: {"max_rel": 1e-6} for k in ("bp", "pc", "pc64", "pc_bp")},
        "tensors": {"bp": rows, "pc64": [{"rel": pc64_rel}, {"rel": 0.0}]},
    }


class TestGradSummary:
    def test_the_ratios_and_qp3_qp4(self):
        g = grad_summary(
            {
                "init": _point([], 1e-15, 1000.0, 24.0, 1024.0 + 2.0**-20),
                "00300.0": _point([], 2e-15, 1000.0, 24.0, 1024.0),
            }
        )
        p = g["points"]["init"]
        # bound 2^-23 * 1024 = 2^-13; gap 2^-20: ratio 2^-7
        assert math.isclose(p["loss_ratio"], 2.0**-7)
        assert math.isclose(p["rel_ratio"], 1e-9)  # the None row is skipped
        assert g["QP3"] is True and g["QP4"] is True

    def test_failures_split_into_qp3_and_qp4(self):
        g = grad_summary({"init": _point(["loss"], 0.0, 1.0, 0.0, 1.0)})
        assert g["QP3"] is False and g["QP4"] is True
        g = grad_summary({"init": _point(["enc1.w"], 0.0, 1.0, 0.0, 1.0)})
        assert g["QP3"] is True and g["QP4"] is False


class TestQP6:
    def _rows(self, scale: float):
        stream = [{"t": float(t), "loss": 100.0 * t * scale} for t in range(1, 200)]
        evals = [
            {"t": t, "split": "val", "bpd": 7.0 * scale} for t in (60.0, 120.0, 180.0)
        ]
        return stream, evals

    def test_equal_runs_hold(self):
        s, e = self._rows(1.0)
        q = qp6(s, s, e, e)
        assert q["held"] is True and q["loss_rel"] == 0.0 and q["rows"] == 120

    def test_a_relative_difference_beyond_the_bound_fails(self):
        s, e = self._rows(1.0)
        s2, e2 = self._rows(1 + 2 * QP6_REL)
        q = qp6(s2, s, e2, e)
        assert q["held"] is False
        assert math.isclose(q["loss_rel"], 2 * QP6_REL, rel_tol=1e-6)

    def test_a_missing_value_is_unknown(self):
        s, e = self._rows(1.0)
        assert qp6(s, s, e[:1], e)["held"] is None
