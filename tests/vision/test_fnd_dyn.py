r"""``exp.vision.fnd_dyn``: the sequences of stacks, the rows and the
synthetic checks of the minGRU.

The oracles: the stacks that do not overlap by counting; a video whose
pixels are their frame's number gives each stack its own frames; a cut
drops exactly the anchors whose span holds it; the two-point linear's
ratios do not move under an orthogonal map of the codes (ridge included:
``dynpred``'s invariance) and on an AR(1) of $r$ its $R_{\mathrm{clim}}$ is
$1 - r^{2\Delta}$ in both directions within the blocks' standard errors.
The checks (slow) are theory's (the module docstring): (a) an AR(1), (b)
AR(1) states seen mixed through noise (the Kalman filter, and better than
the two-point linear; its filter is each state's scalar one, whose prior
variance $M$ solves $M^2 + (R(1 - \phi^2) - q) M - qR = 0$, of a decay
$\phi R / (M + R)$ in (0, 1): a minGRU's unit), (c) a reversible process (the future and the past
alike), (d) white noise ($R_{\mathrm{clim}} \approx 1$); untrained (the
two-point linear alone) it fails (b), so the check is not empty.
"""

import json
import math

import pytest
import torch

from exp.lib.gru_dyn import GRUConf, seq_set
from exp.lib.synth_series import kalman_errors, ou
from exp.vision.fnd_dyn import (
    CheckConf,
    check_model,
    check_unit,
    cut_keep,
    encode_stacks,
    gru_rows,
    lin_rows,
    load_sets,
    rot_sets,
    save_sets,
    stack_starts,
)

SMALL = GRUConf(width=16, window=32, burn=4, lags=(1, 2, 4), batch=32, patience=5,
                max_epochs=300)  # fmt: skip
CC = CheckConf(k=3, n_train=30_000, n_val=5_000, n_test=30_000, block=200)
CPU = torch.device("cpu")


class TestStacks:
    @pytest.mark.parametrize(
        ("n", "frames", "stride", "want"),
        [
            (30, 8, 1, [0, 8, 16]),
            (32, 8, 1, [0, 8, 16, 24]),
            (30, 8, 2, [0]),
            (7, 8, 1, []),
        ],
    )
    def test_starts(self, n, frames, stride, want):
        assert stack_starts(n, frames, stride).tolist() == want

    def test_each_stack_is_its_own_frames(self):
        frames = torch.arange(50, dtype=torch.uint8)[:, None, None, None].expand(
            50, 3, 2, 2
        )
        starts = stack_starts(50, 4, 2)  # frames t, t+2, t+4, t+6; 8 apart
        got = encode_stacks(lambda x: x[:, :, 0, 0, 0].float(), lambda a, b: frames[a:b],
                            starts, 4, 2, CPU, lambda z: z, read=2)  # fmt: skip
        want = starts[:, None] + torch.arange(4) * 2
        assert got.tolist() == want.float().tolist()

    def test_a_cut_drops_the_anchors_whose_span_holds_it(self):
        """A cut between frames 20 and 21: with stacks 4 frames apart of
        span 3 and a margin of 1, anchor $t$ spans the frames $4(t-1) ..
        4(t+1)+3$."""
        before = torch.cat([torch.zeros(21), torch.ones(19)]).long()  # 40 frames
        starts = stack_starts(40, 4, 1)  # 10 stacks
        keep = cut_keep(before, starts, 3, 1)
        want = [t for t in range(1, 9) if not (4 * (t - 1) <= 20 < 4 * (t + 1) + 3)]
        assert torch.nonzero(keep).flatten().tolist() == want


def _ou_sets(r: float, k: int = 3) -> dict:
    def one(n: int, seed: int, shifted: bool = False):
        w = ou(n, k, r, torch.Generator().manual_seed(seed)).float()
        return seq_set(w, [n // 2, n - n // 2], SMALL.margin, 200, shifted=shifted)

    return {"train": one(20_000, 1), "val": one(4_000, 2), "rec": one(4_000, 3, True),
            "zurich": one(20_000, 4, True)}  # fmt: skip


class TestLinear:
    def test_an_orthogonal_map_moves_no_ratio(self):
        sets = _ou_sets(0.8)
        a, b = lin_rows(sets, SMALL), lin_rows(rot_sets(sets, 7), SMALL)
        for ra, rb in zip(a, b, strict=True):
            assert ra["scale"] == rb["scale"]
            za, zb = ra["sets"]["zurich"], rb["sets"]["zurich"]
            for key in ("num_b", "copy_b", "clim_b"):
                torch.testing.assert_close(torch.tensor(za[key]), torch.tensor(zb[key]),
                                           rtol=1e-5, atol=1e-6)  # fmt: skip

    def test_an_ar1_in_both_directions(self):
        """$R_{\\mathrm{clim}} = 1 - r^{2\\Delta}$ (the mean less than 0.02
        from 0 adds its square to the denominator; the blocks' SE)."""
        r = 0.8
        for row in lin_rows(_ou_sets(r), SMALL):
            z = row["sets"]["zurich"]
            num, clim = torch.tensor(z["num_b"]), torch.tensor(z["clim_b"])
            per = num / clim
            se = per.std(1) / math.sqrt(per.shape[1])
            got = num.sum(1) / clim.sum(1)
            want = torch.tensor([1 - r ** (2 * lag) for lag in row["lags"]])
            assert ((got - want).abs() <= 4 * se + 2e-3).all(), (row["dir"], got, want)


class TestRows:
    def test_a_gru_row_and_its_sets(self):
        conf = GRUConf(width=8, window=8, burn=2, lags=(1, 2), batch=16, patience=1,
                       max_epochs=2)  # fmt: skip
        sets = {n: seq_set(s.w, (s.bounds[1:] - s.bounds[:-1]).tolist(), conf.margin, 200,
                           shifted=n in ("rec", "zurich"))
                for n, s in _ou_sets(0.5).items()}  # fmt: skip
        row = gru_rows(sets, "past", 2, 11, conf, CPU)
        assert row["layers"] == 2 and row["dir"] == "past"
        assert len(row["val_curve"]) == row["epochs"] + 1
        assert set(row["sets"]) == {"val", "rec", "zurich"}
        assert "num_b_s" in row["sets"]["zurich"] and "num" in row["sets"]["val"]
        assert len(row["sets"]["zurich"]["num_b"]) == 2

    def test_sets_round_trip(self, tmp_path):
        sets = _ou_sets(0.5)
        save_sets(tmp_path / "x.pt", sets, {"code": "pca"})
        got, meta = load_sets(tmp_path / "x.pt")
        assert meta == {"code": "pca"}
        for n, s in sets.items():
            assert torch.equal(got[n].w, s.w) and torch.equal(got[n].anchors, s.anchors)
            assert got[n].n_blocks_s == s.n_blocks_s


class TestChecks:
    def test_the_kalman_check_is_each_states_scalar_filter(self):
        """The mixed AR(1) states' least errors are the sum of the scalar
        filters' $\\phi^{2\\Delta} P + 1 - \\phi^{2\\Delta} + R$ (a unit
        variance), $P = M R / (M + R)$, each of a decay in (0, 1)."""
        cc = CheckConf(k=5)
        lags = (1, 3, 8)
        got = kalman_errors(*check_model("kalman", cc), lags)
        want = torch.zeros(len(lags), dtype=torch.float64)
        noise = cc.noise
        for phi in torch.linspace(*cc.phis, cc.k, dtype=torch.float64).tolist():
            q = 1 - phi**2
            b = noise * (1 - phi**2) - q
            m = (-b + math.sqrt(b * b + 4 * q * noise)) / 2
            p = m * noise / (m + noise)
            assert 0 < phi * noise / (m + noise) < 1
            for i, lag in enumerate(lags):
                want[i] += phi ** (2 * lag) * p + 1 - phi ** (2 * lag) + noise
        torch.testing.assert_close(got, want, rtol=1e-9, atol=0)

    def test_the_two_point_linear_alone_fails_the_kalman_check(self):
        """Untrained, the predictor is the two-point linear: it does not
        beat itself, so the check is not empty."""
        row = check_unit(
            "kalman", GRUConf(**{**SMALL.__dict__, "max_epochs": 0}), CC, 7, CPU
        )
        assert not row["two_point"]["ok"] and not row["ok"]

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["ou", "kalman", "reverse", "iid"])
    def test_the_oracles(self, name):
        row = check_unit(name, SMALL, CC, 7, CPU)
        assert row["ok"], row
        if name == "kalman":
            assert row["two_point"]["ok"] and row["window_gap"]["ok"]
        if name == "reverse":
            assert row["symmetry"]["ok"] and row["past"]["ok"]
        if name == "iid":
            assert all(abs(v - 1) < 0.02 for v in row["future"]["r_clim"])
            assert all(abs(v - 0.5) < 0.01 for v in row["future"]["r_copy"])

    @pytest.mark.slow
    def test_two_layers_pass_the_kalman_check(self):
        row = check_unit(
            "kalman", GRUConf(**{**SMALL.__dict__, "layers": 2}), CC, 7, CPU
        )
        assert row["ok"], row


class TestCli:
    def test_rot_and_lin_of_saved_codes(self, tmp_path):
        """``rot`` writes ``pca``'s rotated codes and ``lin`` the two
        directions' rows of the decided lags (stamped with the code)."""
        from exp.lib.gru_dyn import LAGS
        from exp.vision.fnd_dyn import FndConf, main

        margin = FndConf().gru.margin

        def one(n: int, seed: int, shifted: bool = False):
            w = ou(n, 4, 0.9, torch.Generator().manual_seed(seed)).float()
            return seq_set(w, [n], margin, 160, shifted=shifted)

        sets = {"train": one(3000, 1), "val": one(1000, 2), "rec": one(1000, 3, True),
                "zurich": one(2000, 4, True)}  # fmt: skip
        save_sets(tmp_path / "codes" / "pca.pt", sets, {"code": "pca", "key": "k"})
        main(["rot", "--out", str(tmp_path)])
        rot, meta = load_sets(tmp_path / "codes" / "rot.pt")
        assert meta["code"] == "rot" and not torch.equal(
            rot["train"].w, sets["train"].w
        )
        target = main(["lin", "--out", str(tmp_path), "--code", "rot"])
        rows = json.loads(target.read_text())
        assert [r["dir"] for r in rows] == ["future", "past"]
        assert all(r["lags"] == list(LAGS) and r["code"] == "rot" for r in rows)
