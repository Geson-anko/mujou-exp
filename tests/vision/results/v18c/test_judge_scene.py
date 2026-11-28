r"""``exp.vision.results.v18c.judge_scene`` (the judgement of V18c's scene
measure).

The oracles: the three readings at their edges by hand (each claim needs the
jackknife's 9 values on its side: ``above`` and ``below`` of 0, ``excl`` of
$\Delta_*$); a code against itself has $d = 0$, $\mathrm{SE}^* = 0$ and reads
``excl``; the audit's shares counted by hand, a group with no answer but the
unsure failing and an answer not of the ids leaving it undecided; the pooled
seed variance by hand; V-c4's null of labels independent of everything: the
sd of the 20 draws' $m$ is the pooled bootstrap $\mathrm{SE}^*$ of the draws
themselves (the law of total variance, $\mathbb E[m \mid \hat y] = 0$), and
a prediction constant within each video leaves the size unchecked; the hits
of a Markov chain that keeps its value with probability $\phi$ have $\tau =
(1 + \phi) / (1 - \phi)$.
"""

import math

import pytest
import torch

from exp.lib.readout import scene_m
from exp.vision.results.v18c.judge_scene import (
    UNSURE_LABEL,
    Scene,
    audit,
    compare,
    e1f_premise,
    hit_tau,
    null_check,
    reading,
    seed_var,
)
from exp.vision.scene_eval import DELTA_STAR, N_RAND


def scene_of(t: torch.Tensor, video: torch.Tensor, k: int, length: int) -> Scene:
    """A ``Scene`` of the kept labels ``t`` (the quarters the same)."""
    names = [f"v{i}" for i in range(int(video.max()) + 1)]
    meta = {
        "video": video.tolist(),
        "videos": names,
        "held_out": names[-1],
        "full": {"t": t.tolist(), "K": k, "L": length, "L_max": 4 * length},
    }
    for q in ("tl", "tr", "bl", "br"):
        meta[f"quad/{q}"] = {"t": t.tolist()}
    return Scene(meta)


def markov(n: int, phi: float, k: int, g: torch.Generator) -> torch.Tensor:
    """A chain of ``k`` values that keeps its value with probability ``phi``,
    else draws it anew uniformly: the centred one-hot's autocorrelation is
    $\\phi^\\ell$."""
    keep = torch.rand(n, generator=g) < phi
    fresh = torch.randint(k, (n,), generator=g)
    out = fresh.clone()
    for i in range(1, n):
        if keep[i]:
            out[i] = out[i - 1]
    return out


class TestReading:
    def test_above_and_below_need_every_video_on_their_side(self):
        assert reading(0.1, 0.05, [0.1] * 9) == ["above"]
        assert reading(0.1, 0.05, [0.1] * 8 + [-0.01]) == ["undecided"]
        assert reading(0.04, 0.05, [0.1] * 9) == ["undecided"]
        assert reading(-0.1, 0.05, [-0.1] * 9) == ["below", "excl"]
        assert reading(0.1, 0.05, None) == ["undecided"]

    def test_excl_needs_every_video_below_delta_star(self):
        # d + delta < Delta* on all 9 videos, and each 8 of them below Delta*
        assert reading(0.01, 0.01, [0.0] * 9) == ["excl"]
        assert reading(0.01, 0.01, [0.0] * 8 + [DELTA_STAR]) == ["undecided"]
        assert reading(0.01, 0.02, [0.0] * 9) == ["undecided"]  # 0.03, not below
        assert reading(0.01, 0.01, None) == ["undecided"]


class TestCompare:
    def test_a_code_against_itself_excludes_a_delta_star_gain(self):
        g = torch.Generator().manual_seed(0)
        video = torch.arange(600) // 200
        t = torch.randint(3, (600,), generator=g)
        sc = scene_of(t, video, 3, 10)
        pred = torch.where(torch.rand(600, generator=g) < 0.5, t, 0)
        jack = pred.repeat(3, 1)
        for u in range(3):
            jack[u, video == u] = -1
        row = {"m": scene_m(pred, t, video, 3), "pred": pred.tolist(),
               "jack_pred": jack.tolist()}  # fmt: skip
        c = compare(sc, row, row)
        assert c["d"] == 0.0 and c["se"] == 0.0 and c["d_jack"] == [0.0] * 3
        assert c["read"] == ["excl"]
        alone = compare(sc, row, None)
        assert alone["d"] == row["m"] and alone["se"] > 0


def item(number: int, gid: int, quarter: str | None = None) -> dict:
    return {"number": number, "teacher_id": gid, "teacher": f"g{gid}",
            "quarter": quarter}  # fmt: skip


class TestAudit:
    def test_the_shares_by_hand(self):
        key = [item(i, 3) for i in range(1, 11)] + [item(i, 5) for i in range(11, 21)]
        answers = {str(i): [3] for i in range(1, 9)}  # 8 of 10 hold 3
        answers |= {"9": [5, 3], "10": UNSURE_LABEL}  # 9 of 9 judged
        answers |= {str(i): [5] for i in range(11, 18)}  # 7 of 10 hold 5
        answers |= {str(i): [3] for i in range(18, 21)}
        got = audit(key, answers, quarter=False)
        assert got["agree"] == {"3": 1.0, "5": 0.7}
        assert got["unsure"] == pytest.approx(1 / 20)
        assert got["pass"] is True
        answers["17"] = [3]  # 5: 6 of 10
        assert audit(key, answers, quarter=False)["pass"] is False

    def test_a_group_of_none_but_the_unsure_fails(self):
        key = [item(i, 3) for i in range(1, 31)] + [item(i, 5) for i in range(31, 35)]
        answers = {str(i): [3] for i in range(1, 31)}
        answers |= {str(i): UNSURE_LABEL for i in range(31, 35)}  # 4 of 34: < 0.2
        got = audit(key, answers, quarter=False)
        assert got["agree"]["5"] is None and got["judged"]["5"] == 0
        assert got["pass"] is False

    def test_an_answer_not_of_the_ids_leaves_it_undecided(self):
        key = [item(1, 3), item(2, 3)]
        for bad in ("3", "g3", [], [3, 5, 7], [True], None):
            answers = {"1": [3], "2": bad} if bad is not None else {"1": [3]}
            got = audit(key, answers, quarter=False)
            assert got["pass"] is None and got["bad"] == [2]

    def test_the_quarters_apart(self):
        key = [item(1, 3), item(2, 5, "tl")]
        answers = {"1": [3], "2": [3]}
        assert audit(key, answers, quarter=False)["pass"] is True
        assert audit(key, answers, quarter=True)["pass"] is False


class TestSeedVar:
    def test_by_hand(self):
        pairs = [({"m": {"a": 1.0}}, {"m": {"a": 0.0}}),
                 ({"m": {"a": 2.0}}, {"m": {"a": 5.0}}),
                 ({"m": {}}, {"m": {"a": 1.0}}), (None, None)]  # fmt: skip
        var, dof = seed_var(pairs, lambda m: m.get("a"))
        assert dof == 2 and var == pytest.approx((1 / 2 + 9 / 2) / 2)
        assert seed_var([], lambda m: m.get("a")) == (None, 0)


class TestNullCheck:
    @staticmethod
    def setting(n: int = 1500, videos: int = 3, k: int = 3):
        g = torch.Generator().manual_seed(1)
        video = torch.arange(n) * videos // n
        share = torch.tensor([0.5, 0.3, 0.2])
        t = torch.multinomial(share, n, replacement=True, generator=g)
        return g, video, t, share

    def test_the_draws_spread_by_their_own_bootstrap(self):
        # labels independent of everything, slow predictions: m's sd over
        # the draws is the pooled bootstrap SE* of the draws
        g, video, t, share = self.setting()
        sc = scene_of(t, video, 3, 16)
        rows = []
        for _ in range(N_RAND):
            r = torch.multinomial(share, len(t), replacement=True, generator=g)
            pred = markov(len(t), 0.95, 3, g)
            rows.append({"m": scene_m(pred, r, video, 3), "pred": pred.tolist(),
                         "t": r.tolist()})  # fmt: skip
        got = null_check(sc, rows)
        assert got is not None and got["pass"] is True
        assert 0.6 <= got["sd"] / got["se"] <= 1.6
        assert got["se"] == pytest.approx(
            math.sqrt(sum(x * x for x in got["se_rows"]) / N_RAND)
        )

    def test_constant_predictions_leave_the_size_unchecked(self):
        g, video, t, share = self.setting()
        sc = scene_of(t, video, 3, 16)
        rows = []
        for _ in range(N_RAND):
            r = torch.multinomial(share, len(t), replacement=True, generator=g)
            pred = torch.randint(3, (3,), generator=g)[video]
            rows.append({"m": scene_m(pred, r, video, 3), "pred": pred.tolist(),
                         "t": r.tolist()})  # fmt: skip
        got = null_check(sc, rows)
        assert got is not None and got["se"] == 0.0 and got["pass"] is None

    def test_a_row_missing_or_without_its_labels(self):
        _, video, t, _ = self.setting(150)
        sc = scene_of(t, video, 3, 16)
        row = {"m": 0.0, "pred": t.tolist(), "t": t.tolist()}
        assert null_check(sc, [row] * (N_RAND - 1)) is None
        assert null_check(sc, [row] * (N_RAND - 1) + [{"m": 0.0}]) is None


class TestHitTau:
    def test_a_markov_chain_of_hits(self):
        g = torch.Generator().manual_seed(2)
        n, phi = 30000, 0.8
        video = torch.arange(n) * 3 // n
        t = torch.randint(3, (n,), generator=g)
        hit = markov(n, phi, 2, g).bool()
        pred = torch.where(hit, t, (t + 1) % 3)
        sc = scene_of(t, video, 3, 16)
        got = hit_tau(sc, {"pred": pred.tolist()})
        assert got == pytest.approx((1 + phi) / (1 - phi), rel=0.1)
        assert hit_tau(sc, {"pred": t.tolist()}) is None  # every one a hit


class TestE1fPremise:
    def test_the_main_checks_it_shares_and_its_own(self):
        ok = {"pass": True}
        v = {"V-c0": ok, "V-c3": ok, "V-c4": ok, "V-E1'5": True, "V-E1'6": ok,
             "V-c1": {"pass": False}, "V-c6": {"pass": False}}  # fmt: skip
        assert e1f_premise(v) is True  # V-c1, V-c6 are not its premises
        assert e1f_premise(v | {"V-c4": {"pass": None}}) is False
        assert e1f_premise(v | {"V-E1'5": False}) is False
