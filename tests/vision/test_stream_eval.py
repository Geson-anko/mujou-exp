"""Tests of exp.vision.stream_eval on tiny synthetic Walking Tours (16 x 16
frames, stacks of 2; videos A, B, C of 20 frames in a loop of 60, H held
out): the oracles are the formulas of the module docstring."""

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.vision.compose import to_dict
from exp.vision.eval import stacks_at
from exp.vision.schema import DataConf, MeasureConf, ModelConf, RunConf, ToursConf
from exp.vision.stream_eval import (
    StreamEvaluator,
    evaluate_stream,
    groups,
    prequential,
    stream_positions,
)
from tests.vision.test_eval import SIZE, ColorCode, MixCode, N, random_tours

DURATION = 3.0  # 30 frames of 0.1 s
# 0.5 s past, 1 s future over 2 s, pairs every 0.1 s in blocks of 0.5 s
EVERY = {
    "past_every": 0.5,
    "future_every": 1.0,
    "future": 2.0,
    "probe_every": 0.1,
    "block": 0.5,
}
FIT = 3


def conf_of(seeds: int = 2) -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=N)),
        model=ModelConf(latent=4),
        measure=MeasureConf(
            duration=DURATION,
            ckpt_times=[0.0, DURATION],
            clips=3,
            windows=1,
            posthoc_seeds=seeds,
        ),
    )


@pytest.fixture
def root(tmp_path) -> Path:
    return random_tours(tmp_path / "d")


def evaluator(root: Path, run: Path, seeds: int = 2) -> StreamEvaluator:
    run.mkdir(exist_ok=True)
    return StreamEvaluator(
        conf_of(seeds), run, root, run / "out", fit_steps=FIT, **EVERY
    )


def brute_mean_stack(root: Path) -> torch.Tensor:
    """The mean of the kept stacks every 0.5 s of the span, one at a time."""
    _, x = stacks_at(conf_of(), root, range(0, 30, 5))
    return x.double().mean(0) / 255


class TestPositions:
    def test_every_past_every_over_the_span_then_future_every_after(self):
        conf = conf_of()
        assert stream_positions(conf, past_every=0.5, future_every=1.0, future=2.0) == [
            0,
            5,
            10,
            15,
            20,
            25,
            30,
            40,
        ]

    def test_the_units_marked_not_ready_or_reset_are_left_out(self, root, tmp_path):
        # k = 0, 20, 40 end at the first frame of A, B, C
        assert evaluator(root, tmp_path / "run").ks == [5, 10, 15, 25, 30]

    def test_groups_by_age_and_lead_by_hand(self):
        ks = [0, 50, 89, 90, 95, 99, 100, 105, 109, 110, 170, 500]
        g = groups(ks, 100, 1.0)
        at = {name: [ks[i] for i in idx] for name, idx in g.items()}
        assert at["past_300_"] == []
        assert at["past_60_300"] == [0]
        assert at["past_10_60"] == [50, 89]
        assert at["now"] == [90, 95, 99]  # ages 10, 5, 1
        assert at["next_0_10"] == [100, 105, 109]  # leads 0, 5, 9
        assert at["next_10_60"] == [110]
        assert at["next_60_300"] == [170]
        assert at["next_300_"] == [500]
        assert at["past"] == [0, 50, 89, 90, 95, 99]
        assert at["future"] == [100, 105, 109, 110, 170, 500]


class TestMatrix:
    @pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
    def test_the_stack_mixed_with_the_mean_stack_by_c_gives_c_squared(
        self, root, tmp_path, c
    ):
        ev = evaluator(root, tmp_path / "run")
        code = MixCode(brute_mean_stack(root))
        code.net.c.data.fill_(c)
        err = ev.errors(code)
        assert len(err) == len(ev.ks)
        row = ev.summary(err, frame=12)
        for name in ("now", "past", "next_0_10", "future"):
            assert row[name] == pytest.approx(c * c, rel=1e-5, abs=1e-12), name
        # at frame 12: past 5, 10 (ages 0.7, 0.2 s: now); future 15 .. 30
        assert row["n_past"] == row["n_now"] == 2 and row["n_future"] == 3
        assert math.isnan(row["past_10_60"]) and row["n_past_10_60"] == 0

    def test_the_base_is_the_squared_distance_to_the_mean_stack(self, root, tmp_path):
        ev = evaluator(root, tmp_path / "run")
        m = brute_mean_stack(root)
        _, x = stacks_at(conf_of(), root, ev.ks)
        want = ((x.double() / 255 - m) ** 2).flatten(1).sum(1)
        torch.testing.assert_close(
            torch.tensor(ev.base).double(), want, rtol=1e-5, atol=0
        )


class TestPrequential:
    def test_the_second_that_holds_each_unit_in_units_of_x_over_255(self, tmp_path):
        rows = [{"t": 1.0, "sq": 1.0}, {"t": 2.0, "sq": 2.0}, {"t": 3.0, "sq": None}]
        (tmp_path / "stream.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        p = N * 3 * SIZE * SIZE
        got = prequential(tmp_path, conf_of(), [0, 9, 10, 25, 30])

        def conv(sq: float) -> float:
            return (256 / 255) ** 2 * sq - p / (12 * 255**2)

        assert got == [conv(1.0), conv(1.0), conv(2.0), None, None]

    def test_pre_of_a_group_is_the_sum_over_the_sum_of_the_base(self, root, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        rows = [{"t": float(s), "sq": 10.0 * s} for s in (1, 2, 3)]
        (run / "stream.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        ev = evaluator(root, run)
        row = ev.summary([0.0] * len(ev.ks), frame=30)
        past = [i for i, k in enumerate(ev.ks) if k < 30]
        pre = [ev.pre[i] for i in past]
        assert all(v is not None for v in pre)
        want = sum(v for v in pre if v is not None) / sum(ev.base[i] for i in past)
        assert row["pre_past"] == pytest.approx(want, rel=1e-12)
        assert row["past"] == 0.0


class TestProbes:
    def test_span_after_and_zurich_with_every_seed(self, root, tmp_path):
        ev = evaluator(root, tmp_path / "run")
        rows = ev.probes(ColorCode(), "x")
        assert [r["group"] for r in rows] == ["span", "after", "zurich"]
        # span: even blocks of 0.5 s, every 0.1 s, a pair (k, k + 1) both kept
        # (k + 1 = 20 is not); after: k = 30, 40, but 40 is not kept
        assert rows[0]["pairs"] == len(
            [k for k in range(29) if k // 5 % 2 == 0 and k % 20 and (k + 1) % 20]
        )
        assert rows[1]["pairs"] == 1
        for r in (rows[0], rows[2]):
            assert len(r["recon_blur_k"]) == 2 and math.isfinite(r["recon_blur"])
            assert math.isfinite(r["dyn_1/copy_1"])
        assert "recon_blur" not in rows[1]  # one pair: no spread to judge
        assert (tmp_path / "run" / "out" / "x-span.png").exists()

    def test_the_same_code_gives_the_same_rows(self, root, tmp_path):
        ev = evaluator(root, tmp_path / "run")
        assert ev.probes(ColorCode(), "a") == ev.probes(ColorCode(), "b")


class TestEvaluateStream:
    def test_every_checkpoint_in_order_probes_at_probe_at_only(self, root, tmp_path):
        conf = conf_of()
        run = tmp_path / "run"
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        m = brute_mean_stack(root)
        saved = MixCode(m)
        for t, c in ((DURATION, 0.0), (0.0, 1.0)):  # saved out of order
            saved.net.c.data.fill_(c)
            ckpt.save_eval(
                run,
                program="vae",
                group="g",
                t=t,
                frame=round(t / 0.1),
                modules=saved.modules(),
            )
        out = evaluate_stream(
            run, lambda _: MixCode(m), root=root, fit_steps=FIT, spacing=EVERY
        )

        def jsonl(name: str) -> list[dict]:
            return [json.loads(x) for x in (out / name).read_text().splitlines()]

        matrix, summary, probes = (
            jsonl(f) for f in ("matrix.jsonl", "summary.jsonl", "probes.jsonl")
        )
        assert [(r["t"], r["frame"]) for r in summary] == [(0.0, 0), (DURATION, 30)]
        positions = json.loads((out / "positions.json").read_text())
        assert all(len(r["err"]) == len(positions["k"]) for r in matrix)
        assert summary[0]["future"] == pytest.approx(1.0, rel=1e-5)  # c = 1
        assert summary[1]["past"] == 0.0 and summary[1]["future"] == 0.0  # c = 0
        assert summary[0]["span"] == pytest.approx(1.0, rel=1e-5)
        assert summary[1]["span"] == summary[1]["after"] == 0.0
        assert summary[1]["n_span"] == summary[1]["n_past"]  # all of it, at the end
        assert {r["t"] for r in probes} == {0.0}  # 3.0 is not in PROBE_AT
        assert [r["group"] for r in probes] == ["span", "after", "zurich"]
