"""Tests of exp.vision.z_use on the tiny synthetic Walking Tours of
test_stream_eval: the oracle is ``MixCode``, whose reconstruction is its code
(the stack) at $c = 0$ and the mean stack whatever the code at $c = 1$."""

import json

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.vision.compose import to_dict
from exp.vision.eval import stacks_at
from exp.vision.z_use import ZUse, evaluate_z_use, shifted
from tests.vision.test_eval import MixCode, random_tours
from tests.vision.test_stream_eval import (
    DURATION,
    brute_mean_stack,
    chunk_conf,
    conf_of,
    write_run,
)

EVERY = 0.5  # span positions 0, 5, ..., 25; kept 5, 10, 15, 25


@pytest.fixture
def root(tmp_path):
    return random_tours(tmp_path / "d")


def test_shifted_takes_the_code_of_the_stack_half_the_span_away():
    assert shifted(4) == [2, 3, 0, 1]
    assert shifted(5) == [2, 3, 4, 0, 1]


class TestRow:
    def test_a_decoder_of_its_code_by_hand(self, root):
        """$c = 0$: the true code reconstructs exactly (0); the shifted one
        gives $\\sum \\lVert x_i - x_{\\pi(i)}\\rVert^2 / \\sum \\lVert x_i -
        m\\rVert^2$; the mean code decodes to the mean stack $m$ (1)."""
        z = ZUse(conf_of(), root, every=EVERY)
        assert z.ks == [5, 10, 15, 25]
        m = brute_mean_stack(root)
        code = MixCode(m)
        code.net.c.data.fill_(0.0)
        row = z.row(code)
        _, x = stacks_at(conf_of(), root, z.ks)
        x = x.double() / 255
        base = ((x - m) ** 2).flatten(1).sum(1)
        moved = ((x - x[shifted(4)]) ** 2).flatten(1).sum(1)
        assert row["n"] == 4
        assert row["span"] == pytest.approx(0.0, abs=1e-12)
        assert row["span_shifted"] == pytest.approx(
            float(moved.sum() / base.sum()), rel=1e-5
        )
        assert row["span_mean_z"] == pytest.approx(1.0, rel=1e-5)

    def test_a_decoder_that_ignores_its_code_is_the_same_for_every_code(self, root):
        z = ZUse(conf_of(), root, every=EVERY)
        code = MixCode(brute_mean_stack(root))
        code.net.c.data.fill_(1.0)
        row = z.row(code)
        for k in ("span", "span_shifted", "span_mean_z"):
            assert row[k] == pytest.approx(1.0, rel=1e-5), k


def test_evaluate_writes_a_row_per_chosen_checkpoint(root, tmp_path):
    conf = conf_of()
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
    m = brute_mean_stack(root)
    saved = MixCode(m)
    for t, c in ((DURATION, 0.0), (0.0, 1.0), (1.0, 0.5)):
        saved.net.c.data.fill_(c)
        ckpt.save_eval(
            run,
            program="vae",
            group="g",
            t=t,
            frame=round(t / 0.1),
            modules=saved.modules(),
        )
    path = evaluate_z_use(
        run, lambda _: MixCode(m), root=root, every=EVERY, at=(0.0, DURATION)
    )
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert path == run / "posthoc_stream" / "z_use.jsonl"
    assert [(r["t"], r["frame"]) for r in rows] == [(0.0, 0), (DURATION, 30)]
    assert rows[0]["span_shifted"] == pytest.approx(1.0, rel=1e-5)  # c = 1
    assert rows[1]["span"] == 0.0 and rows[1]["span_shifted"] > 0.0  # c = 0
    assert torch.isfinite(torch.tensor(rows[1]["span_mean_z"]))


def test_a_chunk_run_gives_the_rows_of_its_single_stream(root, tmp_path):
    """The oracle is the run of one stream (``eval.stream_view``)."""
    m = brute_mean_stack(root)
    write_run(tmp_path / "one", conf_of(), m)
    write_run(tmp_path / "chunk", chunk_conf(), m, c=2)
    rows = [
        evaluate_z_use(
            tmp_path / name,
            lambda _: MixCode(m),
            root=root,
            every=EVERY,
            at=(0.0, 1.0, DURATION),
        ).read_text()
        for name in ("one", "chunk")
    ]
    assert rows[0] == rows[1]
    assert [json.loads(x)["t"] for x in rows[1].splitlines()] == [0.0, 1.0, DURATION]


def test_span_takes_the_positions_of_that_span(root, tmp_path):
    m = brute_mean_stack(root)
    write_run(tmp_path / "run", conf_of(), m)
    path = evaluate_z_use(
        tmp_path / "run", lambda _: MixCode(m), root=root, every=EVERY, span=2.0
    )
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    # span positions 0, 5, 10, 15 below 2 s; kept 5, 10, 15
    assert {r["n"] for r in rows} == {3}
