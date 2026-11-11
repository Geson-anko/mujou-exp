import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from exp.lib.runs import export, index, main, series, table, where

REPO = Path(__file__).resolve().parents[2]
SNAP = "a" * 40
MUJOU = "b" * 40


def make_job(
    *,
    d_s: int = 64,
    kind: str = "pc",
    seed: int = 0,
    batch: str = "b1",
    snap: str = SNAP,
    mujou: str = MUJOU,
    dirty: bool = False,
    res: str = "cpu",
    tags: tuple[str, ...] = ("confirm",),
) -> dict[str, Any]:
    run_id = f"m14-s1-{kind}{d_s}-s{seed}"
    return {
        "schema": 1,
        "run_id": run_id,
        "batch": batch,
        "group": f"{kind}{d_s}".ljust(12, "0"),
        "program": "dynamics",
        "seed": seed,
        "overrides": ["+experiment=m14/s1", f"learner={kind}", f"seed={seed}"],
        "config": {
            "seed": seed,
            "device": "cpu",
            "learner": {"kind": kind},
            "dynamics": {"d_s": d_s, "trace": True},
            "optim": {"lr": 0.02},
        },
        "code": {"snap": snap, "base_head": snap, "dirty": dirty, "mujou_snap": mujou},
        "res": {"class": res, "ram_gb": 2.0, "gpu_mem_gb": 0.0},
        "tags": list(tags),
        "audit_of": None,
    }


def write_run(
    runs: Path,
    job: dict[str, Any],
    *,
    status: str = "completed",
    final: dict[str, Any] | None = None,
    evals: list[dict[str, Any]] | None = None,
    manifest: bool = True,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """A run directory of the contract (job, evals, result, manifest, provenance)."""
    d = runs / job["batch"] / job["run_id"]
    d.mkdir(parents=True)
    (d / "job.json").write_text(json.dumps(job))
    lines = evals if evals is not None else []
    (d / "evals.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    if status in ("completed", "diverged"):
        result = {"schema": 1, "status": status, "reason": None, "final": final or {}}
        (d / "result.json").write_text(json.dumps(result))
    if manifest:
        (d / "manifest.json").write_text("{}")
    (d / "provenance").mkdir()
    prov = provenance if provenance is not None else {"device": "cpu"}
    (d / "provenance" / "attempt-1.json").write_text(json.dumps(prov))
    return d


def write_done(nas: Path, job: dict[str, Any], status: str) -> None:
    """The fleet's ``done/`` entry of a run that left no directory."""
    (nas / "done").mkdir(exist_ok=True)
    entry = job | {"status": status, "host": "std-00", "attempt": 1}
    (nas / "done" / f"{job['run_id']}.json").write_text(json.dumps(entry))


def final_of(value: float) -> dict[str, Any]:
    return {"test": {"roll10/cv10": value}, "val": {"roll10/cv10": value + 1}}


def grid(root: Path, values: Mapping[tuple[int, str, int], float | str]) -> None:
    """Runs of (d_s, kind, seed) -> a value of test roll10/cv10, or a status.

    ``root`` is a NAS root: a diverged run has a directory, a pruned or failed
    one only a ``done/`` entry."""
    for (d_s, kind, seed), v in values.items():
        job = make_job(d_s=d_s, kind=kind, seed=seed)
        if v == "diverged":
            write_run(root / "runs", job, status="diverged")
        elif isinstance(v, str):
            write_done(root, job, v)
        else:
            write_run(root / "runs", job, final=final_of(v))


FULL = {
    (d_s, kind, seed): round(d_s / 1000 + seed / 100 + (0.5 if kind == "pc" else 0), 4)
    for d_s in (64, 256)
    for kind in ("bp", "pc")
    for seed in (0, 1, 2)
}


def body(text: str) -> list[str]:
    return text.split("\n\n")[0].splitlines()


def notes(text: str) -> str:
    return text.split("\n\n")[1]


class TestIndex:
    def test_reads_every_run_sorted_and_filters_by_batch(self, tmp_path):
        write_run(tmp_path, make_job(seed=1))
        write_run(tmp_path, make_job(seed=0))
        write_run(tmp_path, make_job(seed=0, batch="b2"))
        assert [r.run_id for r in index(tmp_path)] == [
            "m14-s1-pc64-s0",
            "m14-s1-pc64-s1",
            "m14-s1-pc64-s0",
        ]
        assert [r.batch for r in index(tmp_path)] == ["b1", "b1", "b2"]
        assert [r.seed for r in index(tmp_path, batch="b1")] == [0, 1]

    def test_status_from_result_only_with_a_manifest(self, tmp_path):
        write_run(tmp_path, make_job(seed=0), final=final_of(0.1))
        write_run(tmp_path, make_job(seed=1), status="diverged")
        write_run(tmp_path, make_job(seed=2), final=final_of(0.1), manifest=False)
        assert [r.status for r in index(tmp_path)] == [
            "completed",
            "diverged",
            "running",
        ]

    def test_the_nas_root_adds_done_claimed_and_queued(self, tmp_path):
        runs = tmp_path / "runs"
        write_run(runs, make_job(seed=0), final=final_of(0.1))
        for sub in ("done", "queue", "claimed/std-01"):
            (tmp_path / sub).mkdir(parents=True)
        pruned = make_job(seed=1) | {"status": "pruned", "exit": 75}
        (tmp_path / "done" / "m14-s1-pc64-s1.json").write_text(json.dumps(pruned))
        (tmp_path / "claimed/std-01/050-1-m14-s1-pc64-s2.json").write_text(
            json.dumps(make_job(seed=2))
        )
        (tmp_path / "claimed/std-01/m14-s1-pc64-s2.stop").write_text("")
        (tmp_path / "queue/050-2-m14-s1-pc64-s3.json").write_text(
            json.dumps(make_job(seed=3))
        )
        assert [(r.seed, r.status) for r in index(tmp_path)] == [
            (0, "completed"),
            (1, "pruned"),
            (2, "running"),
            (3, "queued"),
        ]

    def test_a_requeued_run_is_queued_and_a_claim_moving_to_done_is_done(
        self, tmp_path
    ):
        runs = tmp_path / "runs"
        write_run(runs, make_job(seed=0), final=final_of(0.1))
        for sub in ("done", "queue", "claimed/std-00"):
            (tmp_path / sub).mkdir(parents=True)
        (tmp_path / "queue/050-1-x.json").write_text(json.dumps(make_job(seed=0)))
        job1 = make_job(seed=1)
        (tmp_path / "done/x.json").write_text(json.dumps(job1 | {"status": "failed"}))
        (tmp_path / "claimed/std-00/x.json").write_text(json.dumps(job1))
        assert [(r.seed, r.status) for r in index(tmp_path)] == [
            (0, "queued"),
            (1, "failed"),
        ]

    @pytest.mark.parametrize("status", ["corrupt", "failed"])
    def test_done_overrides_a_directory_with_its_manifest(self, tmp_path, status):
        # a copy that fails its sha256 check keeps its directory and manifest
        write_run(tmp_path / "runs", make_job(seed=0), final=final_of(0.1))
        write_done(tmp_path, make_job(seed=0), status)
        assert [r.status for r in index(tmp_path)] == [status]

    def test_done_says_completed_but_no_manifest_is_incomplete(self, tmp_path):
        write_run(
            tmp_path / "runs", make_job(seed=0), final=final_of(0.1), manifest=False
        )
        (tmp_path / "done").mkdir()
        entry = make_job(seed=0) | {"status": "completed"}
        (tmp_path / "done/m14-s1-pc64-s0.json").write_text(json.dumps(entry))
        assert [r.status for r in index(tmp_path)] == ["incomplete"]

    def test_device_class_from_provenance_else_the_job(self, tmp_path):
        write_run(tmp_path, make_job(seed=0), provenance={"device": "cpu"})
        write_run(
            tmp_path,
            make_job(seed=1, res="any_gpu"),
            provenance={"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 4090"},
        )
        write_run(tmp_path, make_job(seed=2), provenance={"device_class": "4060"})
        write_run(tmp_path, make_job(seed=3, res="any_gpu"), provenance={})
        assert [r.device_class for r in index(tmp_path)] == [
            "cpu",
            "4090",
            "4060",
            "any_gpu",
        ]

    def test_the_last_attempt_names_the_device(self, tmp_path):
        d = write_run(tmp_path, make_job(), provenance={"device": "cpu"})
        (d / "provenance" / "attempt-10.json").write_text(
            json.dumps({"device_class": "4090"})
        )
        (d / "provenance" / "attempt-2.json").write_text(
            json.dumps({"device_class": "4060"})
        )
        assert index(tmp_path)[0].device_class == "4090"


class TestRecord:
    EVALS = [
        {"t": 300.0, "split": "val", "roll10": 0.3},
        {"t": 300.0, "split": "test", "roll10": 0.35},
        {"t": 600.0, "split": "val", "roll10": 0.2},
        {"t": 600.0, "split": "test", "roll10": 0.25},
    ]

    def test_evals_by_split_in_order(self, tmp_path):
        write_run(tmp_path, make_job(), evals=self.EVALS)
        (r,) = index(tmp_path)
        assert [x["roll10"] for x in r.evals()] == [0.3, 0.2]
        assert [x["roll10"] for x in r.evals("test")] == [0.35, 0.25]

    def test_at_last_is_the_final_of_the_result_else_the_last_eval(self, tmp_path):
        write_run(
            tmp_path,
            make_job(seed=0),
            evals=self.EVALS,
            final={"test": {"roll10": 0.9}},
        )
        write_run(
            tmp_path,
            make_job(seed=1),
            evals=self.EVALS,
            manifest=False,
            status="running",
        )
        r0, r1 = index(tmp_path)
        assert r0.at() == {"roll10": 0.9}
        assert r0.at("last", "val") == self.EVALS[2]
        assert r1.at() == self.EVALS[3]

    def test_at_a_time(self, tmp_path):
        write_run(tmp_path, make_job(), evals=self.EVALS)
        (r,) = index(tmp_path)
        assert r.at("t=300", "val") == self.EVALS[0]
        assert r.at("t=600.0", "test") == self.EVALS[3]
        assert r.at("t=900", "test") is None
        with pytest.raises(ValueError, match="t=<seconds>"):
            r.at("first")

    def test_a_live_copy_cut_mid_line_reads_the_whole_lines(self, tmp_path):
        d = write_run(tmp_path, make_job(), evals=self.EVALS[:2], manifest=False)
        with (d / "evals.jsonl").open("a") as f:
            f.write('{"t": 600.0, "spl')
        (r,) = index(tmp_path)
        assert [x["t"] for x in r.evals("val") + r.evals("test")] == [300.0, 300.0]

    def test_series(self, tmp_path):
        evals = [*self.EVALS, {"t": 900.0, "split": "val"}]
        write_run(tmp_path, make_job(), evals=evals)
        (r,) = index(tmp_path)
        assert series(r, "roll10") == [(300.0, 0.3), (600.0, 0.2)]
        assert series(r, "roll10", "test") == [(300.0, 0.35), (600.0, 0.25)]
        assert series(r, "roll20") == []


class TestWhere:
    @pytest.fixture
    def runs(self, tmp_path):
        write_run(tmp_path, make_job(kind="pc", d_s=64, tags=("confirm", "big")))
        write_run(tmp_path, make_job(kind="bp", d_s=256, tags=("explore",)))
        write_run(tmp_path, make_job(kind="bp", d_s=64, dirty=True), status="diverged")
        return index(tmp_path)

    def ids(self, runs):
        return [r.run_id.split("-")[2] for r in runs]

    def test_equality_on_strings_numbers_and_booleans(self, runs):
        assert self.ids(where(runs, "config.learner.kind=bp")) == ["bp256", "bp64"]
        assert self.ids(where(runs, "config.dynamics.d_s=64")) == ["bp64", "pc64"]
        assert self.ids(where(runs, "config.dynamics.d_s=64.0")) == ["bp64", "pc64"]
        assert self.ids(where(runs, "config.optim.lr=0.020")) == [
            "bp256",
            "bp64",
            "pc64",
        ]
        assert self.ids(where(runs, "code.dirty=true")) == ["bp64"]

    def test_conditions_are_all_met(self, runs):
        assert self.ids(
            where(runs, "config.learner.kind=bp", "config.dynamics.d_s!=64")
        ) == ["bp256"]

    def test_membership_and_status(self, runs):
        assert self.ids(where(runs, "tags~=confirm")) == ["bp64", "pc64"]
        assert self.ids(where(runs, "tags~=big")) == ["pc64"]
        assert self.ids(where(runs, "run_id~=pc")) == ["pc64"]
        assert self.ids(where(runs, "status=diverged")) == ["bp64"]
        assert self.ids(where(runs, "result.status=completed")) == ["bp256", "pc64"]

    def test_an_absent_path_fails_equality_and_passes_inequality(self, runs):
        assert where(runs, "config.vision.kind=vit") == []
        assert len(where(runs, "config.vision.kind!=vit")) == 3

    def test_a_malformed_condition_raises(self, runs):
        with pytest.raises(ValueError, match="path=value"):
            where(runs, "config.learner.kind")


ROWS = ["config.dynamics.d_s"]
COLS = ["config.learner.kind"]


class TestTable:
    def test_the_markdown_lists_the_seeds_in_numeric_order_of_rows(self, tmp_path):
        grid(tmp_path, FULL)
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text) == [
            "| dynamics.d_s | learner.kind=bp | learner.kind=pc |",
            "| --- | --- | --- |",
            "| 64 | 0.064 / 0.074 / 0.084 | 0.56 / 0.57 / 0.58 |",
            "| 256 | 0.26 / 0.27 / 0.28 | 0.76 / 0.77 / 0.78 |",
        ]

    def test_split_and_digits(self, tmp_path):
        grid(tmp_path, FULL)
        text = table(
            index(tmp_path),
            rows=ROWS,
            cols=COLS,
            value="roll10/cv10",
            split="val",
            digits=3,
        )
        assert body(text)[2] == "| 64 | 1.06 / 1.07 / 1.08 | 1.56 / 1.57 / 1.58 |"

    @pytest.mark.parametrize(
        ("v", "digits", "shown"),
        [
            (0.004, 2, "0.0040"),  # a raw error: small, not rounded to 0
            (0.62, 2, "0.62"),
            (12.345, 2, "12"),
            (1234.5, 2, "1234"),  # never in exponent notation
            (0.0996, 2, "0.10"),  # rounding up gains a digit before the point
            (0.0, 2, "0.0"),
            (-0.0123, 3, "-0.0123"),
            (float("nan"), 2, "nan"),
        ],
    )
    def test_digits_are_significant_figures(self, tmp_path, v, digits, shown):
        write_run(tmp_path, make_job(), final={"test": {"roll1": v}})
        text = table(index(tmp_path), rows=ROWS, cols=[], value="roll1", digits=digits)
        assert body(text)[2] == f"| 64 | {shown} |"

    def test_diverged_pruned_failed_and_missing_seeds_are_marked_not_averaged(
        self, tmp_path
    ):
        values: dict[tuple[int, str, int], float | str] = dict(FULL)
        values[(64, "pc", 1)] = "diverged"
        values[(64, "bp", 2)] = "failed"
        values[(256, "pc", 2)] = "pruned"
        del values[(256, "bp", 0)]
        grid(tmp_path, values)
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2:] == [
            "| 64 | 0.064 / 0.074 / 欠 | 0.56 / 発散 / 0.58 |",
            "| 256 | 欠 / 0.27 / 0.28 | 0.76 / 0.77 / 打切 |",
        ]

    def test_a_seed_missing_from_a_cell_is_marked_by_the_seeds_of_the_table(
        self, tmp_path
    ):
        grid(tmp_path, {(64, "bp", 0): 0.1, (64, "bp", 2): 0.3, (64, "pc", 1): 0.2})
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2] == "| 64 | 0.10 / 欠 / 0.30 | 欠 / 0.20 / 欠 |"

    def test_a_cell_without_runs_and_a_dirty_cell(self, tmp_path):
        grid(tmp_path, {(64, "bp", 0): 0.1, (256, "pc", 0): 0.2})
        write_run(
            tmp_path / "runs",
            make_job(d_s=64, kind="pc", dirty=True),
            final=final_of(0.3),
        )
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2:] == ["| 64 | 0.10 | 0.30 * |", "| 256 | — | 0.20 |"]
        assert "`*`: 汚れた snapshot" in notes(text)

    def test_two_row_paths_and_no_cols(self, tmp_path):
        grid(tmp_path, {(64, "bp", 0): 0.1, (64, "pc", 0): 0.2})
        text = table(
            index(tmp_path),
            rows=["config.dynamics.d_s", "config.learner.kind"],
            cols=[],
            value="roll10/cv10",
        )
        assert body(text) == [
            "| dynamics.d_s | learner.kind | roll10/cv10 |",
            "| --- | --- | --- |",
            "| 64 | bp | 0.10 |",
            "| 64 | pc | 0.20 |",
        ]

    def test_the_footnote(self, tmp_path):
        grid(
            tmp_path,
            {(64, "bp", 0): 0.1, (64, "bp", 1): "diverged", (64, "bp", 2): 0.3},
        )
        text = table(
            index(tmp_path),
            rows=ROWS,
            cols=COLS,
            value="roll10/cv10",
            command="just fleet table x",
        )
        assert notes(text).splitlines() == [
            "- 値: `roll10/cv10`（test、last）。セルの並びは seed 0 / 1 / 2",
            "- batch `b1`",
            f"- snapshot `{SNAP[:12]}`、mujou `{MUJOU[:12]}`、デバイス cpu",
            "- 3 run: completed 2、diverged 1",
            "- コマンド: `just fleet table x`",
        ]

    @pytest.mark.parametrize(
        ("change", "what"),
        [
            ({"snap": "c" * 40}, "snapshots"),
            ({"mujou": "d" * 40}, "mujou versions"),
            ({"res": "any_gpu"}, "device classes"),
        ],
    )
    def test_a_cell_that_mixes_raises_unless_allowed(self, tmp_path, change, what):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        prov = {"device": "cuda:0", "gpu": "RTX 4090"} if "res" in change else None
        write_run(
            tmp_path / "runs",
            make_job(seed=1, **change),
            final=final_of(0.2),
            provenance=prov,
        )
        runs = index(tmp_path)
        with pytest.raises(
            ValueError, match=rf"cell \(dynamics.d_s=64, learner.kind=pc\) mixes {what}"
        ):
            table(runs, rows=ROWS, cols=COLS, value="roll10/cv10")
        text = table(runs, rows=ROWS, cols=COLS, value="roll10/cv10", allow_mixed=True)
        assert body(text)[2] == "| 64 | 0.10 / 0.20 |"
        assert (
            f"混在を許した: cell (dynamics.d_s=64, learner.kind=pc) mixes {what}"
            in notes(text)
        )

    def test_different_snapshots_in_different_cells_are_not_a_mix(self, tmp_path):
        write_run(tmp_path / "runs", make_job(kind="bp"), final=final_of(0.1))
        write_run(
            tmp_path / "runs", make_job(kind="pc", snap="c" * 40), final=final_of(0.2)
        )
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert f"snapshot `{SNAP[:12]}`、`{'c' * 12}`" in notes(text)

    def test_a_failed_run_does_not_count_as_a_mix(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        write_done(tmp_path, make_job(seed=1, snap="c" * 40, res="any_gpu"), "failed")
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2] == "| 64 | 0.10 / 欠 |"
        assert "- 2 run: completed 1、failed 1" in notes(text)

    def test_a_corrupt_copy_is_missing_not_a_value(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1, (64, "pc", 1): 0.2})
        write_done(tmp_path, make_job(seed=1), "corrupt")
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2] == "| 64 | 0.10 / 欠 |"
        assert "- 2 run: completed 1、corrupt 1" in notes(text)

    def test_a_pruned_run_without_provenance_is_not_a_device_mix(self, tmp_path):
        gpu = {"device": "cuda:0", "gpu": "NVIDIA GeForce RTX 4090"}
        write_run(
            tmp_path / "runs",
            make_job(seed=0, res="any_gpu"),
            final=final_of(0.1),
            provenance=gpu,
        )
        write_done(tmp_path, make_job(seed=1, res="any_gpu"), "pruned")
        text = table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")
        assert body(text)[2] == "| 64 | 0.10 / 打切 |"
        assert "デバイス 4090\n" in notes(text)

    def test_an_unfinished_batch_is_refused_unless_allowed(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        write_run(tmp_path / "runs", make_job(seed=1), status="running", manifest=False)
        runs = index(tmp_path)
        with pytest.raises(
            ValueError, match=r"1 run\(s\) have not finished: m14-s1-pc64-s1"
        ):
            table(runs, rows=ROWS, cols=COLS, value="roll10/cv10")
        text = table(
            runs, rows=ROWS, cols=COLS, value="roll10/cv10", allow_unfinished=True
        )
        assert body(text)[2] == "| 64 | 0.10 / 未 |"

    def test_a_done_run_without_its_manifest_is_refused_as_broken(self, tmp_path):
        write_run(tmp_path / "runs", make_job(), final=final_of(0.1), manifest=False)
        write_done(tmp_path, make_job(), "completed")
        with pytest.raises(ValueError, match=r"done but their directory has no"):
            table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10")

    def test_two_runs_of_one_seed_in_a_cell_raise(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1, (256, "pc", 0): 0.2})
        with pytest.raises(ValueError, match="has two runs of seed 0"):
            table(index(tmp_path), rows=[], cols=COLS, value="roll10/cv10")

    def test_a_value_no_run_has_raises(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        with pytest.raises(ValueError, match="no completed run has 'roll10/cv1'"):
            table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv1")
        with pytest.raises(ValueError, match="no runs"):
            table([], rows=ROWS, cols=COLS, value="roll10/cv10")
        with pytest.raises(ValueError, match="significant figures"):
            table(index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10", digits=0)

    def test_at_a_time(self, tmp_path):
        evals = [{"t": 600.0, "split": "test", "roll10/cv10": 0.5}]
        write_run(tmp_path / "runs", make_job(seed=0), evals=evals, final=final_of(0.1))
        write_run(tmp_path / "runs", make_job(seed=1), final=final_of(0.2))
        text = table(
            index(tmp_path), rows=ROWS, cols=COLS, value="roll10/cv10", at="t=600"
        )
        assert body(text)[2] == "| 64 | 0.50 / 欠 |"


class TestExport:
    def test_identity_final_and_series_round_trip_through_json(self, tmp_path):
        evals = [
            {"t": 300.0, "split": "val", "roll10": 0.3, "amp10": 0.9},
            {"t": 600.0, "split": "val", "roll10": 0.2, "amp10": 0.95},
            {"t": 600.0, "split": "test", "roll10": 0.25},
        ]
        write_run(
            tmp_path,
            make_job(),
            evals=evals,
            final={"test": {"roll10": 0.25, "roll1": 0.0}},
        )
        out = json.loads(json.dumps(export(index(tmp_path), ["roll10"])))
        (run,) = out["runs"]
        assert out["keys"] == ["roll10"]
        assert (run["run_id"], run["seed"], run["status"], run["device_class"]) == (
            "m14-s1-pc64-s0",
            0,
            "completed",
            "cpu",
        )
        assert run["config"]["learner"]["kind"] == "pc"
        assert run["code"] == {"snap": SNAP, "mujou_snap": MUJOU, "dirty": False}
        assert run["final"] == {"test": {"roll10": 0.25}}
        assert run["series"] == {
            "test": {"roll10": [[600.0, 0.25]]},
            "val": {"roll10": [[300.0, 0.3], [600.0, 0.2]]},
        }


class TestCli:
    def test_main_prints_the_table_with_the_command(self, tmp_path, capsys):
        grid(tmp_path, FULL)
        argv = [
            "table",
            "--root",
            str(tmp_path),
            "--batch",
            "b1",
            "--where",
            "config.dynamics.d_s=64",
        ]
        argv += [
            "--rows",
            "config.dynamics.d_s",
            "--cols",
            "config.learner.kind",
            "--value",
            "roll10/cv10",
        ]
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert body(out) == [
            "| dynamics.d_s | learner.kind=bp | learner.kind=pc |",
            "| --- | --- | --- |",
            "| 64 | 0.064 / 0.074 / 0.084 | 0.56 / 0.57 / 0.58 |",
        ]
        command = "python -m exp.lib.runs " + " ".join(argv)
        assert notes(out).splitlines()[-1] == f"- コマンド: `{command}`"

    def test_main_shows_the_command_a_forwarding_cli_passes(self, tmp_path, capsys):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        argv = ["table", "--root", str(tmp_path), "--rows", "seed"]
        argv += ["--value", "roll10/cv10"]
        assert main(argv, command="just fleet table --value roll10/cv10") == 0
        last = notes(capsys.readouterr().out).splitlines()[-1]
        assert last == "- コマンド: `just fleet table --value roll10/cv10`"

    @pytest.mark.parametrize("bad", ["where", "json"])
    def test_a_bad_condition_or_file_exits_2(self, tmp_path, capsys, bad):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        argv = ["table", "--root", str(tmp_path), "--rows", "seed", "--value", "x"]
        if bad == "where":
            argv += ["--where", "config.learner.kind"]
        else:
            (tmp_path / "runs/b1/m14-s1-pc64-s0/result.json").write_text("{")
        assert main(argv) == 2
        assert capsys.readouterr().err.startswith("runs table: ")

    def test_main_refuses_with_exit_2(self, tmp_path, capsys):
        write_run(tmp_path / "runs", make_job(), status="running", manifest=False)
        assert (
            main(["table", "--root", str(tmp_path), "--rows", "seed", "--value", "x"])
            == 2
        )
        assert "have not finished" in capsys.readouterr().err

    def test_runs_as_a_module_without_torch_or_hydra(self, tmp_path):
        grid(tmp_path, {(64, "pc", 0): 0.1})
        code = (
            "import sys, runpy; sys.argv = ['runs', *sys.argv[1:]];"
            "import exp.lib.runs;"
            "assert not {'torch', 'hydra', 'omegaconf', 'mujou'} & set(sys.modules), sys.modules.keys();"
            "runpy.run_module('exp.lib.runs', run_name='__main__')"
        )
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                "table",
                "--root",
                str(tmp_path),
                "--rows",
                "seed",
                "--value",
                "roll10/cv10",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.splitlines()[2] == "| 0 | 0.10 |"
