r"""``exp.vision.gen_eval``: the candidates of the plan's grid (512, at
least 70 frames apart, inside the video), the hard sets (33, the ends left
out), the reference set, the training set of no run's seed, ``uncond``, and
tiny runs, yardsticks and references end to end on the synthetic Walking
Tours of ``test_eval`` (a cheap feature in place of Inception's)."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from torch import Tensor
from torch.nn import functional as F

from exp.lib import ckpt
from exp.lib.png import read_png
from exp.vision.compose import to_dict
from exp.vision.gen_eval import (
    GenConf,
    ZeroCode,
    candidates,
    evaluate_refs,
    evaluate_run,
    evaluate_yardsticks,
    hard_sets,
    spread,
    training_set,
)
from exp.vision.jepa_diag import grid_starts
from exp.vision.jepa_eval import W_CHUNK, W_DIM, W_SEED
from exp.vision.models import build_vae
from tests.lib.test_flow_decoder import SMALL
from tests.vision.test_eval import SIZE
from tests.vision.test_jepa_eval import conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

ZURICH, STACK, STEP = 38996, 8, 10  # the plan's: frames, stack, grid step

TINY = GenConf(
    n_cand=8,
    samples=2,
    sample_steps=3,
    set_size=12,
    fit_steps=4,
    fit_batch=4,
    warmup=2,
    loss_window=2,
    log_every=2,
    ref_count=8,
    grid_step=1,
    hard_reach=2,
    hard_stride=2,
    fm_points=2,
    picture_rows=4,
    picture_samples=2,
    picture_scale=1,
    sample_batch=8,
    net=SMALL,
)


def features(x: Tensor) -> Tensor:
    """A cheap feature of u8 frames: the means of 4 x 4 blocks."""
    return F.avg_pool2d(x.float() / 255, 4).flatten(1)


@pytest.fixture
def root(tmp_path) -> Path:
    from tests.vision.test_eval import random_tours

    return random_tours(tmp_path / "d")


class TestCandidates:
    def test_512_of_the_plans_grid_apart_and_inside(self):
        starts = grid_starts(ZURICH, STACK, 1, STEP)
        assert len(starts) == 3899
        c = candidates(len(starts), 512)
        assert len(c) == 512 and c[0] == 0 and c[-1] == len(starts) - 1
        first = starts[c]
        assert int(first.diff().min()) >= 70  # 7 or 8 s
        assert int(first.diff().min()) >= STACK  # no frame shared
        assert int(first[-1]) + STACK - 1 <= ZURICH - 1

    def test_too_many_are_refused(self):
        with pytest.raises(ValueError, match="candidates"):
            candidates(10, 11)


class TestHardSets:
    def test_33_neighbours_its_own_first_and_the_ends_left_out(self):
        g = 3899
        c = candidates(g, 512)
        q, among = hard_sets(c, g, 16, 2)
        assert among.shape == (len(q), 33)
        assert torch.equal(among[:, 0], c[q])
        assert int(among.min()) >= 0 and int(among.max()) <= g - 1
        kept = [i for i, ci in enumerate(c.tolist()) if 32 <= ci <= g - 33]
        assert q.tolist() == kept and len(c) - len(q) < 12
        diffs = (among[:, 1:] - among[:, :1]).abs()
        assert set(diffs.flatten().tolist()) == set(range(2, 33, 2))


class TestSpread:
    def test_evenly_over_the_video_from_its_first_to_its_last(self):
        last = spread(4096, ZURICH, STACK - 1)
        assert len(last) == 4096 and int(last[0]) == STACK - 1
        assert int(last[-1]) == ZURICH - 1
        assert int(last.diff().min()) >= 9


class TestTrainingSet:
    def test_the_same_stacks_whatever_the_runs_seed(self, root):
        a = replace(conf_of(), seed=0)
        b = replace(conf_of(), seed=1)
        za, fa = training_set(a, root, {"u": ZeroCode(3)}, 10)
        zb, fb = training_set(b, root, {}, 10)
        assert torch.equal(fa, fb) and fa.shape == (10, 3, SIZE, SIZE)
        assert torch.equal(za["u"], torch.zeros(10, 3))


def rows_of(out: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("rows/*.json"))}


KEYS = ("top1", "top5", "hard_top1", "feat_dist", "feat_ratio", "fid")
KEYS += ("fm_zurich", "fit_loss", "fit_loss_first")


class TestEvaluate:
    def test_a_tiny_runs_jobs_rows_pictures_and_decoders(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        kw = {"root": root, "features": features, "gconf": TINY, "spacing": EVERY}
        out = evaluate_run(run, steps=(0, 4), codes=("z", "z_w"), **kw)
        rows = rows_of(out)
        assert list(rows) == [f"{s:06d}-{c}-d0" for s in (0, 4) for c in ("z", "z_w")]
        for name, row in rows.items():
            assert all(math.isfinite(row[k]) for k in KEYS), name
            assert 0 <= row["top1"] <= row["top5"] <= 1
            assert len(row["hits"]) == len(row["feat_i"]) == TINY.n_cand
            assert len(row["hard_hits"]) == row["n_hard"]
            assert row["dim"] == (1024 if row["code"] == "z_w" else 64 * 16)
            assert len(row["fit_log"]) == TINY.fit_steps // TINY.log_every
        image = read_png(out / "000004-z_w-d0.png")
        # rows of 4 candidates; true, 2 samples, mean and sd
        assert image.shape[:2] == (4 * (SIZE + 2), 5 * (SIZE + 2))
        dec = torch.load(out / "ckpt" / "dec-000004-z-d0.pt", weights_only=True)
        assert dec["code_dim"] == 1024 and dec["mean"].shape == (1024,)
        config = json.loads((out / "config.json").read_text())
        # H of 40 frames, stacks of 2, every frame: a grid of 39
        assert config["cand"] == candidates(39, 8).tolist()
        assert config["gen"]["fit_steps"] == TINY.fit_steps
        w = {"w": W_SEED, "w_dim": W_DIM, "w_chunk": W_CHUNK}
        assert {k: config["seeds"][k] for k in w} == w
        assert set(config["params"]) == {"1024"}
        for row in rows.values():
            assert row["key"] == config["key"]
            # FID of each sample apart (n images of n candidates), then the mean
            assert len(row["fid_j"]) == TINY.samples
            assert row["fid"] == pytest.approx(sum(row["fid_j"]) / TINY.samples)
        # a job whose row is there is not done again
        before = (out / "rows" / "000000-z-d0.json").read_text()
        (out / "rows" / "000004-z-d0.json").unlink()
        evaluate_run(run, steps=(0, 4), codes=("z",), **kw)
        assert (out / "rows" / "000000-z-d0.json").read_text() == before
        again = rows_of(out)["000004-z-d0"]
        # the same seeds, the same row (the timings aside)
        for k in (*KEYS, "hits"):
            assert again[k] == rows["000004-z-d0"][k], k

    def test_other_settings_are_refused_and_written_elsewhere(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        kw = {"root": root, "features": features, "spacing": EVERY}
        trial = replace(TINY, fit_steps=2, log_every=1)
        evaluate_run(run, steps=(4,), codes=("z",), gconf=trial, **kw)
        # the plan's run after a trial of fewer steps into the same place
        with pytest.raises(ValueError, match="other settings"):
            evaluate_run(run, steps=(4,), codes=("z",), gconf=TINY, **kw)
        # the PCAs' settings and the data are in the key too
        with pytest.raises(ValueError, match="other settings"):
            evaluate_run(run, steps=(4,), codes=("z",), gconf=trial, zpca_dim=8, **kw)
        config = json.loads((run / "posthoc_gen" / "config.json").read_text())
        assert config["tours"]["held_out"] == "H" and "pca" in config  # test_eval's
        # its config.json gone, its row is refused all the same
        (run / "posthoc_gen" / "config.json").unlink()
        with pytest.raises(ValueError, match="other settings"):
            evaluate_run(run, steps=(4,), codes=("z",), gconf=TINY, **kw)
        out = evaluate_run(
            run, steps=(4,), codes=("z",), gconf=TINY, out_name="posthoc_gen_b", **kw
        )
        assert rows_of(out)["000004-z-d0"]["fit_steps"] == TINY.fit_steps

    def test_an_arm_m_runs_z_is_its_zpca(self, root, tmp_path):
        run = tiny_run(root, tmp_path, "--patch", "2", "--sigreg-on", "mean")
        out = evaluate_run(
            run,
            steps=(4,),
            codes=("z",),
            root=root,
            features=features,
            gconf=TINY,
            spacing=EVERY,
            zpca_dim=4,
        )
        assert rows_of(out)["000004-z-d0"]["dim"] == 4

    def test_a_step_without_its_checkpoint_is_refused(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        with pytest.raises(ValueError, match="no checkpoint"):
            evaluate_run(run, steps=(3,), root=root, features=features, gconf=TINY)

    def test_the_yardsticks(self, root, tmp_path):
        conf = conf_of()
        arm_m = tiny_run(root, tmp_path, "--patch", "2", "--sigreg-on", "mean")
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        path = ckpt.save_eval(
            vae_run,
            program="vae",
            group="iid",
            t=1.0,
            frame=10,
            modules=build_vae(conf, seed=5).modules(),
        )
        out = evaluate_yardsticks(
            tmp_path / "y",
            codes=("color", "pca", "v16", "init_m", "uncond"),
            conf=conf,
            v16=path,
            m_run=arm_m,
            dec_seeds=(0, 1),
            root=root,
            features=features,
            gconf=TINY,
            spacing=EVERY,
            pca_components=4,
            zpca_dim=4,
        )
        rows = rows_of(out)
        assert set(rows) == {
            f"yardstick-{c}-d{j}"
            for c in ("color", "pca", "v16", "init_m", "uncond")
            for j in (0, 1)
        }
        dims = {r["code"]: r["dim"] for r in rows.values()}
        assert dims == {
            "color": 16 * 6,
            "pca": 4,
            "v16": 4,
            "init_m": 4,
            "uncond": 1024,
        }
        assert (
            rows["yardstick-uncond-d0"]["fit_loss"]
            != rows["yardstick-uncond-d1"]["fit_loss"]
        )
        dec = torch.load(out / "ckpt" / "dec-yardstick-uncond-d0.pt", weights_only=True)
        assert dec["mean"] is None  # uncond is not standardized
        # one code at a time: every code's dimension counted
        config = json.loads((out / "config.json").read_text())
        assert set(config["params"]) == {"96", "4", "1024"}

    def test_a_yardstick_without_its_checkpoint_is_refused(self, root, tmp_path):
        with pytest.raises(ValueError, match="v17"):
            evaluate_yardsticks(tmp_path, codes=("v17",), conf=conf_of(), root=root)

    def test_the_references_of_real_frames(self, root, tmp_path):
        path = evaluate_refs(
            tmp_path / "y", conf=conf_of(), root=root, features=features, gconf=TINY
        )
        refs = json.loads(path.read_text())
        assert 0 <= refs["real_shift_top1"] <= refs["real_shift_top5"] <= 1
        assert len(refs["real_shift_hits"]) == TINY.n_cand
        assert refs["fid_cand"] >= 0 and refs["fid_train"] >= 0
        assert len(refs["fid_train_j"]) == TINY.samples


class TestZeroCode:
    def test_every_stack_is_0(self):
        x = torch.zeros(3, 2, 3, 4, 4, dtype=torch.uint8)
        assert torch.equal(ZeroCode().mu(x), torch.zeros(3, 1024))
