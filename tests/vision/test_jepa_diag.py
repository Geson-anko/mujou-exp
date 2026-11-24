"""``exp.vision.jepa_diag`` on the tiny synthetic Walking Tours of
``test_jepa_eval``: the groups and their lags by hand, the autocovariance by
its sum, the images by hand, and the three commands end to end on tiny runs
(of V18's grid, and of arm M on a finer one), whose shared parts reproduce
``jepa_eval``'s rows. The codes of $h$ (``TokenCode``) are tested with
``jepa_eval``."""

import math
from dataclasses import replace

import pytest
import torch

from exp.lib import ckpt
from exp.lib.png import read_png
from exp.vision.jepa_diag import (
    autocov,
    evaluate_imagenette,
    evaluate_sigreg,
    evaluate_tokens,
    grid_starts,
    lag_pairs,
    partition,
    shrink,
    still_stacks,
)
from exp.vision.jepa_eval import evaluate_run
from exp.vision.models import build_vae
from exp.vision.schema import RunConf
from tests.vision.test_eval import SIZE
from tests.vision.test_jepa_eval import (
    FIT,
    HELDIN,
    SPAN,
    conf_of,
    jsonl,
    root,  # noqa: F401  (a fixture)
    tiny_run,
)
from tests.vision.test_stream_eval import EVERY


class TestGroups:
    def test_both_layouts_cut_the_same_indices_by_hand(self):
        assert partition(10, 3, "consec").tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert partition(10, 3, "spaced").tolist() == [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

    def test_too_few_or_an_unknown_layout_is_refused(self):
        with pytest.raises(ValueError):
            partition(2, 3, "consec")
        with pytest.raises(ValueError):
            partition(9, 3, "random")

    def test_the_lags_and_pairs_of_n_neighbours_are_2_n_minus_l(self):
        lags, pairs = lag_pairs(partition(8, 4, "consec"))
        assert lags.tolist() == [1, 2, 3] and pairs.tolist() == [6.0, 4.0, 2.0]

    def test_the_lags_of_spaced_groups_are_multiples_of_p(self):
        lags, pairs = lag_pairs(partition(9, 4, "spaced"))  # P = 2
        assert lags.tolist() == [2, 4, 6] and pairs.tolist() == [6.0, 4.0, 2.0]

    @pytest.mark.parametrize(
        ("frames", "want"), [(38, [0, 10, 20, 30]), (37, [0, 10, 20])]
    )
    def test_the_grid_keeps_the_stacks_that_fit(self, frames, want):
        """A stack of 8 frames from 30 ends at frame 37."""
        assert grid_starts(frames, 8, 1, 10).tolist() == want


def test_autocov_is_the_lagged_sum_over_its_pairs():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 3, generator=g, dtype=torch.float64)
    lags = torch.tensor([0, 1, 7, 49])
    u = x - x.mean(0)
    want = torch.stack([(u[: 50 - k] * u[k:]).sum(0) / (50 - k) for k in lags.tolist()])
    torch.testing.assert_close(autocov(x, lags), want)


def test_shrink_averages_blocks_and_a_still_stack_repeats_the_image():
    x = torch.arange(16, dtype=torch.uint8).view(1, 1, 4, 4).expand(2, 3, 4, 4)
    got = shrink(x, 2)
    # the blocks' means 2.5, 4.5, 10.5, 12.5 rounded half to even
    assert got[0, 0].tolist() == [[2, 4], [10, 12]] and got.dtype == torch.uint8
    assert torch.equal(shrink(x, 4), x)
    with pytest.raises(ValueError):
        shrink(x, 3)
    s = still_stacks(got, 5)
    assert s.shape == (2, 5, 3, 2, 2) and all(torch.equal(f, got) for f in s.unbind(1))


class TestCommands:
    def test_tokens_rows_and_z_as_jepa_eval(self, root, tmp_path):  # noqa: F811
        run = tiny_run(root, tmp_path)
        kw = {"root": root, "span": SPAN, "fit_steps": FIT, "spacing": EVERY}
        out = evaluate_tokens(run, steps=(0, 4), **kw)
        rows = jsonl(out / "tokens.jsonl")
        assert [(r["step"], r["code"], r["group"]) for r in rows] == [
            (s, c, g)
            for s in (0, 4)
            for c in ("z", "h_mean", "h_flat")
            for g in ("span", "after", "zurich")
        ]
        zurich = [r for r in rows if r["group"] == "zurich"]
        assert all(math.isfinite(r["recon_blur"]) for r in zurich)
        # the same decoders and dynamics as jepa_eval's for z
        ref = evaluate_run(run, probe_steps=(0,), heldin=HELDIN, **kw)
        want = {
            (r["step"], r["group"]): r
            for r in jsonl(ref / "probes.jsonl")
            if r["code"] == "z"
        }
        for r in rows:
            if r["code"] == "z" and r["group"] != "after":
                w = want[r["step"], r["group"]]
                assert r["recon_blur"] == pytest.approx(w["recon_blur"], rel=1e-6)
                assert r["dyn_1/copy_1"] == pytest.approx(w["dyn_1/copy_1"], rel=1e-6)
        assert read_png(out / "00000.8-h_flat-zurich.png").shape[0] == 2 * (
            2 * SIZE + 2
        )

    def test_tokens_of_arm_m_on_a_finer_grid(self, root, tmp_path):  # noqa: F811
        run = tiny_run(root, tmp_path, "--patch", "2", "--sigreg-on", "mean")
        kw = {"root": root, "span": SPAN, "fit_steps": FIT, "spacing": EVERY}
        out = evaluate_tokens(run, steps=(4,), pools=("h_pool", "zbar"), **kw)
        rows = jsonl(out / "tokens.jsonl")
        assert [(r["code"], r["group"]) for r in rows] == [
            (c, g) for c in ("h_pool", "zbar") for g in ("span", "after", "zurich")
        ]
        assert all(
            math.isfinite(r["recon_blur"]) for r in rows if r["group"] != "after"
        )

    def test_tokens_refuses_a_step_without_a_checkpoint(self, root, tmp_path):  # noqa: F811
        run = tiny_run(root, tmp_path)
        with pytest.raises(ValueError, match="no checkpoint"):
            evaluate_tokens(run, root=root, span=SPAN, steps=(2,), spacing=EVERY)

    def test_sigreg_rows_and_val_and_heldin_as_jepa_eval(self, root, tmp_path):  # noqa: F811
        run = tiny_run(root, tmp_path)
        out = evaluate_sigreg(run, root=root, step=1, sizes=(4, 8), heldin=HELDIN)
        rows = jsonl(out / "sigreg.jsonl")
        # H has 40 frames: 39 stacks of 2, 9 groups of 4 and 4 groups of 8
        sets = [(r["set"], r["n"], r["groups"]) for r in rows if r["step"] == 4]
        assert sets == [
            ("consec", 4, 9),
            ("spaced", 4, 9),
            ("consec", 8, 4),
            ("spaced", 8, 4),
            ("val", 3, 1),
            ("heldin", 4, 2),
            ("heldin", 8, 1),
        ]
        spaced = [r for r in rows if r["set"] == "spaced"]
        assert [r["spacing"] for r in spaced] == [9, 4] * 2  # P frames
        assert all(
            math.isfinite(r["pred"]) and math.isfinite(r["pred_iid"]) for r in spaced
        )
        # val and the held-in batches are those of jepa_eval's statistics
        ref = evaluate_run(
            run,
            root=root,
            span=SPAN,
            probe_steps=(),
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
        )
        want = {r["step"]: r for r in jsonl(ref / "jepa.jsonl") if r["code"] == "z"}
        for r in rows:
            if r["set"] == "val":
                assert r["sigreg"] == pytest.approx(
                    want[r["step"]]["sigreg_zurich"], rel=1e-5
                )
            if r["set"] == "heldin" and r["n"] == HELDIN[1]:
                assert r["sigreg"] == pytest.approx(
                    want[r["step"]]["sigreg_heldin"], rel=1e-5
                )

    def test_imagenette_rows(self, root, tmp_path):  # noqa: F811
        run = tiny_run(root, tmp_path)
        arm_m = tiny_run(
            root, tmp_path, "--patch", "2", "--sigreg-on", "mean", name="m"
        )
        g = torch.Generator().manual_seed(0)
        data = tmp_path / "inet"
        data.mkdir()
        for split, count in (("train", 30), ("val", 10)):
            x = torch.randint(256, (count, 3, 2 * SIZE, 2 * SIZE), generator=g)
            y = torch.arange(count) % 2
            item = {"x": x.to(torch.uint8), "y": y, "classes": ["a", "b"]}
            torch.save(item, data / f"{split}.pt")
        conf: RunConf = replace(conf_of(), device="cpu")
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text((run / "config.yaml").read_text())
        path = ckpt.save_eval(
            vae_run,
            program="vae",
            group="iid",
            t=1.0,
            frame=10,
            modules=build_vae(conf, seed=5).modules(),
        )
        out = evaluate_imagenette(
            data, tmp_path / "out", conf=conf, runs=[run, arm_m], v16=path, side=SIZE
        )
        rows = jsonl(out / "imagenette.jsonl")
        # the flat arm's h_mean and z; arm M's h_mean and zbar
        assert [(r.get("run"), r.get("step"), r["code"]) for r in rows] == [
            (None, None, "color"),
            (None, None, "v16"),
            ("run", 0, "h_mean"),
            ("run", 0, "z"),
            ("run", 4, "h_mean"),
            ("run", 4, "z"),
            ("m", 0, "h_mean"),
            ("m", 0, "zbar"),
            ("m", 4, "h_mean"),
            ("m", 4, "zbar"),
        ]
        assert all(0 <= r["acc"] <= 1 for r in rows)


def test_imagenette_rows_of_a_v18c_run(root, tmp_path):  # noqa: F811
    run = tiny_run(
        root,
        tmp_path,
        *("--patch", "2", "--token-dim", "16", "--proj", "fixed"),
        *("--sigreg-on", "mix"),
    )
    g = torch.Generator().manual_seed(0)
    data = tmp_path / "inet"
    data.mkdir()
    for split, count in (("train", 30), ("val", 10)):
        x = torch.randint(256, (count, 3, SIZE, SIZE), generator=g)
        item = {"x": x.to(torch.uint8), "y": torch.arange(count) % 2}
        torch.save(item | {"classes": ["a", "b"]}, data / f"{split}.pt")
    conf: RunConf = replace(conf_of(), device="cpu")
    out = evaluate_imagenette(
        data, tmp_path / "out", conf=conf, runs=[run], steps=(4,), side=SIZE
    )
    rows = jsonl(out / "imagenette.jsonl")
    # V18c's h_mean and z (= W vec(t))
    assert [(r.get("step"), r["code"]) for r in rows] == [
        (None, "color"),
        (4, "h_mean"),
        (4, "z"),
    ]
    assert all(0 <= r["acc"] <= 1 for r in rows)
