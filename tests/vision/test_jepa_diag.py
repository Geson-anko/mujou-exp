"""``exp.vision.jepa_diag`` on the tiny synthetic Walking Tours of
``test_jepa_eval``: h by a forward built by hand, the groups and their lags
by hand, the autocovariance by its sum, the images by hand, and the three
commands end to end on a tiny run, whose shared parts reproduce
``jepa_eval``'s rows."""

import math
from dataclasses import replace

import pytest
import torch

from exp.lib import ckpt
from exp.lib.png import read_png
from exp.vision.jepa import TokenEncoder, build_jepa
from exp.vision.jepa_diag import (
    TokenCode,
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
from exp.vision.models import ViTPart, build_vae
from exp.vision.schema import RunConf
from tests.vision.test_eval import SIZE, N
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


class TestTokenCode:
    def test_h_is_the_last_layer_norm_of_a_forward_by_hand(self):
        conf = conf_of()
        mods = build_jepa(conf, seed=3)
        g = torch.Generator().manual_seed(0)
        x = torch.randint(256, (3, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
        assert isinstance(mods.encoder, TokenEncoder)
        parts = [p for p in mods.encoder.parts if isinstance(p, ViTPart)]
        embed, norm = parts[0].embed, parts[-1].norm
        assert embed is not None and norm is not None
        h = embed(x.reshape(3, N * 3, SIZE, SIZE).float() / 255)
        for part in parts:
            for block in part.blocks:
                h = block(h)
        h = norm(h)  # (3, T, w)
        with torch.no_grad():
            got = {p: TokenCode(mods, p).mu(x) for p in ("z", "h_mean", "h_flat")}
            torch.testing.assert_close(got["h_flat"], h.flatten(1))
            torch.testing.assert_close(got["h_mean"], h.mean(1))
            # h is the projection's input: z_p = P h_p, flattened
            torch.testing.assert_close(got["z"], mods.proj(h).flatten(1))

    def test_an_unknown_pool_is_refused(self):
        with pytest.raises(ValueError, match="pool"):
            TokenCode(build_jepa(conf_of(), seed=0), "h_max")


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
        want = {(r["step"], r["group"]): r for r in jsonl(ref / "probes.jsonl")}
        for r in rows:
            if r["code"] == "z" and r["group"] != "after":
                w = want[r["step"], r["group"]]
                assert r["recon_blur"] == pytest.approx(w["recon_blur"], rel=1e-6)
                assert r["dyn_1/copy_1"] == pytest.approx(w["dyn_1/copy_1"], rel=1e-6)
        assert read_png(out / "00000.8-h_flat-zurich.png").shape[0] == 2 * (
            2 * SIZE + 2
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
        want = {r["step"]: r for r in jsonl(ref / "jepa.jsonl")}
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
            data, tmp_path / "out", conf=conf, runs=[run], v16=path, side=SIZE
        )
        rows = jsonl(out / "imagenette.jsonl")
        assert [(r.get("step"), r["code"]) for r in rows] == [
            (None, "color"),
            (None, "v16"),
            (0, "h_mean"),
            (0, "z"),
            (4, "h_mean"),
            (4, "z"),
        ]
        assert all(0 <= r["acc"] <= 1 for r in rows)
