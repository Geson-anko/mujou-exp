r"""``exp.vision.sem_eval`` (V18c's main measure $m$, remade).

The oracles: ``zurich_split``'s sets are disjoint, leave out each segment's
first ``gap`` and with them cover the grid; ``far_pairs`` is a permutation
whose pairs are far apart; the nuisances by hand (brightness, white balance,
a blur's taps of sum 1) and ``match_stats``'s means and sds; the yardsticks'
nesting by least squares (the first $k$ of one PCA, ``color`` a linear
function of ``color8``); ``z_k3`` is $W \mathrm{vec}(1 \otimes \bar t)$ of
rank at most $d$ once centred. End to end on the synthetic Walking Tours
(hires frames the 64 px ones enlarged, a cheap feature for Inception's):
the frames' list, the targets (whitened on the fit set), the yardsticks
(``mean`` exactly 0, ``init_c_k3`` under its bound $b_d$) and a V18c-like
run's rows, then ``fill_eval``'s E1 and E1f on the split and its diagnosis.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from exp.lib.readout import centered_r2, fit_ridge, predict
from exp.lib.tours.hires import FRAME_LIST, finalize as hires_finalize
from exp.lib.video_io import U8Reader, U8Writer
from exp.vision import fill_eval
from exp.vision.jepa import build_jepa
from exp.vision.jepa_eval import ColorCode, PCACode, pixels
from exp.vision.sem_eval import (
    OUT,
    SPLITS,
    K3Code,
    SemConf,
    blur,
    evaluate_frames,
    evaluate_run,
    evaluate_targets,
    evaluate_yardsticks,
    far_pairs,
    match_stats,
    white_balance,
    zurich_split,
)
from tests.vision.test_eval import SIZE
from tests.vision.test_fill_eval import TINY, features, stacks
from tests.vision.test_jepa_eval import SPAN, conf_of, tiny_run
from tests.vision.test_stream_eval import EVERY

FACTOR = 5  # the synthetic hires frames: the 16 px ones, each pixel 5 x 5
TINY_SEM = SemConf(
    grid_step=1,
    segments=6,
    gap=1,
    sem_dim=8,
    scene_dim=2,
    sens_gap=5,
    l_min=2,
    l_max=4,
    k3_rank=4,
    pca_max=8,
    batch=8,
)
V18C = ("--patch", "2", "--token-dim", "4", "--proj", "fixed", "--sigreg-on", "mix")


class TestSplit:
    def test_the_sets_are_disjoint_and_cover_all_but_the_gaps(self):
        g, segments, gap = 3899, 6, 30
        got = zurich_split(g, segments, gap)
        sets = [set(got[s].tolist()) for s in SPLITS]
        assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1 :])
        size = g // segments
        dropped = {s * size + j for s in range(segments) for j in range(gap)}
        assert not dropped & set().union(*sets)
        assert set().union(*sets) | dropped == set(range(g))
        assert (got["segment"] == -1).sum() == len(dropped)
        # calib, sel and final have segments (0, 3), (1, 4), (2, 5)
        for j, s in enumerate(SPLITS):
            assert set(got["segment"][got[s]].tolist()) == {j, j + 3}
        again = zurich_split(g, segments, gap)
        assert all(torch.equal(got[k], again[k]) for k in got)

    def test_segments_no_longer_than_the_gap_are_refused(self):
        with pytest.raises(ValueError, match="gap"):
            zurich_split(60, 6, 10)


class TestFarPairs:
    def test_a_permutation_of_pairs_far_apart_the_same_each_time(self):
        split = zurich_split(3899)
        idx = split["sel"]
        pi = far_pairs(idx, 300, torch.Generator().manual_seed(103))
        assert sorted(pi.tolist()) == list(range(len(idx)))
        assert bool(((idx[pi] - idx).abs() >= 300).all())
        again = far_pairs(idx, 300, torch.Generator().manual_seed(103))
        assert torch.equal(pi, again)


class TestNuisances:
    def test_white_balance_and_brightness_by_hand(self):
        x = torch.tensor([[[[0, 100]], [[200, 255]], [[10, 255]]]], dtype=torch.uint8)
        got = white_balance(x)
        want = [[[[0, 110]], [[200, 255]], [[9, 230]]]]  # 255 x 1.1 cut at 255
        assert got.tolist() == want

    def test_a_blur_keeps_a_constant_frame(self):
        # the taps sum to 1 and the edges are reflected: a constant stays
        x = torch.full((1, 3, 40, 40), 77, dtype=torch.uint8)
        assert torch.equal(blur(x, 5.0), x)

    def test_matched_frames_have_the_references_means_and_sds(self):
        g = torch.Generator().manual_seed(0)
        src = torch.randint(256, (2, 3, 20, 20), generator=g, dtype=torch.uint8)
        ref = torch.randint(50, 120, (2, 3, 20, 20), generator=g, dtype=torch.uint8)
        got = match_stats(src, ref)
        r = ref.double()
        torch.testing.assert_close(got.mean((2, 3)), r.mean((2, 3)), atol=1e-6, rtol=0)
        torch.testing.assert_close(got.std((2, 3)), r.std((2, 3)), atol=1e-6, rtol=0)


class TestNesting:
    def test_the_first_k_of_one_pca_never_fit_worse(self):
        # least squares on nested column spaces: on the fit set itself, at
        # the least ridge, m does not fall as k grows
        x = stacks(40, 11)
        gen = torch.Generator().manual_seed(4)
        code = PCACode.fit(pixels, x, 32, gen=gen, device="cpu")
        z = code.mu(x)
        y = torch.randn(40, 5, generator=torch.Generator().manual_seed(5))
        y = y + 0.01 * z[:, :5].double()
        ms = [
            centered_r2(predict(fit_ridge(z[:, :k], y, 1e-8), z[:, :k]), y)
            for k in (4, 8, 16, 32)
        ]
        assert all(b >= a - 1e-6 for a, b in zip(ms, ms[1:], strict=False))

    def test_color_is_the_mean_of_four_of_color8(self):
        x = stacks(3, 12)
        c16, c8 = ColorCode(8).mu(x), ColorCode(4).mu(x)  # SIZE 16: 2 x 2, 4 x 4
        t8 = ColorCode(4).tokens(x).reshape(3, 4, 4, -1)  # (B, rows, cols, 3n)
        mean = t8.reshape(3, 2, 2, 2, 2, -1).mean((2, 4)).reshape(3, 4, -1)
        torch.testing.assert_close(mean.flatten(1), c16, atol=1e-6, rtol=0)
        assert c8.shape[1] == 4 * c16.shape[1]


class TestK3Code:
    def test_it_is_w_of_the_mean_token_repeated_and_of_rank_d(self):
        conf = replace(conf_of(), model=replace(conf_of().model, patch=2))  # 64 tokens
        mods = build_jepa(conf, seed=0, token_dim=4, proj="fixed", latent_dim=64)
        x = stacks(20, 13)
        with torch.no_grad():
            got = K3Code(mods).mu(x)
            t = mods.tokens(x)  # (B, T, d)
            assert mods.mix is not None
            w = mods.mix.weight  # (D, T d)
            assert isinstance(w, torch.Tensor)
            want = torch.einsum("btd,ktd->bk", t.mean(1)[:, None].expand_as(t),
                                w.reshape(64, t.shape[1], 4))  # fmt: skip
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
        s = torch.linalg.svdvals((got - got.mean(0)).double())
        # float32 codes: the null directions are rounding, below 1e-4 of the top
        assert int((s > 1e-4 * s[0]).sum()) <= 4

    def test_a_jepa_without_w_is_refused(self):
        mods = build_jepa(conf_of(), seed=0)
        with pytest.raises(ValueError, match="mix"):
            K3Code(mods)


# -- end to end ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path) -> Path:
    from tests.vision.test_eval import random_tours

    return random_tours(tmp_path / "d")


def hires_of(root: Path, lists: dict[str, list[int]], out: Path) -> Path:
    """Hires data of the frames ``lists`` of ``root``: each pixel ``FACTOR``
    x ``FACTOR``, so that their means are the frames."""
    size = SIZE * FACTOR
    for video, frames in lists.items():
        ref = U8Reader(root / video)
        meta = {
            "video": video,
            "data": root.name,
            "data_index_sha256": "0",
            "fps": 10.0,
            "geometry": {"side": SIZE, "x0": 0, "size": size},
            "filter": "nearest",
        }
        with U8Writer(out / video, (3, size, size), meta) as w:
            for f in frames:
                w.append(
                    ref[f].repeat_interleave(FACTOR, 1).repeat_interleave(FACTOR, 2)
                )
        (out / video / FRAME_LIST).write_text(
            json.dumps({"frames": frames, "max_diff": 0})
        )
    return hires_finalize(out)


KW = {"sconf": TINY_SEM, "spacing": EVERY, "span": SPAN}


@pytest.mark.slow
class TestEvaluate:
    def test_targets_yardsticks_a_run_and_e1_on_the_split(self, root, tmp_path):
        conf = conf_of()
        lists = json.loads(
            evaluate_frames(
                tmp_path / "frames.json", conf=conf, root=root, **KW
            ).read_text()
        )
        assert "H" in lists and set(lists) <= {"A", "B", "C", "H"}
        assert all(f == sorted(set(f)) for f in lists.values())
        hires = hires_of(root, lists, tmp_path / "hires")
        cache = tmp_path / "cache"
        path = evaluate_targets(
            hires, cache, conf=conf, root=root, features=features, **KW
        )
        assert path.name.startswith("v18c-sem-")
        meta = json.loads((path / "targets.json").read_text())
        values = torch.load(path / "targets.pt", weights_only=True)["values"]
        fit = values["sem"]["fit"]
        torch.testing.assert_close(fit.mean(0), torch.zeros(8, dtype=fit.dtype))
        torch.testing.assert_close(fit.std(0), torch.ones(8, dtype=fit.dtype))
        assert TINY_SEM.l_min <= meta["L"] <= TINY_SEM.l_max
        assert set(meta["sens"]["rho"]) == {"bright", "white", "blur"}
        assert 0 <= meta["shift_top1"] <= 1
        for s in SPLITS:
            split = meta["sets_split"][s]
            assert len(split["v"]) == len(split["idx"]) == len(split["blocks"])
        # the yardsticks: the constant exactly 0, the K3 copy under its bound
        run = tiny_run(root, tmp_path, *V18C)
        yard = evaluate_yardsticks(
            tmp_path / "y", codes=("mean", "color", "color8", "inc64", "init_c",
                                   "init_c_k3"),
            conf=conf, init_run=run, root=root, cache=cache, **KW,
        )  # fmt: skip
        rows = {
            p.name: json.loads(p.read_text()) for p in sorted((yard / "rows").iterdir())
        }
        mean = rows["yardstick-mean.json"]
        assert mean["m_sel"] == mean["m_final"] == mean["m_calib"] == 0.0
        k3 = rows["yardstick-init_c_k3.json"]
        assert (
            k3["m_sel"] <= k3["k3_bound_sel"] and k3["m_final"] <= k3["k3_bound_final"]
        )
        assert rows["yardstick-init_c.json"]["dim"] == 1024
        assert len(rows["yardstick-color.json"]["e_sel"]) == len(
            meta["sets_split"]["sel"]["idx"]
        )
        # a run's rows; init_c is its own step 0
        out = evaluate_run(run, steps=(0, 4), root=root, cache=cache, **KW)
        assert out == run / OUT
        step0 = json.loads((out / "rows" / "000000-z.json").read_text())
        same = ("m_calib", "m_sel", "m_final", "alpha")
        init = rows["yardstick-init_c.json"]
        assert {k: step0[k] for k in same} == {k: init[k] for k in same}
        z_k3 = json.loads((out / "rows" / "000004-z_k3.json").read_text())
        assert z_k3["m_sel"] <= z_k3["k3_bound_sel"]
        with pytest.raises(ValueError, match="other settings"):
            evaluate_run(
                run, steps=(4,), codes=("h_mean",), root=root, sem=path,
                sconf=replace(TINY_SEM, l_max=3), spacing=EVERY, span=SPAN,
            )  # fmt: skip
        # fill_eval on the split: E1 with the K3 stages, E1f, and the diagnosis
        # neighbours of +-1: the synthetic segments keep 5 or 8 stacks each
        fkw = {
            "fconf": replace(TINY, hard_reach=1, hard_stride=1),
            "spacing": EVERY,
            "span": SPAN,
        }
        fill_eval.evaluate_targets(
            cache, conf=conf, root=root, features=features, **fkw
        )
        fill = fill_eval.evaluate_run(
            run, steps=(4,), root=root, cache=cache, split="sem", sem=path,
            parts=("e1", "e1f"), **fkw,
        )  # fmt: skip
        e1 = json.loads((fill / "rows" / "000004-e1.json").read_text())
        n_cand = e1[0]["n_cand"]
        swaps = [r for r in e1 if r["mask"] == "S"]
        assert {r["target"] for r in swaps} == {"color", "quad"}
        for r in swaps:
            if r["stage"] in ("mean", "oracle_k3", "pred_k3"):
                assert r["swap"] == 0.5 and r["n_tie"] == n_cand
        assert {r["target"] for r in e1 if r["mask"] == "H_l"} == {"color"}
        e1f = json.loads((fill / "rows" / "000004-e1f.json").read_text())
        assert len(e1f) == 3 * 6
        assert all(
            r["m_sel"] == r["m_final"] == 0.0 for r in e1f if r["stage"] == "mean"
        )
        diag = fill_eval.evaluate_diag(run, steps=(4,), root=root, cache=cache, **fkw)
        d = json.loads((diag / "rows" / "000004-e1diag.json").read_text())
        assert all(r["resid"] <= 1e-9 for r in d)
        assert (diag / "e1diag-000004.png").is_file()
