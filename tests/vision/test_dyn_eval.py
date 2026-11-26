r"""``exp.vision.dyn_eval``: the sets, the encoding's time order, the clock,
the synthetic oracle's autocovariances, and the checks split and merged.

The oracles: the roles of the permutation by hand; the layout's blocks of
``block - span`` stacks and ``block - span - 2 n`` centres, every triple of
every lag in its centre's block (no overlap); a video whose frames carry
their index encodes each stack to its own frames (the input end, the target
and the copy of a triple by hand, V-h0l's); the clock's directions the
fastest and of unit variance; ``synth_gamma`` of the rotated code the
original's conjugate, of the mixture the products of a realization within
their standard errors; the checks of one process and of parts merged, the
same rows to the byte.
"""

import json
import math

import pytest
import torch

from exp.lib.dynpred import GRID, MLPConf, Triples
from exp.lib.video_io import U8Writer
from exp.vision.dyn_eval import (
    FAMILIES,
    CheckConf,
    DynConf,
    _realization,
    check_parts,
    check_units,
    clock_code,
    code_rows,
    encode_layout,
    layout_of,
    merge_checks,
    split_videos,
    synth_gamma,
    video_sets,
)

F64 = torch.float64


class TestSets:
    def test_roles_of_the_permutation(self):
        names = [f"v{i}" for i in range(9)]
        roles = split_videos(list(reversed(names)), 4101)
        perm = torch.randperm(9, generator=torch.Generator().manual_seed(4101)).tolist()
        assert roles["rec"] == (names[perm[0]],) and roles["val"] == (names[perm[1]],)
        assert roles["train"] == tuple(names[i] for i in perm[2:])

    def test_layout_counts_and_blocks(self):
        lay = layout_of([2600, 1300, 900], span=7, block=1280, max_lag=256)
        per = 1280 - 7
        assert lay.n_blocks == 3  # 2 + 1 + 0 whole blocks
        assert len(lay.frame) == 3 * per
        assert len(lay.centres) == 3 * (per - 2 * 256)
        for n in (1, 100, 256):
            c = lay.centres
            for idx in (c - n, c, c + n):
                assert torch.equal(lay.block[idx], lay.blocks)  # one block a triple
            assert torch.equal(lay.video[c - n], lay.video[c + n])
            # the last frame of the future stack lies in the block's frames
            assert ((lay.frame[c + n] + 7) // 1280 == lay.frame[c - n] // 1280).all()

    def test_the_sets_key_follows_the_videos(self):
        a = video_sets(["a", "b", "c"], [2600, 2600, 2600], 7, DynConf())
        b = video_sets(["a", "b", "c"], [2600, 2600, 2601], 7, DynConf())
        assert a.key != b.key


class TestEncoding:
    def test_each_stack_is_its_own_frames(self, tmp_path):
        frames, stride, block = 3, 2, 40
        for name, length in (("a", 90), ("b", 45)):
            with U8Writer(tmp_path / name, (3, 2, 2)) as w:
                for f in range(length):
                    w.append(
                        torch.full(
                            (3, 2, 2),
                            f + (100 if name == "b" else 0),
                            dtype=torch.uint8,
                        )
                    )
        span = (frames - 1) * stride
        lay = layout_of([90, 45], span, block, max_lag=5)
        fn = lambda x: {"own": x[:, :, 0, 0, 0].double()}  # noqa: E731
        got = encode_layout(
            fn, tmp_path, ["a", "b"], lay, block, frames, stride, torch.device("cpu")
        )["own"]
        base = torch.where(lay.video == 1, 100, 0)[:, None]
        want = base + lay.frame[:, None] + torch.arange(frames) * stride
        assert torch.equal(got, want.double())
        # a triple's input end, centre and target: the stacks n before and after
        s = Triples(got[:, -1:], lay.centres, lay.blocks, lay.n_blocks)
        now, past, fut = s.at(3)
        assert torch.equal(now - past, torch.full_like(now, 3.0))
        assert torch.equal(fut - now, torch.full_like(now, 3.0))


class TestClock:
    def test_the_fastest_directions_turn(self):
        gen = torch.Generator().manual_seed(0)
        n, k = 4000, 12
        slow = torch.cumsum(torch.randn(n, k, generator=gen, dtype=F64), 0) / 30
        fast = torch.randn(n, k, generator=gen, dtype=F64)
        mix = torch.arange(k) % 3 == 0  # columns 0, 3, 6, 9 fast
        w = torch.where(mix, fast, slow)
        cen = torch.arange(20, n - 20)
        sets = {
            name: Triples(w, cen, cen // 1000, 4)
            for name in ("train", "val", "rec", "zurich")
        }
        times = {name: (torch.arange(n), torch.arange(n) // 1000, 4) for name in sets}
        conf = DynConf(clock_dirs=4, clock_lag=1)
        out = clock_code(sets, times, conf)
        changed = (out["train"].w != w).any(0)
        assert changed.tolist() == mix.tolist()
        x = out["zurich"].w[:, mix]
        assert float(x.square().sum(1).mean()) == pytest.approx(
            4.0, rel=1e-12
        )  # 2 planes of radius sqrt2 x 2


class TestOracle:
    def test_the_rotated_code_is_the_conjugate(self):
        cc = CheckConf()
        from exp.vision.content_eval import orthogonal

        q = orthogonal(cc.d, cc.rot_seed)
        for lag in (0, 8):
            g = synth_gamma("S9-S4", lag, torch.ones(1), cc)
            torch.testing.assert_close(
                g, q @ synth_gamma("S4", lag, torch.ones(1), cc) @ q.T
            )

    def test_s10_is_the_rotation_with_its_noise(self):
        cc = CheckConf()
        g0, g1 = (
            synth_gamma("S10a", 0, torch.ones(1), cc),
            synth_gamma("S10a", 5, torch.ones(1), cc),
        )
        torch.testing.assert_close(g0, torch.eye(cc.d, dtype=F64))
        torch.testing.assert_close(
            g1 * (1 + cc.nus[0]), synth_gamma("S1", 5, torch.ones(1), cc)
        )

    @pytest.mark.slow
    def test_the_mixtures_follow_their_realization(self):
        # S7 and S8 on a realization: the eval set's products at lag n
        # against the mixture of its own blocks' weights
        cc = CheckConf(train_n=9 * 2600, eval_n=6 * 1280)
        for code, fam in (("S7", "chain"), ("S8", "amp")):
            syn = _realization(fam, 1, 6, cc)
            ev = syn.sets[code]["eval"]
            kinds = 3 if code == "S7" else 5
            weights = torch.bincount(
                syn.kinds["eval"] % kinds, minlength=kinds
            ).double()
            weights = weights / weights.sum()
            n = 23
            prod = (
                ev.w[ev.centres + n, :, None] * ev.w[ev.centres, None, :]
            )  # (C, d, d)
            per_block = torch.zeros(ev.n_blocks, cc.d, cc.d, dtype=F64).index_add_(
                0, ev.blocks, prod
            )
            per_block = per_block / torch.bincount(ev.blocks)[:, None, None]
            want = synth_gamma(code, n, weights, cc)
            diag = torch.arange(0, cc.d, 2)
            got = per_block.mean(0)
            # the planes' own 2 x 2 blocks: one phase per block and plane, so
            # the spread over the blocks and the planes is the error
            g = torch.stack([got[diag, diag], got[diag + 1, diag]], 1)  # cos and sin
            w = torch.stack([want[diag, diag], want[diag + 1, diag]], 1)
            se = torch.stack(
                [per_block[:, diag, diag], per_block[:, diag + 1, diag]], 2
            ).std((0, 1)) / math.sqrt(ev.n_blocks * len(diag))
            assert ((g.mean(0) - w.mean(0)).abs() <= 4.5 * se).all()


def _tiny() -> tuple[CheckConf, MLPConf]:
    cc = CheckConf(train_n=9 * 2600, eval_n=2 * 1280 + 100, long_parts=2,
                   cover_seeds=(5101, 5103), null_seeds=(5301, 5303), boot=50)  # fmt: skip
    return cc, MLPConf(width=8, batch=16, steps=4, every=2)


@pytest.mark.slow
class TestChecks:
    def test_parts_merge_to_the_rows_of_one_run(self):
        cc, mc = _tiny()
        whole = check_parts(cc, check_units(cc), None, mc)
        parts = []
        units = list(check_units(cc))
        for u in units[::-1]:
            if u in ("V-p1f", "V-p1h"):
                parts += check_parts(cc, [u], (5102, 6000), mc)
                parts += check_parts(cc, [u], (0, 5102), mc)
            else:
                parts += check_parts(cc, [u], None, mc)
        a = json.dumps(merge_checks(whole, cc), sort_keys=True)
        b = json.dumps(merge_checks(parts, cc), sort_keys=True)
        assert a == b

    def test_a_missing_unit_stops(self):
        cc, mc = _tiny()
        rows = check_parts(cc, ["long-S4"], None, mc)
        with pytest.raises(ValueError, match="missing"):
            merge_checks(rows, cc)


def test_every_family_code_is_a_unit():
    codes = {c for cs in FAMILIES.values() for c in cs} - {"S4b"}
    assert {f"long-{c}" for c in codes} <= set(check_units())
    assert GRID[-1] == DynConf().max_lag


@pytest.mark.slow
class TestRows:
    """``code_rows`` through ``judge_dyn``: an orthogonal map of the codes
    changes no linear ratio (V-p2a), the ridge's bounds and the MLP's
    validation hold (V-p2b), a summary is made."""

    def _sets(self, w: torch.Tensor) -> dict[str, Triples]:
        lay = layout_of([600, 600], span=0, block=100, max_lag=4)
        n = len(lay.frame)
        out = {}
        for i, name in enumerate(("train", "val", "rec")):
            out[name] = lay.triples(w[i * n : (i + 1) * n])
        z = out["rec"]
        out["zurich"] = Triples(w[3 * n : 4 * n], z.centres, z.blocks, z.n_blocks,
                                (z.blocks + 1) // 2, z.n_blocks // 2 + 1)  # fmt: skip
        return out

    def test_rot_and_the_bounds(self):
        from exp.lib.synth_series import ou
        from exp.vision.content_eval import orthogonal
        from exp.vision.results.dyn.judge_dyn import summary, vp2a, vp2b

        w = ou(4 * 1200, 6, 0.8, torch.Generator().manual_seed(0))
        conf = DynConf(grid=(1, 2, 4), max_lag=4, main=(2, 4),
                       mlp=MLPConf(width=8, batch=16, steps=6, every=2, seeds=(1, 2, 3)))  # fmt: skip
        dev = torch.device("cpu")
        pca = code_rows(self._sets(w), conf, dev)
        rot = code_rows(self._sets(w @ orthogonal(6, 7)), conf, dev)
        got = vp2a(rot, pca)
        assert got["pass"] and got["rows"] == 4 * 4
        assert vp2b(pca)["pass"]
        s = summary(pca)
        assert set(s["curves"]) >= {
            "lin",
            "mlp",
            "lin1",
            "vel",
            "mean",
            "scale",
            "copy",
        }
        # the copy's R_copy is 1 at every lag
        assert all(abs(v - 1) < 1e-12 for v in s["curves"]["copy"]["copy"])
