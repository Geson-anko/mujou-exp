r"""``exp.vision.time_val``: the validation time loss and the stacks of the
collapse numbers.

The oracles: on a Gaussian AR(1) code of coefficient $r$, the best
predictor $r^n z_u$ has $\hat L^{\mathrm{rel}}_T(n) = 1 - r^{2n}$ and the
copy $z_u$ has $2(1 - r^n)$ (within the sampling error); the trapezoid
mean of a constant is it, of a line in $\log n$ the mean of its ends; a
video whose frames carry their index reads each stack to its own frames,
the held stacks from the even blocks and the calib from the odd.
"""

import pytest
import torch

from exp.lib.video_io import U8Writer
from exp.vision.dyn_eval import layout_of
from exp.vision.time_val import ValStacks, even_pick, log_mean, stacks_of, time_loss_rel

F64 = torch.float64


def _ar1(n: int, d: int, r: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    e = torch.randn(n, d, generator=gen, dtype=F64)
    z = torch.empty(n, d, dtype=F64)
    z[0] = e[0]
    s = (1 - r * r) ** 0.5
    for t in range(1, n):
        z[t] = r * z[t - 1] + s * e[t]
    return z


class TestTimeLoss:
    def test_the_best_and_the_copy_of_an_ar1(self):
        r, lags = 0.9, (1, 3, 8)
        z = _ar1(200_000, 3, r, 0)
        centres = torch.arange(8, len(z) - 8)
        best = time_loss_rel(
            z, centres, lambda a, b, n: r ** n[:, None].double() * a, lags
        )
        copy = time_loss_rel(z, centres, lambda a, b, n: a, lags)
        for n, got_b, got_c in zip(lags, best, copy, strict=True):
            assert got_b == pytest.approx(1 - r ** (2 * n), rel=0.03)
            assert got_c == pytest.approx(2 * (1 - r**n), rel=0.03)

    def test_the_mean_predicts_one(self):
        z = _ar1(50_000, 2, 0.5, 1)
        centres = torch.arange(4, len(z) - 4)
        zbar = z.mean(0)
        got = time_loss_rel(z, centres, lambda a, b, n: zbar.expand_as(a), (1, 4))
        assert got == pytest.approx([1.0, 1.0], rel=1e-3)

    def test_the_log_mean(self):
        lags = (8, 11, 16, 512)
        assert log_mean([0.7] * 4, lags) == pytest.approx(0.7, rel=1e-12)
        x = torch.tensor(lags, dtype=F64).log()
        line = (2 * x - 1).tolist()
        assert log_mean(line, lags) == pytest.approx(float(x[0] + x[-1] - 1), rel=1e-12)


def _videos(root):
    for name, length in (("a", 90), ("b", 45)):
        with U8Writer(root / name, (3, 2, 2)) as w:
            for f in range(length):
                w.append(torch.full((3, 2, 2), f + (100 if name == "b" else 0),
                                    dtype=torch.uint8))  # fmt: skip


class TestStacks:
    frames, stride, block = 3, 2, 40

    def test_each_stack_is_its_own_frames(self, tmp_path):
        _videos(tmp_path)
        lay = layout_of([90, 45], (self.frames - 1) * self.stride, self.block, 5)
        which = torch.arange(len(lay.video))
        got = stacks_of(tmp_path, ["a", "b"], lay, which, self.frames, self.stride,
                        self.block)[:, :, 0, 0, 0].long()  # fmt: skip
        base = torch.where(lay.video == 1, 100, 0)[:, None]
        want = base + lay.frame[:, None] + torch.arange(self.frames) * self.stride
        assert torch.equal(got, want)

    def test_held_of_the_even_blocks_and_calib_of_the_odd(self, tmp_path):
        _videos(tmp_path)
        lay = layout_of([90, 45], (self.frames - 1) * self.stride, self.block, 5)
        fit = torch.zeros(1)
        vs = ValStacks.make(tmp_path, ["a", "b"], lay, fit, frames=self.frames,
                            stride=self.stride, block=self.block, count=7)  # fmt: skip
        assert vs.fit_stacks is fit
        assert len(vs.zurich_stacks) == len(vs.zurich_calib_stacks) == 7
        first = {int(v) for v in lay.frame + torch.where(lay.video == 1, 100, 0)}
        # blocks: a's 0 and 1 (frames 0-39, 40-79), b's 2 (100-139)
        held = vs.zurich_stacks[:, 0, 0, 0, 0].long()
        calib = vs.zurich_calib_stacks[:, 0, 0, 0, 0].long()
        assert all(int(f) in first for f in torch.cat([held, calib]))
        assert all(f < 40 or f >= 100 for f in held.tolist())
        assert all(40 <= f < 80 for f in calib.tolist())
        assert (
            held[0] == 0 and held[-1] == 100 + 40 - 1 - (self.frames - 1) * self.stride
        )

    def test_the_even_pick(self):
        pool = torch.arange(10, 30)
        got = even_pick(pool, 5)
        assert got[0] == 10 and got[-1] == 29 and len(got) == 5
        assert torch.equal(even_pick(pool, 20), pool)
