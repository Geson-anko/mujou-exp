r"""``exp.vision.fnd_guard``: the tubelets, ``time_r2``, the half view and
the linear content.

The oracles: a stack whose pixels are their frame's number and their
column's gives each tubelet its frames and its position; the tokens a
half keeps are the pixels of that half; ``time_r2`` is 1 of tokens
constant along time (with an offset per time) and $1/T$ of tokens
independent over time ($\pm$ 5 of its standard error, $\frac1T
\sqrt{2(T-1)/(T n)}$ roughly, taken as $\sqrt{2/n}/T$); the half view of an
AR(1) field over the 4 times has the closed form of its Schur complement in
the whitened coordinates ($R = \mathrm{tr}(\Sigma_h^{-1} S) /
\mathrm{tr}(\Sigma_h^{-1} D)$, $D$ the denominator's covariance), within 4
standard errors of the blocks, and a constant added to one half moves no
sum; the readout of a code that holds a part $u$ of the observation $y =
[u; v]$ has $E^{\mathrm{lin}} = \mathrm{tr}\,\Sigma_u / \mathrm{tr}\,
\Sigma_y$ and, of AR(1) parts of $r_u$, $r_v$, $E^{d,\mathrm{lin}} =
\frac{(1 - r_u)\,\mathrm{tr}\,\Sigma_u}{(1 - r_u)\,\mathrm{tr}\,\Sigma_u +
(1 - r_v)\,\mathrm{tr}\,\Sigma_v}$, within 4 standard errors of the
blocks.
"""

import math

import pytest
import torch

from exp.lib.synth_series import ou, white
from exp.vision.fnd_guard import (
    PAIRS,
    ContentSet,
    HalfSets,
    content_sums,
    fidelity_of,
    half_keep,
    half_rows,
    noninferior,
    pixel_halves,
    time_r2,
    time_r2_threshold,
    tubelets,
)

_F64 = torch.float64


def _stack() -> torch.Tensor:
    """``(1, 8, 3, 64, 64)`` u8: channel 0 the frame, 1 the column // 8,
    2 the row // 8."""
    x = torch.zeros(1, 8, 3, 64, 64, dtype=torch.uint8)
    x[0, :, 0] = torch.arange(8, dtype=torch.uint8)[:, None, None]
    x[0, :, 1] = (torch.arange(64) // 8).to(torch.uint8)[None, None, :]
    x[0, :, 2] = (torch.arange(64) // 8).to(torch.uint8)[None, :, None]
    return x


class TestTubelets:
    def test_each_token_is_its_frames_and_position(self):
        tok = tubelets(_stack()) * 255  # (1, 4, 64, 384): (f, c, py, px)
        assert tok.shape == (1, 4, 64, 384)
        box = tok[0].reshape(4, 64, 2, 3, 8, 8)
        for j in range(4):
            for p in (0, 9, 63):
                assert box[j, p, :, 0].flatten(1).unique(dim=1).flatten().tolist() == [
                    2 * j,
                    2 * j + 1,
                ]
                assert box[j, p, :, 1].unique().tolist() == [p % 8]
                assert box[j, p, :, 2].unique().tolist() == [p // 8]

    @pytest.mark.parametrize(
        "half", ["left", "right", "top", "bottom", "early", "late"]
    )
    def test_a_half_keeps_its_pixels(self, half):
        """The tubelets of the kept tokens hold the half's pixels, the same
        multiset of values."""
        x = torch.randint(0, 256, (2, 8, 3, 64, 64), dtype=torch.uint8,
                          generator=torch.Generator().manual_seed(0))  # fmt: skip
        keep = half_keep(half).flatten(0)  # (4 * 64,)
        tok = tubelets(x).flatten(1, 2)[:, keep]  # (2, 128, 384)
        pix = pixel_halves(x, half)
        assert keep.sum() == 128
        assert torch.equal(tok.flatten(1).sort(1).values, pix.sort(1).values)

    def test_an_unknown_half(self):
        with pytest.raises(ValueError, match="no half"):
            half_keep("middle")


class TestTimeR2:
    def test_constant_along_time_is_one(self):
        g = torch.Generator().manual_seed(1)
        sigma = torch.randn(500, 1, 6, 3, generator=g)
        offset = torch.randn(1, 4, 6, 3, generator=g)
        assert time_r2(sigma.expand(-1, 4, -1, -1) + offset) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_independent_over_time_is_one_over_t(self):
        n, t, p, d = 4000, 4, 4, 4
        tok = torch.randn(n, t, p, d, generator=torch.Generator().manual_seed(2))
        se = math.sqrt(2 / (n * p * d)) / t
        assert time_r2(tok) == pytest.approx(1 / t, abs=5 * se)

    def test_the_threshold(self):
        assert time_r2_threshold(0.3, 0.41) == pytest.approx(0.46)


def _ar_field(
    n: int, r: float, p: int, gen: torch.Generator
) -> dict[str, torch.Tensor]:
    """The halves of tokens AR(1) over 4 times (each of ``p`` positions
    apart): ``early`` $[a_1, a_2]$, ``late`` $[a_3, a_4]$, ``(n, 2p)``."""
    a = [white(n, p, gen)]
    for _ in range(3):
        a.append(r * a[-1] + math.sqrt(1 - r * r) * white(n, p, gen))
    return {"early": torch.cat(a[:2], 1), "late": torch.cat(a[2:], 1)}


class TestHalfView:
    r = 0.6

    def _sets(self, shift: float = 0.0) -> HalfSets:
        g = torch.Generator().manual_seed(3)
        codes = {s: _ar_field(n, self.r, 2, g)
                 for s, n in (("ref", 400_000), ("train", 50_000), ("val", 10_000), ("eval", 50_000))}  # fmt: skip
        for s in codes:
            codes[s]["late"] = codes[s]["late"] + shift
        n = 50_000
        return HalfSets(codes, torch.arange(n) // 500, n // 500)

    def test_the_future_of_an_ar1_field(self):
        """Per position $\\Sigma_h = [[1, r], [r, 1]]$, the Schur complement
        $S = [[1 - r^2, r(1 - r^2)], [r(1 - r^2), 1 - r^4]]$, the copy's
        $(1 - r^2)[[2, r], [r, 2]]$, the mean's $\\Sigma_h$."""
        r = self.r
        sig = torch.tensor([[1, r], [r, 1]], dtype=_F64)
        s = torch.tensor(
            [[1 - r**2, r * (1 - r**2)], [r * (1 - r**2), 1 - r**4]], dtype=_F64
        )
        copy = (1 - r**2) * torch.tensor([[2, r], [r, 2]], dtype=_F64)
        inv = torch.linalg.inv(sig)
        want = {"copy": float((inv @ s).trace() / (inv @ copy).trace()),
                "clim": float((inv @ s).trace() / 2)}  # fmt: skip
        # the whitening weighs the error: not the raw 1 - (r^2 + r^4) / 2
        assert abs(want["clim"] - (1 - (r**2 + r**4) / 2)) > 0.05
        (row,) = half_rows(self._sets(), PAIRS["future"], k=4)
        num = torch.tensor(row["num_b"])
        for d in ("copy", "clim"):
            den = torch.tensor(row[f"{d}_b"])
            per = num / den
            se = float(per.std() / math.sqrt(len(per)))
            got = float(num.sum() / den.sum())
            assert got == pytest.approx(want[d], abs=4 * se + 4 / 50_000), (
                d,
                got,
                want[d],
            )

    def test_raw_coordinates_by_hand(self):
        """The raw form of the proposal: $1 - (r^2 + r^4)/2$, $(2 + r^2)/4$."""
        r = self.r
        s = 1 - r**2 + 1 - r**4
        assert s / 2 == pytest.approx(1 - (r**2 + r**4) / 2)
        assert s / (4 * (1 - r**2)) == pytest.approx((2 + r**2) / 4)

    def test_a_constant_on_one_half_moves_no_sum(self):
        (a,) = half_rows(self._sets(), PAIRS["past"], k=4)
        (b,) = half_rows(self._sets(shift=3.0), PAIRS["past"], k=4)
        for key in ("num_b", "copy_b", "clim_b"):
            torch.testing.assert_close(
                torch.tensor(a[key]), torch.tensor(b[key]), rtol=1e-6, atol=1e-6
            )
        assert a["scale"] == b["scale"]


class TestContent:
    ru, rv = 0.9, 0.3

    def _set(self, n: int, seed: int, blocks: bool) -> ContentSet:
        g = torch.Generator().manual_seed(seed)
        u = ou(n, 2, self.ru, g)
        v = ou(n, 3, self.rv, g)
        b = torch.arange(n) // 1000 if blocks else None
        return ContentSet(u.float(), torch.cat([u, v], 1).float(), torch.tensor([0, n]),
                          b, n // 1000 if blocks else 0)  # fmt: skip

    def test_a_code_of_a_part(self):
        row = content_sums(self._set(40_000, 1, False), self._set(5_000, 2, False),
                           self._set(40_000, 3, True))  # fmt: skip
        e, ed = fidelity_of(row)
        want_e = 2 / 5
        du, dv = 2 * (1 - self.ru) * 2, 2 * (1 - self.rv) * 3
        want_ed = du / (du + dv)
        for got, want, num, den in (
            (e, want_e, "err_b", "den_b"),
            (ed, want_ed, "derr_b", "dden_b"),
        ):
            per = 1 - torch.tensor(row[num]) / torch.tensor(row[den])
            se = float(per.std() / math.sqrt(len(per)))
            assert got == pytest.approx(want, abs=4 * se + 1e-3), (got, want)

    def test_the_evaluation_needs_blocks(self):
        s = self._set(100, 1, False)
        with pytest.raises(ValueError, match="blocks"):
            content_sums(s, s, s)

    @pytest.mark.parametrize(("e", "refs", "ok"),
                             [(0.5, [0.5, 0.6], True), (0.46, [0.5, 0.6], True),
                              (0.44, [0.5, 0.6], False)])  # fmt: skip
    def test_noninferiority_of_the_error(self, e, refs, ok):
        """$1 - E \\le 1.10 \\max(1 - E_{\\mathrm{ref}})$: 0.56 <= 0.55 fails."""
        assert noninferior(e, refs) == ok
