"""The token statistics of the checkpoint diagnosis (M17): closed forms of
tokens built from a position part and a stack part, the masked tokens read
from the mask itself, and the arithmetic of the stretches and the videos."""

import torch
from torch import Tensor

from exp.video.diag.diag_ckpt import spread, stretch_ends, videos_of, view_errors


def _views(n: int, t: int, keeps: list[int], g: torch.Generator) -> Tensor:
    """Masks ``(n, views * t)`` with ``keeps[v]`` visible tokens a row."""
    rows = []
    for k in keeps:
        ranks = torch.rand(n, t, generator=g).argsort(1).argsort(1)
        rows.append((ranks >= k).float())
    return torch.cat(rows, 1)


def _packed(target: Tensor, y: Tensor, keeps: list[int]) -> Tensor:
    """The target tokens, then each view's visible targets in id order (an
    encoder that sees each token alone)."""
    t, d = target.shape[1], target.shape[2]
    parts = [target]
    for vi, k in enumerate(keeps):
        ids = y[:, vi * t : (vi + 1) * t].sort(dim=1, stable=True).indices[:, :k]
        parts.append(target.gather(1, ids[..., None].expand(-1, -1, d)))
    return torch.cat(parts, 1)


def test_spread_of_a_position_part_plus_a_stack_part():
    """s(x, slot, p) = a(p) + c(x): within is the variance of a over the
    positions, between that of c over the stacks, temporal 0."""
    g = torch.Generator().manual_seed(0)
    n, slots, per, d = 5, 3, 4, 2
    a = torch.randn(per, d, generator=g, dtype=torch.float64)
    c = torch.randn(n, d, generator=g, dtype=torch.float64)
    s = (a[None, None] + c[:, None, None]).expand(n, slots, per, d)
    out = spread(s.reshape(n, slots * per, d), per)
    torch.testing.assert_close(out["within"], a.var(0, unbiased=False).mean())
    torch.testing.assert_close(out["between"], c.var(0, unbiased=False).mean())
    torch.testing.assert_close(out["temporal"], torch.zeros((), dtype=s.dtype))
    torch.testing.assert_close(out["norm2"], s.pow(2).mean())


def test_view_errors_are_those_of_the_masked_tokens():
    """The oracle predictor has error 0; the masked targets' own mean has
    error ``self``, their variance read from the mask; ``tgtmean`` is the
    error of the stack's mean token."""
    g = torch.Generator().manual_seed(1)
    n, t, d, keeps = 3, 12, 2, [7, 4]
    target = torch.randn(n, t, d, generator=g, dtype=torch.float64)
    y = _views(n, t, keeps, g)
    tok = _packed(target, y, keeps)

    def goal(ti: Tensor) -> Tensor:
        return target.gather(1, ti[..., None].expand(-1, -1, d))

    exact = view_errors(tok, y, keeps, lambda c, ci, ti: goal(ti))
    own = view_errors(
        tok,
        y,
        keeps,
        lambda c, ci, ti: goal(ti).mean(1, keepdim=True).expand_as(goal(ti)),
    )
    for vi, k in enumerate(keeps):
        masked = y[:, vi * t : (vi + 1) * t] == 1
        m = torch.stack([target[i][masked[i]] for i in range(n)])  # (n, t - k, d)
        assert m.shape[1] == t - k
        self_ = (m - m.mean(1, keepdim=True)).pow(2).mean()
        assert float(exact[vi]["pred"]) == 0.0
        torch.testing.assert_close(exact[vi]["self"], self_)
        torch.testing.assert_close(own[vi]["pred"], self_)
        mean = target.mean(1, keepdim=True)
        torch.testing.assert_close(exact[vi]["tgtmean"], (m - mean).pow(2).mean())


def test_tokens_constant_in_a_stack_are_completed_by_the_context_mean():
    """The collapse of the audit: a constant of each stack is predicted
    exactly by the mean of the visible tokens."""
    g = torch.Generator().manual_seed(2)
    n, t, d, keeps = 4, 10, 3, [6, 3]
    target = torch.randn(n, 1, d, generator=g, dtype=torch.float64).expand(n, t, d)
    y = _views(n, t, keeps, g)
    tok = _packed(target, y, keeps)
    errs = view_errors(
        tok,
        y,
        keeps,
        lambda c, ci, ti: c.mean(1, keepdim=True).expand(-1, ti.shape[1], -1),
    )
    zero = torch.zeros((), dtype=torch.float64)
    for e in errs:
        for key in ("pred", "ctxmean", "tgtmean", "self"):
            torch.testing.assert_close(e[key], zero)
    torch.testing.assert_close(spread(target, 5)["within"], zero)


def test_stretch_ends_are_distinct_frames_of_one_stretch():
    length, n, frac, first = 2400, 32, 1 / 24, 48
    span = int(length * frac)
    for seed in range(5):
        e = stretch_ends(
            length,
            n,
            frac=frac,
            first=first,
            generator=torch.Generator().manual_seed(seed),
        )
        assert len(e) == n and len(set(e.tolist())) == n
        assert torch.equal(e, e.sort().values)
        assert int(e[0]) >= first and int(e[-1]) < length
        assert int(e[-1] - e[0]) < span


def test_videos_of_counts_by_the_first_frames():
    firsts = torch.tensor([0, 10, 25])  # videos of 10, 15 and 5 frames
    ends = torch.tensor([0, 9, 10, 24, 25, 29, 3])
    assert videos_of(ends, firsts, 3) == [3, 2, 2]
    assert videos_of(torch.tensor([12]), firsts, 3) == [0, 1, 0]
