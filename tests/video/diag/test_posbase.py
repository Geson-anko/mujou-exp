"""The trivial predictors of the ncomp check (diagnosis, M17): each is exact on
the tokens it is built for."""

import torch

from exp.video.diag.posbase import trivial_predictions


def _masked_ids(masked: torch.Tensor, keep: int) -> torch.Tensor:
    return masked.sort(dim=1, stable=True).indices[:, keep:]


def test_pos_and_pos_off_are_exact_on_position_tokens():
    """s(x, p) = g(p): pos is exact; s(x, p) = g(p) + c(x): pos_off is."""
    g = torch.Generator().manual_seed(0)
    n, slots, per, D = 3, 4, 6, 5
    T = slots * per
    gp = torch.randn(T, D, generator=g)
    masked = (torch.rand(n, T, generator=g).argsort(1) < 10).float()  # 10 masked
    keep = T - 10
    tgt = _masked_ids(masked, keep)
    s = gp.expand(n, -1, -1)
    p = trivial_predictions(s, gp, masked, keep, future=False, slots=slots, per=per)
    torch.testing.assert_close(p["pos"], gp[tgt])
    shifted = s + torch.randn(n, 1, D, generator=g)
    p = trivial_predictions(
        shifted, gp, masked, keep, future=False, slots=slots, per=per
    )
    goal = shifted.gather(1, tgt[..., None].expand(-1, -1, D))
    torch.testing.assert_close(p["pos_off"], goal)
    torch.testing.assert_close(p["copy"], p["pos_off"])


def test_copy_is_exact_on_tokens_constant_in_time():
    """future: the last f slots masked; tokens that do not change over the
    slots are the copy of the last visible slot."""
    g = torch.Generator().manual_seed(1)
    n, slots, per, D, f = 2, 4, 6, 3, 2
    space = torch.randn(n, 1, per, D, generator=g)
    s = space.expand(-1, slots, -1, -1).reshape(n, slots * per, D)
    masked = torch.zeros(n, slots * per)
    masked[:, (slots - f) * per :] = 1
    keep = (slots - f) * per
    pos = torch.randn(slots * per, D, generator=g)  # unrelated means
    p = trivial_predictions(s, pos, masked, keep, future=True, slots=slots, per=per)
    tgt = _masked_ids(masked, keep)
    torch.testing.assert_close(p["copy"], s.gather(1, tgt[..., None].expand(-1, -1, D)))
