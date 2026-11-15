"""The held-out row of the BP replica (M17): the normalized completion error
is 0 for the exact prediction and 1 for the best constant of the stack."""

import torch
from torch import Tensor

from exp.video.diag.mini_train import eval_row


def _case(seed: int) -> tuple[Tensor, Tensor, Tensor, list[int]]:
    g = torch.Generator().manual_seed(seed)
    n, slots, per, d, keeps = 3, 2, 6, 4, [8, 5]
    t = slots * per
    target = torch.randn(n, t, d, generator=g, dtype=torch.float64)
    rows, parts = [], [target]
    for k in keeps:
        ranks = torch.rand(n, t, generator=g).argsort(1).argsort(1)
        masked = (ranks >= k).float()
        rows.append(masked)
        ids = masked.sort(dim=1, stable=True).indices[:, :k]
        parts.append(target.gather(1, ids[..., None].expand(-1, -1, d)))
    return target, torch.cat(parts, 1), torch.cat(rows, 1), keeps


def test_ncomp_of_the_exact_and_of_the_constant_prediction():
    target, tok, y, keeps = _case(0)
    d = target.shape[2]

    def goal(ti: Tensor) -> Tensor:
        return target.gather(1, ti[..., None].expand(-1, -1, d))

    exact = eval_row(tok, y, keeps, 6, lambda c, ci, ti: goal(ti))
    const = eval_row(
        tok,
        y,
        keeps,
        6,
        lambda c, ci, ti: goal(ti).mean(1, keepdim=True).expand_as(goal(ti)),
    )
    assert list(exact) == [
        "within",
        "between",
        "temporal",
        "pred0",
        "ncomp0",
        "pred1",
        "ncomp1",
    ]
    for vi in range(len(keeps)):
        assert float(exact[f"pred{vi}"]) == 0.0
        assert float(exact[f"ncomp{vi}"]) == 0.0
        torch.testing.assert_close(
            const[f"ncomp{vi}"], torch.ones((), dtype=torch.float64)
        )
