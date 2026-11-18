r"""The mean color of each tubelet of a stack (M17's check of the evaluation,
results/m17.md 「評価の確認：平均の色の結果」): a code of the data alone, no
learned feature, $D = T C$. M17 found that the untrained ViT's ``recon_blur``
is about that of this code, so the evaluations of M18 show it beside U
(``vae.eval``, ``diag.mean_color_ref``)."""

from torch import Tensor

__all__ = ["tubelet_means"]


def tubelet_means(x: Tensor, patch: int, tubelet: int) -> Tensor:
    """The ``tube`` code ``(n, T C)`` of stacks ``x`` ``(n, N, C, H, W)``
    float: the mean color of each tubelet, the tubelets in
    ``patchify_tubelets``' order (time slot, row, column), the $C$ colors of
    one together."""
    n, f, c, h, w = x.shape
    gt, gh, gw = f // tubelet, h // patch, w // patch
    m = x.reshape(n, gt, tubelet, c, gh, patch, gw, patch).mean((2, 5, 7))
    return m.permute(0, 1, 3, 4, 2).reshape(n, gt * gh * gw * c)  # (n, gt gh gw C)
