import torch

from exp.video.jepa.vit import patchify_tubelets
from exp.video.measure.mean_color import tubelet_means

N, C, S, P, TAU = 4, 3, 8, 4, 2  # 2 slots x 2 x 2 tubelets of 2 frames x 4 x 4


def test_the_tube_code_is_the_mean_of_each_vit_tubelet():
    """Token t of ``patchify_tubelets`` (its tubelet's (tau, C, p, p) values)
    averaged over its frames and pixels is the code's colors t C .. t C + C."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(5, N, C, S, S, generator=g, dtype=torch.float64)
    tok = patchify_tubelets(x, P, TAU).reshape(len(x), -1, TAU, C, P, P)
    want = tok.mean((2, 4, 5)).flatten(1)
    torch.testing.assert_close(tubelet_means(x, P, TAU), want, rtol=0, atol=1e-12)
