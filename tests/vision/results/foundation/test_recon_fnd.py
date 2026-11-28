"""``exp.vision.results.foundation.recon_fnd``: the readout is the content's.

The oracles: on a code that holds a part of its observation, the readout's
error on the evaluation equals ``fnd_guard.content_sums``'s (the same ridge
and the same choice of $\\alpha$); the grid puts each image at its cell.
"""

import numpy as np
import pytest
import torch

from exp.vision.fnd_guard import ContentSet, content_sums
from exp.vision.results.foundation.recon_fnd import grid, readout


def _set(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    y = torch.rand(n, 12, generator=g)
    w = y[:, :5] + 0.1 * torch.randn(n, 5, generator=g)
    return w, y


def test_the_readout_is_the_contents():
    (wt, yt), (wv, yv), (we, ye) = _set(400, 0), _set(200, 1), _set(300, 2)
    bounds = lambda n: torch.tensor([0, n])  # noqa: E731
    row = content_sums(ContentSet(wt, yt, bounds(400)), ContentSet(wv, yv, bounds(200)),
                       ContentSet(we, ye, bounds(300), torch.zeros(300, dtype=torch.long), 1))  # fmt: skip
    a, b = readout(wt, yt, wv, yv)
    err = float((we.double() @ a.T + b - ye.double()).square().sum())
    assert err == pytest.approx(sum(row["err_b"]), rel=1e-9)


def test_the_grid_places_each_image():
    size = 4
    rows = [torch.zeros(3, 3 * size * size), torch.ones(3, 3 * size * size)]
    rows[0][1] = 0.5
    arr = torch.from_numpy(np.asarray(grid(rows, size, scale=1, gap=1)))
    assert arr.shape == (2 * size + 1, 3 * size + 2, 3)
    assert int(arr[0, 0, 0]) == 0
    assert int(arr[0, size + 1, 0]) == 128
    assert int(arr[size + 1, 0, 0]) == 255
    assert int(arr[size, 0, 0]) == 255  # the gap
