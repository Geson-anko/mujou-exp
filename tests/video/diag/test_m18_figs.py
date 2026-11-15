"""The pictures of M18 stage 1 (diagnosis): an autoencoder that returns its
target draws the true frames again, a tiled image splits back into its rows,
and the figure stacks the decoded rows under the one true row."""

from pathlib import Path

import pytest
import torch

from exp.lib.png import tile, write_png
from exp.video.compose import compose
from exp.video.diag.m18_figs import (
    ROWS,
    chroma,
    figure,
    frames_of,
    own_rows,
    patch_means,
    picture,
    prior_rows,
    split_rows,
    stack_pictures,
    stats,
)
from exp.video.vae.loss import target
from exp.video.vae.modules import build_vae
from tests.video.vae.test_program import tiny_vae


def _stacks(n: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, 4, 3, 8, 8), generator=g, dtype=torch.uint8)


def test_the_identity_autoencoder_draws_the_true_frames():
    rows = own_rows(target, _stacks())  # y_hat = x / 256 is x / 255 drawn
    image = picture(rows)
    true, own = split_rows(image, 2)
    assert torch.equal(true, own)


def test_split_rows_is_the_tile_of_each_row():
    frames = torch.rand(3, 2, 3, 8, 8)
    parts = split_rows(picture(frames), 3)
    for i, part in enumerate(parts):
        assert torch.equal(part, picture(frames[i : i + 1]))
    with pytest.raises(ValueError, match="rows"):
        split_rows(picture(frames), 4)  # 3 rows of 98 px are not 4


def test_figure_stacks_the_decoded_rows_and_checks_the_truth(tmp_path: Path):
    truth = torch.rand(1, 2, 3, 8, 8)
    post, base = tmp_path / "post", tmp_path / "base"
    post.mkdir()
    base.mkdir()
    names = [tmp_path / "init-own.png", post / "t-own.png", base / "init.png"]
    names += [base / "tube.png", post / "t.png"]
    decoded = [torch.full_like(truth, i / 5) for i in range(5)]
    for p, d in zip(names, decoded, strict=True):
        write_png(p, picture(torch.cat([truth, d])))
    image = figure(post, base, names[0], "t")
    assert torch.equal(image, picture(torch.cat([truth, *decoded])))
    assert len(split_rows(image, len(ROWS))) == len(ROWS)
    write_png(names[3], picture(torch.cat([1 - truth, decoded[3]])))
    with pytest.raises(ValueError, match="true frames"):
        figure(post, base, names[0], "t")


def test_frames_of_inverts_the_tile_of_a_row():
    frames = torch.rand(1, 3, 3, 8, 8)  # (rows, cols, C, H, W)
    back = frames_of(tile(frames, scale=1), 8)  # (cols, H, W, C)
    u8 = (frames[0] * 255).round() / 255
    assert torch.equal(back, u8.permute(0, 2, 3, 1))


def test_chroma_is_zero_for_grey_and_patch_means_average_each_patch():
    grey = (torch.randint(0, 5, (2, 4, 1)) / 4).expand(2, 4, 3)  # exact means
    assert chroma(grey) == 0.0
    assert chroma(torch.tensor([[0.0, 0.0, 0.3]])) == pytest.approx(0.4 / 3)
    x = torch.zeros(1, 4, 4, 3)
    x[0, :2, 2:] = 1.0  # the top-right 2 x 2 patch
    m = patch_means(x, 2)
    assert m.shape == (1, 2, 2, 3)
    assert m[0, 0, 1].tolist() == [1.0, 1.0, 1.0] and m.sum() == 3.0


def test_stats_of_a_figure_whose_rows_are_the_truth():
    g = torch.Generator().manual_seed(1)
    truth = torch.rand(1, 3, 3, 64, 64, generator=g)
    image = tile(torch.cat([truth] * len(ROWS)), scale=1)
    first = frames_of(split_rows(image, len(ROWS))[0], 64)
    for _, c, r in stats(image, 64):
        assert c == pytest.approx(chroma(patch_means(first)))
        assert r == pytest.approx(1.0)


def test_stack_pictures_keeps_the_given_order(tmp_path: Path):
    truth = torch.rand(1, 2, 3, 8, 8)
    decoded = [torch.full_like(truth, i / 3) for i in range(3)]
    paths = [tmp_path / f"{i}.png" for i in range(3)]
    for p, d in zip(paths, decoded, strict=True):
        write_png(p, picture(torch.cat([truth, d])))
    order = [2, 0, 1]
    image = stack_pictures([paths[i] for i in order])
    want = torch.cat([truth, *(decoded[i] for i in order)])
    assert torch.equal(image, picture(want))


def test_prior_rows_are_the_decoder_at_standard_normal_codes():
    """The last frame of g(z), z the standard normal draws of the generator
    in the code's shape, in the units of own_rows."""
    conf = compose(tiny_vae("unused"))
    mods = build_vae(conf, seed=conf.seed)
    got = prior_rows(mods, 3, gen=torch.Generator().manual_seed(5))
    z = torch.randn(3, *mods.code_shape, generator=torch.Generator().manual_seed(5))
    want = (mods.decoder(z)[:, -1] * (256 / 255)).clamp(0, 1)
    assert got.shape == (1, *want.shape)
    torch.testing.assert_close(got[0], want.detach(), rtol=0, atol=0)
