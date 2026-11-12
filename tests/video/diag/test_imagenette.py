"""The sanity benchmark's parts (diagnosis code, M17): LeJEPA's loss by hand,
the probe on separable and on random labels, the views, the decoding, and a
few steps of each mode."""

import json
import subprocess

import imageio_ffmpeg
import pytest
import torch

from exp.video.diag.imagenette import (
    augment,
    decode_jpeg,
    lejepa_loss,
    main,
    probe_accuracy,
)
from exp.video.jepa.loss import weighted_sigreg


def test_lejepa_loss_by_hand():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(3, 8, 4, generator=g, dtype=torch.float64)
    dirs = torch.nn.functional.normalize(torch.randn(5, 4, generator=g), dim=1)
    dirs = dirs.double()
    loss, s = lejepa_loss(u, dirs, 0.25)
    ones = torch.ones(8, dtype=torch.float64)
    want_s = sum(weighted_sigreg(v, ones, dirs) for v in u) / 3
    pred = ((u.mean(0, keepdim=True) - u) ** 2).mean()
    torch.testing.assert_close(s, want_s)
    torch.testing.assert_close(loss, 0.75 * pred + 0.25 * want_s)
    same = u[:1].expand(3, 8, 4)  # views that agree: SIGReg alone
    loss_same, s_same = lejepa_loss(same, dirs, 0.25)
    torch.testing.assert_close(loss_same, 0.25 * s_same)


def test_the_probe_reads_separable_labels_and_not_random_ones():
    g = torch.Generator().manual_seed(1)
    y = torch.randint(3, (600,), generator=g)
    centres = 4 * torch.randn(3, 5, generator=g)
    f = centres[y] + torch.randn(600, 5, generator=g)
    assert probe_accuracy(f[:400], y[:400], f[400:], y[400:], 3) > 0.95
    noise = torch.randn(600, 5, generator=g)
    assert probe_accuracy(noise[:400], y[:400], noise[400:], y[400:], 3) < 0.5


def test_a_view_is_of_the_shape_and_range_and_fixed_by_the_generator():
    x = torch.rand(4, 3, 32, 32, generator=torch.Generator().manual_seed(2))
    a = augment(x, torch.Generator().manual_seed(3), side=16)
    b = augment(x, torch.Generator().manual_seed(3), side=16)
    assert a.shape == (4, 3, 16, 16) and torch.equal(a, b)
    assert float(a.min()) >= 0 and float(a.max()) <= 1


def test_decode_resizes_the_shorter_side_and_crops_the_centre(tmp_path):
    """A 40 x 20 picture (left half black, right half white) of PNG: the
    shorter side to 16, the centre 16 x 16 cut from 32 x 16."""
    img = torch.zeros(20, 40, 3, dtype=torch.uint8)
    img[:, 20:] = 255
    path = tmp_path / "a.png"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "rawvideo"]
    cmd += ["-pix_fmt", "rgb24", "-s", "40x20", "-i", "-", str(path)]
    subprocess.run(cmd, input=bytes(img.flatten().tolist()), check=True)
    x = decode_jpeg(path, 16)
    assert x.shape == (3, 16, 16)
    assert int(x[:, :, 0].max()) < 30 and int(x[:, :, -1].min()) > 225


@pytest.mark.parametrize(
    "mode",
    [
        ["--mode", "lejepa"],
        ["--mode", "masked"],
        ["--mode", "masked", "--objective", "z"],
    ],
)
def test_a_few_steps_write_the_log(tmp_path, mode):
    g = torch.Generator().manual_seed(4)
    for split, n in (("train", 20), ("val", 10)):
        x = torch.randint(0, 256, (n, 3, 128, 128), generator=g).to(torch.uint8)
        y = torch.arange(n) % 2
        torch.save({"x": x, "y": y, "classes": ["a", "b"]}, tmp_path / f"{split}.pt")
    args = ["train", "--data", str(tmp_path), "--out", str(tmp_path / "run")]
    args += ["--device", "cpu", "--steps", "2", "--batch", "4", "--views", "2"]
    args += ["--every", "1", "--warmup", "1", "--projector", "32", "8", *mode]
    args += ["--set", "vision.dim=32", "vision.depth=2", "vision.nodes=1"]
    args += ["vision.heads=2", "vision.pred_depth=1", "vision.sigreg_directions=8"]
    main(args)
    lines = [
        json.loads(s) for s in (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in lines] == [0, 1, 2]
    assert all(0 <= r["probe"] <= 1 for r in lines)
    assert all(r["loss"] == r["loss"] for r in lines[1:])  # finite, not nan
