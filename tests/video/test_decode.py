import json
import math
import struct
import zlib
from dataclasses import replace

import pytest
import torch

from exp.video import decode, run
from exp.video.config import Config, DataConfig


class TestDecoder:
    @pytest.mark.parametrize("shape", [(1, 32, 32), (3, 64, 64)])
    def test_it_makes_frames_of_the_shape(self, shape):
        dec = decode.decoder(8, shape)
        assert dec(torch.zeros(5, 8)).shape == (5, *shape)

    def test_it_learns_frames_that_z_determines(self):
        """Frames that are a fixed linear image of z: the fitted decoder's
        error falls far below the variance of the frames."""
        g = torch.Generator().manual_seed(1)
        z = torch.randn(512, 4, generator=g)
        basis = torch.rand(4, 1, 8, 8, generator=g)
        images = torch.einsum("nk,kchw->nchw", z, basis)
        dec = decode.fit_decoder(z, images, steps=400)
        with torch.no_grad():
            err = float(((dec(z) - images) ** 2).mean())
        assert err < 0.05 * float(images.var())


class TestImage:
    def test_the_png_holds_the_pixels(self, tmp_path):
        """Read back by hand: the signature, the size in IHDR, and the filtered
        rows (filter 0) in IDAT."""
        g = torch.Generator().manual_seed(0)
        image = torch.randint(0, 256, (3, 5, 3), generator=g, dtype=torch.uint8)
        path = tmp_path / "a.png"
        decode.write_png(path, image)
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        w, h = struct.unpack(">II", data[16:24])
        assert (w, h) == (5, 3)
        start = data.index(b"IDAT") + 4
        (length,) = struct.unpack(">I", data[start - 8 : start - 4])
        raw = zlib.decompress(data[start : start + length])
        rows = [raw[i * 16 : (i + 1) * 16] for i in range(3)]
        assert all(r[0] == 0 for r in rows)
        assert [list(r[1:]) for r in rows] == image.reshape(3, 15).tolist()

    def test_the_tile_places_each_frame_with_a_gap(self):
        frames = torch.zeros(2, 3, 1, 4, 4)
        frames[1, 2] = 1.0
        tile = decode._tile(frames, scale=2)
        assert tile.shape == (2 * 10, 3 * 10, 3)
        assert tile[0, 0, 0] == 128  # the gap
        assert (tile[11:19, 21:29] == 255).all()  # frame (1, 2), inside its gap
        assert (tile[1:9, 1:9] == 0).all()


def _decoded(tmp_path, device: str) -> dict[str, float]:
    config = replace(
        Config(name="dec", eval_interval=1.0, checkpoint_interval=1.0),
        data=DataConfig(kind="video", seconds=2.0),
    )
    path = tmp_path / "c.json"
    path.write_text(config.to_json())
    out = run.main(["--config", str(path), "--out", str(tmp_path)])
    snap = run.load_checkpoint(out / "ckpt" / "00002.pt", device)
    train = decode.stream_frames(config, 256).to(device)
    _, clips = run.source(config)
    clips = clips._replace(
        image=clips.image.to(device),
        state=clips.state.to(device),
        params=clips.params.to(device),
    )
    line = decode.decode_checkpoint(snap, config, clips, train, tmp_path, 2, steps=5)
    assert (tmp_path / "00002.png").stat().st_size > 0
    return line


class TestDecodeCheckpoint:
    @pytest.mark.slow
    def test_a_run_with_checkpoints_decodes(self, tmp_path):
        line = _decoded(tmp_path, "cpu")
        assert all(math.isfinite(v) for v in line.values())
        assert line["img_rollout10"] > 0
        json.dumps(line)

    @pytest.mark.slow
    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_it_decodes_on_the_gpu(self, tmp_path):
        line = _decoded(tmp_path, "cuda:0")
        assert all(math.isfinite(v) for v in line.values())
