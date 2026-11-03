"""The decoder learned afterwards: its shapes, that it learns an image $z$
determines and reaches but cannot beat the mean image when $z$ says nothing,
the chunks of the run's stream it learns on (those the run showed), and the
files ``decode_run`` writes from tiny vision runs (BP and PC-ALM) made here."""

import json
from pathlib import Path

import pytest
import torch

from exp.lib.png import read_png
from exp.lib.rundir import Exit
from exp.video import loop
from exp.video.compose import compose
from exp.video.data.frames import BallFrames, TourFrames
from exp.video.data.physics import render
from exp.video.main import local_job
from exp.video.measure.posthoc import (
    SHOWN,
    Decoding,
    Parent,
    decode_run,
    decoder,
    fit_decoder,
    stream_chunks,
)
from exp.video.programs.vision import PROGRAM, Measures
from tests.video.data.test_walking_tours import numbered

TINY = [
    "+experiment=m15/balls_n1",
    "data.streams=2",
    "data.balls.size=16",
    "measure.clips=4",
    "measure.every=0.5",
    "seconds=1",
    "vision.dim=16",
    "vision.depth=2",
    "vision.heads=2",
    "vision.pred_depth=1",
    "vision.sigreg_directions=16",
    "vision.frames=2",
    "ctl.ckpt_every=0.5",
    "phase=confirm",
]


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestDecoder:
    @pytest.mark.parametrize(
        "shape", [(1, 8, 8), (3, 16, 16), (1, 32, 32), (3, 64, 64)]
    )
    def test_it_maps_z_to_frames(self, shape):
        assert decoder(5, shape)(torch.zeros(2, 5)).shape == (2, *shape)

    @pytest.mark.parametrize("shape", [(1, 4, 4), (1, 12, 12), (1, 16, 8)])
    def test_other_sides_are_refused(self, shape):
        with pytest.raises(ValueError, match="power of 2"):
            decoder(5, shape)


def _blobs(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions ``(n, 2)`` in the box and their images ``(n, 1, 8, 8)``: a
    blob of radius 0.3 at each (``render``)."""
    xy = 0.2 + 0.6 * torch.rand(n, 2, generator=gen)
    states = torch.cat([xy, torch.zeros(n, 2)], 1)
    return xy, render(states, 8, [0.3])


class TestFitDecoder:
    def test_it_learns_the_image_z_determines(self):
        """$z$ is the position of the blob: on positions it did not learn on,
        the decoder's error is a small part of the mean image's (the error of
        a decoder that ignores $z$)."""
        xy, images = _blobs(1024, _gen(1))
        dec = fit_decoder(
            xy[:768], images[:768], steps=400, batch=64, lr=3e-3, gen=_gen(2)
        )
        with torch.no_grad():
            err = float(((dec(xy[768:]) - images[768:]) ** 2).mean())
        mean = float(((images[:768].mean(0) - images[768:]) ** 2).mean())
        assert err < 0.1 * mean, (err, mean)

    def test_it_reaches_but_cannot_beat_the_mean_image_when_z_says_nothing(self):
        """$z$ independent of the images: the best decoder is the mean image,
        so on new pairs the error is not below the mean image's, and a
        decoder that learned is not far above it (an untrained one, about 0
        everywhere, is at the mean square of the images, 3 times as much)."""
        _, images = _blobs(1024, _gen(3))
        z = torch.randn(1024, 2, generator=_gen(4))
        dec = fit_decoder(z[:768], images[:768], steps=200, batch=64, gen=_gen(5))
        with torch.no_grad():
            err = float(((dec(z[768:]) - images[768:]) ** 2).mean())
        mean = float(((images[:768].mean(0) - images[768:]) ** 2).mean())
        assert 0.95 * mean < err < 1.1 * mean, (err, mean)
        assert float((images[768:] ** 2).mean()) > 1.5 * mean  # the bar means it

    def test_it_is_deterministic_and_leaves_the_global_rng(self):
        xy, images = _blobs(64, _gen(6))
        before = torch.random.get_rng_state()
        a = fit_decoder(xy, images, steps=5, batch=8, gen=_gen(7))
        assert torch.equal(torch.random.get_rng_state(), before)
        b = fit_decoder(xy, images, steps=5, batch=8, gen=_gen(7))
        for (name, p), q in zip(a.state_dict().items(), b.state_dict().values()):
            assert torch.equal(p, q), name


class TestStreamChunks:
    def test_the_balls_are_the_chunks_the_run_showed(self):
        """Each chunk is the last N frames of one stream of ``BallFrames`` of
        the run's seed, the ends spread over all the frames and streams."""
        keep = [o for o in TINY if not o.startswith(("data.streams", "vision.frames"))]
        conf = compose([*keep, "data.streams=3", "seed=4", "vision.frames=3"])
        assert conf.data.balls is not None
        source = stream_chunks(conf, 10)
        frames = BallFrames.of(conf.data.balls, 3, frame=conf.time.frame, seed=4)
        shown = torch.stack([frames.next()[0] for _ in range(30)])  # (30, 3, 1, 16, 16)
        total = 28 * 3  # full chunks: frames 2 .. 29 of 3 streams
        ends = [(2 + p // 3, p % 3) for p in (j * total // 10 for j in range(10))]
        want = torch.stack([shown[k - 2 : k + 1, i] for k, i in ends])
        assert ends[0] == (2, 0) and ends[-1][0] >= 27
        assert len(source) == 10
        assert torch.equal(source[0:10], want)
        assert torch.equal(source[3:5], want[3:5])

    def test_walking_tours_are_the_chunks_the_run_showed(self, tmp_path):
        """Read from the local copy as ``TourFrames`` streams it: each chunk is
        the last N frames of one stream, and the ends whose chunk would reach
        back past a video's first frame (a reset) are left out."""
        root = numbered(tmp_path / "d", {"A": 9, "B": 7, "H": 40}, size=8)
        conf = compose(
            [
                "+experiment=m15/wt_n8",
                "data.streams=2",
                f"data.tours.data_version={root.name}",
                "data.tours.held_out=H",
                "data.tours.size=8",
                "vision.patch=4",
                "vision.frames=3",
                "seconds=1",
            ]
        )
        with pytest.raises(ValueError, match="root"):
            stream_chunks(conf, 40)
        source = stream_chunks(conf, 40, root=root)
        tours = TourFrames(root, n=2, held_out="H")
        steps = [tours.next() for _ in range(30)]
        shown = torch.stack([s[0] for s in steps])  # (30, 2, 3, 8, 8)
        reset = torch.stack([s[2] for s in steps])  # (30, 2)
        total = 28 * 2
        ends = [(2 + p // 2, p % 2) for p in (j * total // 40 for j in range(40))]
        kept = [(k, i) for k, i in ends if not reset[k - 1 : k + 1, i].any()]
        assert 0 < len(kept) < len(ends)
        want = torch.stack([shown[k - 2 : k + 1, i] for k, i in kept])
        assert len(source) == len(kept)
        assert torch.equal(source[0 : len(source)], want)


@pytest.fixture(scope="module", params=["bp", "pc"])
def run_dir(request, tmp_path_factory) -> Path:
    """A tiny vision run of the balls (N = 2), of BP or PC-ALM, with
    checkpoints at 0.5 s and 1 s."""
    out = tmp_path_factory.mktemp(f"vision-{request.param}") / "run"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loop, "assert_free_threaded", lambda where: None)
        job = local_job([*TINY, f"learner={request.param}"], out)
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
    return out


def _evals(parent: Parent) -> dict[str, torch.Tensor]:
    sets = Measures(parent.conf).sets
    return {s: sets[s].chunks for s in ("val", "test")}


def _files(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


class TestDecodeRun:
    def test_it_writes_the_files(self, run_dir, tmp_path):
        parent = Parent.read(run_dir, device="cpu")
        out = tmp_path / "decode"
        lines = decode_run(parent, out, evals=_evals(parent), frames=40, steps=3)
        for name in ("00000.5.png", "00001.0.png", "metrics.jsonl", "config.json"):
            assert (out / name).is_file(), name
        written = [
            json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()
        ]
        assert written == lines
        assert [(x["t"], x["frame"], x["split"]) for x in lines] == [
            (0.5, 15, "val"),
            (0.5, 15, "test"),
            (1.0, 30, "val"),
            (1.0, 30, "test"),
        ]
        for x in lines:
            assert set(x) == {"t", "frame", "split", "recon", "pix_mean", "recon_fit"}
            assert all(v >= 0 for k, v in x.items() if k not in ("t", "frame", "split"))
        config = json.loads((out / "config.json").read_text())
        assert config["chunks"] == 40 and config["steps"] == 3
        assert config["parent"] == {
            "run_id": parent.job["run_id"],
            "group": parent.job["group"],
            "code": parent.job["code"],
        }
        # 2 rows (true, decoded) of SHOWN frames of 16 pixels scaled 6 times
        cell = 16 * 6 + 2
        assert read_png(out / "00001.0.png").shape == (2 * cell, SHOWN * cell, 3)

    def test_the_first_row_is_the_true_frames(self, run_dir, tmp_path):
        """The PNG's first row is the last frames of the first val chunks, as
        the evaluation has them (``Measures``)."""
        parent = Parent.read(run_dir, device="cpu")
        evals = _evals(parent)
        decode_run(parent, tmp_path, evals=evals, frames=40, steps=1)
        x = evals["val"][:SHOWN, -1]  # (SHOWN, 1, 16, 16)
        image = read_png(tmp_path / "00000.5.png")
        cell = 16 * 6 + 2
        for j in range(SHOWN):
            tile = image[1 : cell - 1 : 6, j * cell + 1 : (j + 1) * cell - 1 : 6, 0]
            want = (x[j, 0].clamp(0, 1) * 255).round().to(torch.uint8)
            assert torch.equal(tile, want), j

    def test_a_decoding_resumed_writes_the_bytes_of_one_not_stopped(
        self, run_dir, tmp_path
    ):
        """The same run decodes to the same bytes; one stopped after the first
        checkpoint (and a line of the second cut short, as a crash leaves it)
        and started again at ``done=1`` writes the same files."""
        parent = Parent.read(run_dir, device="cpu")
        evals = _evals(parent)
        whole = []
        for k in range(2):
            decode_run(parent, tmp_path / f"whole{k}", evals=evals, frames=40, steps=3)
            whole.append(_files(tmp_path / f"whole{k}"))
        assert whole[0] == whole[1]
        out = tmp_path / "stopped"
        first = Decoding(parent, out, evals=evals, frames=40, steps=3)
        first.step()
        with (out / "metrics.jsonl").open("a") as f:
            f.write('{"t": 1.0, "fra')
        again = Decoding(parent, out, evals=evals, frames=40, steps=3, done=1)
        again.step()
        assert again.done == len(again) == 2
        assert _files(out) == whole[0]

    def test_a_run_without_checkpoints_is_refused(self, run_dir, tmp_path):
        (tmp_path / "job.json").write_bytes((run_dir / "job.json").read_bytes())
        with pytest.raises(ValueError, match="no evaluation checkpoint"):
            Parent.read(tmp_path, device="cpu")
