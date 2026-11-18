"""Tests of exp.vision.eval on tiny synthetic Walking Tours (16 x 16 frames,
stacks of 2): the oracles are the formulas of the module docstring."""

import json
import statistics
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.blur_error import blur_recon
from exp.lib.png import read_png
from exp.lib.rundir import Exit
from exp.lib.tours.stacks import DATA_ENV
from exp.lib.tours.walking_tours import finalize
from exp.lib.video_io import U8Writer
from exp.vision import loop
from exp.vision.compose import compose, to_dict
from exp.vision.eval import (
    SCALE,
    Evaluator,
    check_splits,
    eval_splits,
    evaluate_run,
    fit_decoder,
    main,
    stream_windows,
)
from exp.vision.models import build_vae
from exp.vision.program import VAEProgram
from exp.vision.schema import DataConf, MeasureConf, ModelConf, RunConf, ToursConf
from tests.lib.tours.test_walking_tours import numbered

SIZE, N = 16, 2  # the frames' side and the stack
# Training videos A, B, C of 20 frames: 8 windows of N + 2 frames give 7 (one
# crosses the end of B, TestStreamWindows), of 3 stacks each.
WINDOWS, STEPS, FIT = 8, 2, 3


def conf_of(
    *, latent: int = 4, seeds: int = 2, size: int = SIZE, clips: int = 3
) -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=size, frames=N)),
        model=ModelConf(latent=latent),
        measure=MeasureConf(clips=clips, windows=1, posthoc_seeds=seeds),
    )


def write_tours(root: Path, frames: Mapping[str, Tensor]) -> Path:
    """Finalized data of the u8 frames ``(L, 3, SIZE, SIZE)`` of each video."""
    for name, x in sorted(frames.items()):
        meta = {"video": name, "fps": 10.0, "size": SIZE}
        with U8Writer(root / name, (3, SIZE, SIZE), meta) as w:
            for f in x:
                w.append(f)
    return finalize(root, expected=list(frames))


def uniform(colors: Tensor) -> Tensor:
    """Frames ``(L, 3, SIZE, SIZE)`` u8 of one color ``(L, 3)`` each."""
    return colors.to(torch.uint8)[:, :, None, None].expand(-1, -1, SIZE, SIZE)


def ramp(start: tuple[int, int, int], step: tuple[int, int, int], n: int) -> Tensor:
    """``n`` frames of the colors ``start + j step``."""
    j = torch.arange(n)[:, None]
    return uniform(torch.tensor(start) + j * torch.tensor(step))


def random_tours(root: Path) -> Path:
    g = torch.Generator().manual_seed(0)
    lengths = {"A": 20, "B": 20, "C": 20, "H": 40}
    return write_tours(
        root,
        {
            v: torch.randint(256, (n, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
            for v, n in lengths.items()
        },
    )


def brute_mean_stack(conf: RunConf, root: Path) -> Tensor:
    """The mean training stack ``(N, 3, SIZE, SIZE)`` in ``x / 255``, from
    every stack of every training window one at a time."""
    windows = stream_windows(conf, root, windows=WINDOWS, steps=STEPS)
    stacks = [w[k : k + N] for w in windows for k in range(len(w) - N + 1)]
    return torch.stack(stacks).double().mean(0) / 255


class Mix(nn.Module):
    """The weight ``c`` of ``MixCode``."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.zeros(()))


class MixCode:
    """A ``VaeCode`` whose code is the stack itself, ``x / 255`` flattened, and
    whose reconstruction is $x + c (m - x)$ with $m$ a given stack: its
    ``stack_mse`` against the mean stack $m$ is $c^2$."""

    def __init__(self, m: Tensor) -> None:
        self.net, self._m = Mix(), m.float()

    def modules(self) -> Mapping[str, nn.Module]:
        return {"mix": self.net}

    def mu(self, x: Tensor) -> Tensor:
        return x.flatten(1).float() / 255

    def decode(self, mu: Tensor) -> Tensor:
        x = mu.view(-1, N, 3, SIZE, SIZE)
        return x + self.net.c * (self._m - x)


class ColorCode:
    """A ``VaeCode`` of $D$ = 2: the mean of the channels 0 and 1 of a
    stack's last frame (``x / 255``); its reconstruction is that color."""

    def modules(self) -> Mapping[str, nn.Module]:
        return {}

    def mu(self, x: Tensor) -> Tensor:
        return x[:, -1, :2].float().mean((-2, -1)) / 255

    def decode(self, mu: Tensor) -> Tensor:
        rgb = torch.cat([mu, torch.zeros_like(mu[:, :1])], 1)
        return rgb[:, None, :, None, None].expand(-1, N, -1, SIZE, SIZE)


def frame_numbers(x: Tensor) -> Tensor:
    """The frame numbers of the frames of ``numbered`` ``(..., 3, H, W)``."""
    px = x[..., 0, 0].long()
    return px[..., 1] + 256 * px[..., 2]


class TestEvalSplits:
    @pytest.fixture
    def root(self, tmp_path) -> Path:
        return numbered(tmp_path / "d", {"A": 20, "H": 40}, size=SIZE)

    def test_thirds_of_the_held_out_video_disjoint_and_in_time_order(self, root):
        splits = eval_splits(conf_of(), root)
        assert list(splits) == ["calib", "val", "test"]
        for x in splits.values():
            assert x.shape == (3, N + 1, 3, SIZE, SIZE) and x.dtype == torch.uint8
            assert (x[..., 0, 0, 0] == 1).all()  # video H only (A is 0)
        j = [frame_numbers(splits[s]) for s in ("calib", "val", "test")]
        for x in j:
            assert (x[:, 1:] - x[:, :-1] == 1).all()  # consecutive frames
        # every frame of a split before every frame of the next: in time order
        # and no frame shared
        assert j[0].max() < j[1].min() and j[1].max() < j[2].min()

    def test_the_seed_fixes_the_windows(self, root):
        a, b = eval_splits(conf_of(), root), eval_splits(conf_of(), root)
        assert all(torch.equal(a[s], b[s]) for s in a)

    def test_frames_of_another_size_raise(self, root):
        with pytest.raises(ValueError, match="pixels"):
            eval_splits(conf_of(size=32), root)

    @pytest.mark.parametrize("clips", [3, 4, 5])
    def test_check_splits_says_whether_eval_splits_can(self, root, clips):
        """H's 40 frames are 13 windows of N + 1 = 3 frames: 3 splits of 4
        fit, of 5 do not."""
        conf = conf_of(clips=clips)
        if 3 * clips <= 40 // (N + 1):
            check_splits(conf, root)
            assert len(eval_splits(conf, root)["test"]) == clips
        else:
            with pytest.raises(ValueError, match="measure.clips"):
                check_splits(conf, root)
            with pytest.raises(ValueError, match="clips"):
                eval_splits(conf, root)


class TestStreamWindows:
    def test_windows_start_at_j_l_over_windows_within_one_video(self, tmp_path):
        root = numbered(tmp_path / "d", {"A": 20, "B": 20, "C": 20, "H": 40}, size=SIZE)
        x = stream_windows(conf_of(), root, windows=WINDOWS, steps=STEPS)
        # The loop of 60 frames: starts 60 j // 8 = 0, 7, 15, 22, 30, 37, 45,
        # 52; the window at 37 (B's frame 17) would cross the end of B.
        video = x[:, :, 0, 0, 0].long()
        j = frame_numbers(x)
        assert video[:, 0].tolist() == [0, 0, 0, 1, 1, 2, 2]
        assert j[:, 0].tolist() == [0, 7, 15, 2, 10, 5, 12]
        assert (video == video[:, :1]).all()
        assert (j[:, 1:] - j[:, :-1] == 1).all()
        assert x.shape[1] == N + STEPS


class TestStackMse:
    @pytest.mark.parametrize("c", [0.0, 0.5, 1.0])
    def test_the_stack_mixed_with_the_mean_stack_by_c_gives_c_squared(
        self, tmp_path, c
    ):
        root = random_tours(tmp_path / "d")
        conf = conf_of()
        code = MixCode(brute_mean_stack(conf, root))
        code.net.c.data.fill_(c)
        ev = Evaluator(
            conf, root, tmp_path / "out", windows=WINDOWS, steps=STEPS, fit_steps=FIT
        )
        (row,) = ev.evaluate(code, "x")
        assert row["code"] == "vae" and row["split"] == "val"
        assert row["stack_mse"] == pytest.approx(c * c, rel=1e-5, abs=1e-12)


class TestDynamics:
    def evaluate(self, tmp_path, frames: Mapping[str, Tensor]) -> dict:
        root = write_tours(tmp_path / "d", frames)
        ev = Evaluator(
            conf_of(),
            root,
            tmp_path / "out",
            windows=WINDOWS,
            steps=STEPS,
            fit_steps=FIT,
        )
        (row,) = ev.evaluate(ColorCode(), "x")
        return row

    def test_still_training_videos_give_the_copy(self, tmp_path):
        # z_{k+1} = z_k on every training pair: A = I up to the ridge, which
        # shrinks the prediction toward the training mean (120, 97) / 255. The
        # held-out codes drift away from it, so the calib windows choose the
        # least ridge, and the prediction is the copy: dyn_1 / copy_1 = 1.
        still = {
            v: uniform(torch.tensor([c] * 20))
            for v, c in {
                "A": (10, 20, 0),
                "B": (250, 30, 0),
                "C": (100, 240, 0),
            }.items()
        }
        row = self.evaluate(tmp_path, still | {"H": ramp((60, 200, 0), (-1, 1, 0), 40)})
        assert row["copy_1"] > 0
        assert row["dyn_1/copy_1"] == pytest.approx(1.0, abs=1e-4)

    def test_a_linear_drift_is_predicted_exactly(self, tmp_path):
        # z_{k+1} = z_k + d on the training pairs and on the held-out video:
        # A = I, b = d, so dyn_1 = 0 while the copy is off by d.
        d = (2, -3, 0)
        frames = {
            "A": ramp((10, 100, 0), d, 20),
            "B": ramp((100, 250, 0), d, 20),
            "C": ramp((50, 180, 0), d, 20),
            "H": ramp((60, 200, 0), d, 40),
        }
        row = self.evaluate(tmp_path, frames)
        assert row["copy_1"] > 1e-3
        assert row["dyn_1"] < 1e-6 * row["copy_1"]


class TestPcaBaseline:
    def test_the_linear_decoder_is_the_closed_form_projection(self, tmp_path):
        # Frames near a 6-dimensional affine subspace (u8 rounding off it),
        # of decreasing scales: the PCA of 3 components is the exact SVD's.
        g = torch.Generator().manual_seed(1)
        basis = torch.randn(6, 3 * SIZE * SIZE, generator=g)
        scale = torch.tensor([40.0, 25.0, 15.0, 8.0, 4.0, 2.0])

        def video(n: int) -> Tensor:
            a = torch.randn(n, 6, generator=g) * scale
            x = 128 + a @ basis
            return x.round().clamp(0, 255).to(torch.uint8).view(n, 3, SIZE, SIZE)

        lengths = {"A": 20, "B": 20, "C": 20, "H": 40}
        root = write_tours(tmp_path / "d", {v: video(n) for v, n in lengths.items()})
        conf = conf_of(latent=3)
        ev = Evaluator(
            conf, root, tmp_path / "out", windows=WINDOWS, steps=STEPS, fit_steps=FIT
        )
        (row,) = ev.baseline()
        # the oracle: the exact SVD of the training stacks' last frames
        windows = stream_windows(conf, root, windows=WINDOWS, steps=STEPS)
        t = windows[:, N - 1 :].flatten(0, 1).flatten(1).double() / 255
        mean = t.mean(0)
        _, _, vh = torch.linalg.svd(t - mean, full_matrices=False)
        w = vh[:3]
        f0 = eval_splits(conf, root)["val"][:, N - 1].double() / 255
        x = f0.flatten(1)
        lin = (((x - mean) @ w.T) @ w + mean).clamp(0, 1).view_as(f0)
        m = mean.view(3, SIZE, SIZE)
        recon_lin = float(((lin - f0) ** 2).mean() / ((m - f0) ** 2).mean())
        assert row["code"] == "pca"
        assert row["recon_lin"] == pytest.approx(recon_lin, rel=1e-4)
        assert row["recon_lin_blur"] == pytest.approx(
            blur_recon(lin, f0, m, conf.measure.blur_sigma), rel=1e-4
        )
        assert row["rank"] <= 3 + 1e-9


class TestFitDecoder:
    @pytest.mark.slow
    def test_a_pca_code_decodes_near_the_closed_form_projection(self):
        # Frames on a 3-dimensional affine subspace (up to u8 rounding), coded
        # by their exact PCA of 3 components: the best linear decoder of the
        # code is the projection, so the decoder learned at the default steps
        # (here batches of 64, for time) should come near its error. Seeds 1-3
        # measured 1.2-1.4 times it; a bad rate or a broken fit is >= 2x.
        g = torch.Generator().manual_seed(1)
        basis = torch.randn(3, 3 * SIZE * SIZE, generator=g)
        scale = torch.tensor([40.0, 25.0, 15.0])

        def frames(n: int) -> Tensor:
            x = 128 + (torch.randn(n, 3, generator=g) * scale) @ basis
            return x.round().clamp(0, 255).to(torch.uint8).view(n, 3, SIZE, SIZE)

        train, val = frames(512), frames(128)
        t = train.flatten(1).double() / 255
        mean = t.mean(0)
        _, _, vh = torch.linalg.svd(t - mean, full_matrices=False)
        w = vh[:3]
        z_train, z_val = (
            (x.flatten(1).double() / 255 - mean) @ w.T for x in (train, val)
        )
        sd = float(z_train.var(0).mean().sqrt())  # one scalar, as the evaluator
        f = val.double() / 255
        base = float(((mean.view(3, SIZE, SIZE) - f) ** 2).mean())
        lin = (z_val @ w + mean).clamp(0, 1).view_as(f)
        closed = float(((lin - f) ** 2).mean()) / base
        dec = fit_decoder(
            (z_train / sd).float(),
            train,
            batch=64,
            gen=torch.Generator().manual_seed(0),
        )
        with torch.no_grad():
            x_hat = dec((z_val / sd).float()).clamp(0, 1).double()
        recon = float(((x_hat - f) ** 2).mean()) / base
        assert closed < 0.02
        assert recon < 2 * closed


class TestSeeds:
    @pytest.fixture
    def root(self, tmp_path) -> Path:
        return random_tours(tmp_path / "d")

    def row(self, root: Path, out: Path, seeds: int) -> dict:
        ev = Evaluator(
            conf_of(seeds=seeds), root, out, windows=WINDOWS, steps=STEPS, fit_steps=FIT
        )
        (row,) = ev.evaluate(ColorCode(), "x")
        return row

    def test_the_same_seeds_give_the_same_row(self, root, tmp_path):
        assert self.row(root, tmp_path / "a", 2) == self.row(root, tmp_path / "b", 2)

    def test_seed_k_does_not_depend_on_the_number_of_seeds(self, root, tmp_path):
        one, two = self.row(root, tmp_path / "a", 1), self.row(root, tmp_path / "b", 2)
        assert two["recon_blur_k"][0] == one["recon_blur"] == one["recon_blur_k"][0]
        assert two["recon_blur_k"][1] != two["recon_blur_k"][0]
        assert two["recon_blur"] == pytest.approx(statistics.fmean(two["recon_blur_k"]))
        assert two["recon_blur_sd"] == pytest.approx(
            statistics.stdev(two["recon_blur_k"])
        )
        assert one["recon_blur_sd"] != one["recon_blur_sd"]  # NaN of one seed


class TestEvaluateRun:
    def test_every_checkpoint_in_time_order_with_its_weights(self, tmp_path):
        root = random_tours(tmp_path / "d")
        conf = conf_of()
        run = tmp_path / "run"
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        m = brute_mean_stack(conf, root)
        saved = MixCode(m)
        for t, c in ((60.0, 0.0), (0.0, 1.0)):  # saved out of order
            saved.net.c.data.fill_(c)
            ckpt.save_eval(
                run,
                program="vae",
                group="g",
                t=t,
                frame=round(t / 0.1),
                modules=saved.modules(),
            )
        out = evaluate_run(
            run,
            lambda _: MixCode(m),
            root=root,
            windows=WINDOWS,
            steps=STEPS,
            fit_steps=FIT,
        )
        rows = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines()]
        assert [(r["t"], r["frame"]) for r in rows] == [(0.0, 0), (60.0, 600)]
        # c = 1 at t = 0 is the mean stack, c = 0 at t = 60 the stack itself
        assert rows[0]["stack_mse"] == pytest.approx(1.0, rel=1e-5)
        assert rows[1]["stack_mse"] == 0.0
        (base,) = (
            json.loads(x) for x in (out / "baselines.jsonl").read_text().splitlines()
        )
        assert base["code"] == "pca" and base["split"] == "val"
        assert json.loads((out / "config.json").read_text())["ckpts"] == [
            "00000.0.pt",
            "00060.0.pt",
        ]
        # 3 rows (true, system, posthoc) of the 3 val windows, scaled by SCALE
        side = SIZE * SCALE + 2
        png = read_png(out / "00060.0.png")
        assert png.shape == (3 * side, 3 * side, 3)
        assert torch.equal(png[:side], png[side : 2 * side])  # g = identity
        png0 = read_png(out / "00000.0.png")  # g = the mean stack
        assert not torch.equal(png0[:side], png0[side : 2 * side])

    def test_a_run_without_checkpoints_raises(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf_of())))
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            evaluate_run(run, lambda _: MixCode(torch.zeros(())), root=tmp_path)


class TestCommandLine:
    def test_a_tiny_run_then_its_posthoc(self, tmp_path, monkeypatch):
        """End to end on synthetic Walking Tours of 16 px: the program "vae"
        runs v0/tiny from the data's stream (checkpoints at 0, 0.5 and 1 s),
        then ``python -m exp.vision.eval`` evaluates each. The row of t = 0
        is that of a VAE freshly built from the run's seed; the last one's
        weights have moved."""
        monkeypatch.setattr(loop, "assert_free_threaded", lambda where: None)
        g = torch.Generator().manual_seed(0)
        lengths = {"A": 30, "B": 30, "C": 30, "Zurich": 40}
        root = write_tours(  # renamed to its data version
            tmp_path / "data" / "new",
            {
                v: torch.randint(256, (n, 3, SIZE, SIZE), generator=g).byte()
                for v, n in lengths.items()
            },
        )
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        conf = compose(
            [
                "+experiment=v0/tiny",
                f"data.tours.size={SIZE}",
                f"data.tours.data_version={root.name}",
            ]
        )
        run = tmp_path / "run"
        run.mkdir()
        job = {
            "run_id": "vae-s0",
            "group": "g",
            "program": "vae",
            "config": to_dict(conf),
            "code": {"snap": "x"},
        }
        (run / "job.json").write_text(json.dumps(job))
        assert loop.run(VAEProgram(), run / "job.json", attempt=1) == Exit.OK

        out = main([str(run), "--windows", str(WINDOWS), "--fit-steps", str(FIT)])
        rows = [json.loads(x) for x in (out / "evals.jsonl").read_text().splitlines()]
        assert [(r["t"], r["split"]) for r in rows] == [
            (0.0, "val"),
            (0.5, "val"),
            (1.0, "val"),
        ]
        fresh = Evaluator(
            conf, root, tmp_path / "fresh", windows=WINDOWS, fit_steps=FIT
        )
        (want,) = fresh.evaluate(build_vae(conf, seed=conf.seed), "fresh")
        assert rows[0]["stack_mse"] == want["stack_mse"]
        assert rows[-1]["stack_mse"] != want["stack_mse"]
