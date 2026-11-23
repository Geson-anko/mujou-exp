"""``exp.vision.jepa_eval`` on the tiny synthetic Walking Tours of
``test_eval`` (stacks of 2 frames of 16 px, patch 4; H held out, its val
split of 3 windows): the color code by hand, the predictions of known
solutions, SIGReg's references, the participation ratio and the tokens' R^2
on codes of known covariance, and a tiny run and the yardsticks end to end.
"""

import json
import math
from pathlib import Path
from typing import override

import pytest
import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt
from exp.lib.png import read_png
from exp.vision import jepa_iid
from exp.vision.compose import to_dict
from exp.vision.jepa import JEPAModules, rho_in, take
from exp.vision.jepa_eval import (
    CodeStats,
    ColorCode,
    evaluate_run,
    evaluate_yardsticks,
    participation,
    token_r2,
)
from exp.vision.models import build_vae
from exp.vision.schema import (
    DataConf,
    MeasureConf,
    ModelConf,
    OptimConf,
    RunConf,
    ToursConf,
)
from exp.vision.stream_eval import StreamEvaluator
from tests.vision.test_eval import SIZE, N, random_tours
from tests.vision.test_jepa import Copy, StackMean
from tests.vision.test_stream_eval import EVERY

FIT = 3  # steps of each decoder's fit
HELDIN = (2, 4)
SPAN = 3.0  # the whole of conf_of's stream
EP_CONST = math.sqrt(2 * math.pi) - 2 * math.sqrt(math.pi) + math.sqrt(2 * math.pi / 3)


def conf_of() -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=N)),
        model=ModelConf(width=64, patch=4, param="mup", latent=4),
        optim=OptimConf(kind="rmsprop"),
        measure=MeasureConf(
            duration=3.0, ckpt_times=[0.0, 3.0], clips=3, windows=1, posthoc_seeds=2
        ),
    )


@pytest.fixture
def root(tmp_path) -> Path:
    return random_tours(tmp_path / "d")


@pytest.fixture
def stats(root, tmp_path) -> CodeStats:
    ev = StreamEvaluator(
        conf_of(), tmp_path, root, tmp_path / "out", fit_steps=FIT, **EVERY
    )
    return CodeStats(
        conf_of(), root, ev, heldin_batches=HELDIN[0], heldin_batch=HELDIN[1]
    )


class PositionOnly(nn.Module):
    """An encoder whose tokens are a learned vector per position, whatever
    the stack."""

    def __init__(self, width: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.v = nn.Parameter(torch.randn(16, width, generator=g))

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        v = self.v.expand(len(images), -1, -1)
        return v if keep is None else take(v, keep)


class TestColorCode:
    def test_the_mean_of_each_patch_and_channel_by_hand(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randint(256, (2, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)
        got = ColorCode(4).tokens(x)
        assert got.shape == (2, 16, 3 * N)
        for b in range(2):
            for p in range(16):
                i, j = divmod(p, 4)
                for f in range(N):
                    for c in range(3):
                        want = x[b, f, c, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4]
                        assert float(got[b, p, 3 * f + c]) == pytest.approx(
                            float(want.double().mean()) / 255, rel=1e-6
                        )
        assert torch.equal(ColorCode(4).mu(x), got.flatten(1))


class TestCodeStats:
    def test_stack_constant_tokens_are_predicted_exactly(self, stats):
        proj = nn.Linear(8, 64)
        mods = JEPAModules(StackMean(8), proj, Copy(proj))
        got = stats.pred(mods, 8)
        assert got["pred"] == pytest.approx(0.0, abs=1e-12)
        assert got["pred_triv"] > 0 and got["pred_ratio"] == pytest.approx(
            0.0, abs=1e-9
        )
        assert got["pred_triv_calib"] > 0
        assert got["pred_ratio_calib"] == pytest.approx(0.0, abs=1e-9)

    def test_tokens_of_the_position_alone_are_their_positions_mean(self, stats):
        proj = nn.Linear(8, 64)
        mods = JEPAModules(PositionOnly(8), proj, Copy(proj))
        got = stats.pred(mods, 4)
        assert got["pred_triv"] == pytest.approx(0.0, abs=1e-10)
        assert got["pred_triv_calib"] == pytest.approx(0.0, abs=1e-10)

    def test_the_references_of_a_constant_are_0_409_n(self, stats):
        row = stats.of(ColorCode(4))
        assert row["sigreg_zurich_const"] == pytest.approx(3 * EP_CONST, rel=0.01)
        assert row["sigreg_heldin_const"] == pytest.approx(
            HELDIN[1] * EP_CONST, rel=0.01
        )
        assert math.isfinite(row["sigreg_heldin_gauss"])
        assert 0 < row["rho_in"] < 1 and row["z_var"] > 0
        assert 1 <= row["participation"] <= 16 * 3 * N
        assert row["token_r2"] <= 1


def stack_constant(n: int, t: int, d: int, g: torch.Generator) -> Tensor:
    """$z_{n,p} = h(x_n)$, $h \\sim N(0, I_d)$: ``(n, t, d)``."""
    return torch.randn(n, 1, d, generator=g).expand(-1, t, -1)


def rotations(t: int, d: int, g: torch.Generator) -> Tensor:
    """$t$ random orthogonal matrices $R_p$ ``(t, d, d)``."""
    q, _ = torch.linalg.qr(torch.randn(t, d, d, generator=g, dtype=torch.float64))
    return q.float()


class TestParticipation:
    def test_isotropic_samples_give_n_d_over_n_plus_d(self):
        # E tr S^2 = D (D + N + 1) / N for S of N(0, I_D) about a known mean
        z = torch.randn(1024, 256, generator=torch.Generator().manual_seed(0))
        assert float(participation(z)) == pytest.approx(1024 * 256 / 1280, rel=0.03)

    def test_a_code_on_a_line_gives_1(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randn(64, 1, generator=g) * torch.randn(1, 32, generator=g)
        assert float(participation(z)) == pytest.approx(1.0, rel=1e-9)

    def test_a_stack_constant_code_gives_about_d(self):
        # Sigma = (1 1^T) x I_d: r_2 = d, the sample's d N / (d + N)
        z = stack_constant(4096, 16, 8, torch.Generator().manual_seed(0)).flatten(1)
        assert float(participation(z)) == pytest.approx(8 * 4096 / 4104, rel=0.03)


class TestTokenR2:
    """The derivation's stack-constant family (``plans/jepa.md``) and its
    rotations by position, against tokens independent of each other."""

    T, D = 16, 8

    def test_stack_constant_tokens_give_1(self):
        g = torch.Generator().manual_seed(0)
        fit, held = (stack_constant(n, self.T, self.D, g) for n in (512, 256))
        assert float(token_r2(fit, held)) == pytest.approx(1.0, abs=1e-3)

    def test_rotated_by_position_they_give_1_where_rho_in_does_not_see_them(self):
        g = torch.Generator().manual_seed(1)
        r = rotations(self.T, self.D, g)
        fit, held = (
            torch.einsum("pij,npj->npi", r, stack_constant(n, self.T, self.D, g))
            for n in (512, 256)
        )
        assert float(token_r2(fit, held)) == pytest.approx(1.0, abs=1e-3)
        # rho_in reads them as tokens nearly independent of each other
        assert float(rho_in(held)) > 0.85

    def test_independent_tokens_give_about_0(self):
        g = torch.Generator().manual_seed(2)
        fit, held = (torch.randn(n, self.T, self.D, generator=g) for n in (4096, 1024))
        assert float(token_r2(fit, held)) == pytest.approx(0.0, abs=0.01)

    def test_half_shared_tokens_give_about_one_quarter(self):
        # z_p = (s + e_p) / sqrt(2): corr(z_p, z_q) = 1/2 per coordinate, R^2 = 1/4
        g = torch.Generator().manual_seed(3)

        def draw(n: int) -> Tensor:
            s = torch.randn(n, 1, self.D, generator=g)
            return (s + torch.randn(n, self.T, self.D, generator=g)) / math.sqrt(2)

        assert float(token_r2(draw(4096), draw(1024))) == pytest.approx(0.25, abs=0.02)


def tiny_run(root: Path, tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(to_dict(conf_of())))
    out = tmp_path / "run"
    argv = ["--config", str(cfg), "--mask-ratio", "0.25", "--lam", "0.5"]
    argv += ["--lr", "1e-3", "--steps", "4", "--batch", "2", "--warmup", "2"]
    argv += ["--directions", "16", "--device", "cpu", "--root", str(root)]
    jepa_iid.main([*argv, "--out", str(out)])
    return out


def jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()]


class TestEvaluate:
    def test_a_tiny_run_every_checkpoint_and_the_probes(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        out = evaluate_run(
            run,
            root=root,
            span=SPAN,
            probe_steps=(0,),
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
        )
        rows = jsonl(out / "jepa.jsonl")
        assert [(r["t"], r["step"]) for r in rows] == [(0.0, 0), (0.8, 4)]
        keys = ("pred", "pred_triv", "pred_ratio", "sigreg_zurich", "sigreg_heldin")
        keys += ("pred_ratio_calib", "rho_in", "token_r2", "participation", "z_var")
        assert all(math.isfinite(r[k]) for r in rows for k in keys)
        assert json.loads((out / "config.json").read_text())["span"] == SPAN
        probes = jsonl(out / "probes.jsonl")
        assert [(r["step"], r["group"]) for r in probes] == [
            (s, g) for s in (0, 4) for g in ("span", "after", "zurich")
        ]
        # two rows: the true last frames and the decoder's (no system's)
        for group in ("zurich", "span"):
            image = read_png(out / f"00000.8-{group}.png")
            assert image.shape[0] == 2 * (SIZE * 2 + 2)

    def test_the_yardsticks_with_a_vae(self, root, tmp_path):
        conf = conf_of()
        vae_run = tmp_path / "vae"
        vae_run.mkdir()
        (vae_run / "config.yaml").write_text(yaml.safe_dump(to_dict(conf)))
        path = ckpt.save_eval(
            vae_run,
            program="vae",
            group="iid",
            t=1.0,
            frame=10,
            modules=build_vae(conf, seed=5).modules(),
        )
        out = evaluate_yardsticks(
            conf,
            tmp_path / "y",
            root=root,
            v16=path,
            span=SPAN,
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
            pca_components=4,
        )
        rows = {r["code"]: r for r in jsonl(out / "jepa.jsonl")}
        # init4: init's z in the VAE's latent dimensions (4 here)
        assert list(rows) == ["init", "init4", "color", "pca", "v16"]
        for name, row in rows.items():
            tokens = name in ("init", "color")
            assert ("rho_in" in row) == ("token_r2" in row) == tokens, name
            assert math.isfinite(row["participation"])
        assert rows["init4"]["participation"] <= 4 + 1e-6
        probes = jsonl(out / "probes.jsonl")
        assert [r["code"] for r in probes] == [c for c in rows for _ in range(3)]
        assert all(
            math.isfinite(r["recon_blur"]) for r in probes if r["group"] != "after"
        )
        # the VAE's pictures have the system's row too
        assert read_png(out / "v16-zurich.png").shape[0] == 3 * (SIZE * 2 + 2)
        assert read_png(out / "pca-span.png").shape[0] == 2 * (SIZE * 2 + 2)
