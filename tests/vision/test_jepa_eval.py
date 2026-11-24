"""``exp.vision.jepa_eval`` on the tiny synthetic Walking Tours of
``test_eval`` (stacks of 2 frames of 16 px, patch 4; H held out, its val
split of 3 windows; patch 2 for V18b's finer grid, 8 x 8): the color code and
the codes of $h$ by hand, the predictions of known solutions, SIGReg's
references, the participation ratio and the tokens' R^2 on codes of known
covariance and against a solve of each pair, the drift of known changes, the
arms' codes, and tiny runs and the yardsticks end to end; V18c's runs (their
R and W read back, the code W vec(t) of the tokens t) and yardsticks.
"""

import itertools
import json
import math
from pathlib import Path
from typing import override

import pytest
import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt, pca
from exp.lib.png import read_png
from exp.vision import jepa_iid
from exp.vision.compose import from_dict, to_dict
from exp.vision.eval import PCA_SEED
from exp.vision.jepa import (
    LATENT_DIM,
    FixedProj,
    JEPAModules,
    Mix,
    TokenEncoder,
    build_jepa,
    rho_in,
    take,
)
from exp.vision.jepa_eval import (
    CodeStats,
    ColorCode,
    TokenCode,
    arm_codes,
    build_run_jepa,
    drift,
    evaluate_run,
    evaluate_yardsticks,
    participation,
    token_r2,
)
from exp.vision.models import Block, PatchEmbed, ViTPart, build_vae, early_fusion
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


def conf_of(patch: int = 4) -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=N)),
        model=ModelConf(width=64, patch=patch, param="mup", latent=4),
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


def random_stacks(b: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, (b, N, 3, SIZE, SIZE), generator=g, dtype=torch.uint8)


class TestTokenCode:
    def test_h_is_the_last_layer_norm_of_a_forward_by_hand(self):
        conf = conf_of()
        mods = build_jepa(conf, seed=3)
        x = random_stacks(3)
        assert isinstance(mods.encoder, TokenEncoder)
        parts = [p for p in mods.encoder.parts if isinstance(p, ViTPart)]
        embed, norm = parts[0].embed, parts[-1].norm
        assert embed is not None and norm is not None
        h = embed(x.reshape(3, N * 3, SIZE, SIZE).float() / 255)
        for part in parts:
            for block in part.blocks:
                h = block(h)
        h = norm(h)  # (3, T, w)
        with torch.no_grad():
            pools = ("z", "zbar", "h_mean", "h_flat", "h_pool")
            got = {p: TokenCode(mods, p).mu(x) for p in pools}
            torch.testing.assert_close(got["h_flat"], h.flatten(1))
            torch.testing.assert_close(got["h_mean"], h.mean(1))
            # h is the projection's input: z_p = P h_p, flattened
            torch.testing.assert_close(got["z"], mods.proj(h).flatten(1))
            torch.testing.assert_close(got["zbar"], mods.proj(h).mean(1))
            # on the coarse grid itself the pooled h is h
            torch.testing.assert_close(got["h_pool"], got["h_flat"])

    def test_h_pool_is_the_mean_of_each_coarse_block_by_hand(self):
        mods = build_jepa(conf_of(patch=2), seed=3)  # the grid 8 x 8, s = 2
        x = random_stacks(3)
        with torch.no_grad():
            h = mods.encoder(early_fusion(x))  # (3, 64, w)
            got = TokenCode(mods, "h_pool").mu(x)
        want = torch.zeros(3, 16, h.shape[2])
        for c in range(16):
            ci, cj = divmod(c, 4)
            fine = [8 * (2 * ci + a) + 2 * cj + b for a in (0, 1) for b in (0, 1)]
            want[:, c] = h[:, fine].mean(1)
        assert got.shape == (3, 16 * h.shape[2])
        torch.testing.assert_close(got, want.flatten(1))

    def test_an_unknown_pool_is_refused(self):
        with pytest.raises(ValueError, match="pool"):
            TokenCode(build_jepa(conf_of(), seed=0), "h_max")


class TestArmCodes:
    def test_the_codes_of_arm_m_by_hand(self):
        conf = conf_of(patch=2)
        mods = build_jepa(conf, seed=0)
        fit, x = random_stacks(40, seed=1), random_stacks(5, seed=2)
        names = ("zpca", "zbar", "zflat", "h_mean")
        with torch.no_grad():
            codes = arm_codes(mods, names, fit, seed=0, device="cpu", zpca_dim=8)
            assert list(codes) == list(names)
            gen = torch.Generator().manual_seed(PCA_SEED)
            mean, comps = pca.fit_pca(mods.mu(fit), 8, gen=gen)
            torch.testing.assert_close(
                codes["zpca"].mu(x), pca.encode(mods.mu(x), mean, comps)
            )
            assert torch.equal(codes["zflat"].mu(x), mods.mu(x))  # (5, 64 x 64)
            torch.testing.assert_close(codes["zbar"].mu(x), mods.tokens(x).mean(1))
            assert codes["h_mean"].mu(x).shape == (5, 64)
        assert hasattr(codes["zflat"], "tokens")  # rho_in and token_r2 of z_p

    def test_an_unknown_code_is_refused(self):
        mods = build_jepa(conf_of(), seed=0)
        with pytest.raises(ValueError, match="code"):
            arm_codes(mods, ("zmax",), random_stacks(4), seed=0, device="cpu")


class TestDrift:
    def test_0_at_the_start_and_the_relative_norm_of_known_changes(self):
        conf = conf_of()
        init, now = build_jepa(conf, seed=0), build_jepa(conf, seed=0)
        assert drift(init.encoder, now.encoder) == {
            "embed_drift": 0.0,
            "block_drift": 0.0,
        }

        def matrices(enc: nn.Module) -> tuple[Tensor, list[Tensor]]:
            (embed,) = (m for m in enc.modules() if isinstance(m, PatchEmbed))
            blocks = [b for b in enc.modules() if isinstance(b, Block)]
            lin = [
                m.weight
                for b in blocks
                for m in b.modules()
                if isinstance(m, nn.Linear)
            ]
            return embed.proj.weight, lin

        e0, b0 = matrices(init.encoder)
        e0, b0 = e0.detach(), [w.detach() for w in b0]
        e1, b1 = matrices(now.encoder)
        g = torch.Generator().manual_seed(1)
        de = 0.01 * torch.randn(e1.shape, generator=g)
        dq = 0.01 * torch.randn(b1[0].shape, generator=g)  # the first qkv alone
        with torch.no_grad():
            e1.add_(de)
            b1[0].add_(dq)
        got = drift(init.encoder, now.encoder)
        norm0 = math.sqrt(sum(float(w.double().square().sum()) for w in b0))
        assert got["embed_drift"] == pytest.approx(
            float(de.double().norm() / e0.double().norm()), rel=1e-5
        )
        assert got["block_drift"] == pytest.approx(
            float(dq.double().norm()) / norm0, rel=1e-5
        )


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

    def test_on_a_finer_grid_the_expanded_masks_and_their_count(self, root, tmp_path):
        # tokens constant within a stack, zero predictions: whichever positions
        # are hidden, pred is the mean over the zurich stacks of |z_n|^2 / d
        ev = StreamEvaluator(
            conf_of(), tmp_path, root, tmp_path / "out", fit_steps=FIT, **EVERY
        )
        stats = CodeStats(
            conf_of(), root, ev, heldin_batches=HELDIN[0], heldin_batch=HELDIN[1]
        )
        proj, zero = nn.Linear(8, 64), nn.Linear(8, 64)
        with torch.no_grad():
            zero.weight.zero_()
            zero.bias.zero_()
        mods = JEPAModules(StackMean(8, tokens=64), proj, Copy(zero))
        got = stats.pred(mods, 8, 2)
        with torch.no_grad():
            z = mods.tokens(ev.zurich_stacks)[:, 0]  # (N, d), every position's
        want = float(z.double().square().mean())
        assert got["pred"] == pytest.approx(want, rel=1e-6)
        exact = JEPAModules(StackMean(8, tokens=64), proj, Copy(proj))
        assert stats.pred(exact, 8, 2)["pred"] == pytest.approx(0.0, abs=1e-12)

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

    def test_the_solve_of_all_pairs_is_that_of_each_pair(self):
        g = torch.Generator().manual_seed(4)
        t, d = 5, 3

        def draw(n: int) -> Tensor:
            s = torch.randn(n, 1, d, generator=g, dtype=torch.float64)
            return s + 0.7 * torch.randn(n, t, d, generator=g, dtype=torch.float64)

        fit, held = draw(64), draw(32)
        r2 = []
        for p, q in itertools.permutations(range(t), 2):
            xp, xq = fit[:, p] - fit[:, p].mean(0), fit[:, q] - fit[:, q].mean(0)
            gram = xp.T @ xp
            alpha = 1e-3 * gram.trace() / d
            w = torch.linalg.solve(
                gram + alpha * torch.eye(d, dtype=gram.dtype), xp.T @ xq
            )
            z_hat = (held[:, p] - fit[:, p].mean(0)) @ w + fit[:, q].mean(0)
            err = (held[:, q] - z_hat).square().sum()
            r2.append(1 - err / (held[:, q] - held[:, q].mean(0)).square().sum())
        want = float(torch.stack(r2).mean())
        assert float(token_r2(fit, held)) == pytest.approx(want, rel=1e-10)


def tiny_run(root: Path, tmp_path: Path, *extra: str, name: str = "run") -> Path:
    cfg = tmp_path / f"{name}.yaml"
    d = to_dict(conf_of())
    cfg.write_text(yaml.safe_dump(d))
    out = tmp_path / name
    argv = ["--config", str(cfg), "--mask-ratio", "0.25", "--lam", "0.5"]
    argv += ["--lr", "1e-3", "--steps", "4", "--batch", "2", "--warmup", "2"]
    argv += ["--directions", "16", "--device", "cpu", "--root", str(root)]
    jepa_iid.main([*argv, "--out", str(out), *extra])
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
        # the flat arm's codes: z (V18's) and h_mean, a row of each a checkpoint
        assert [(r["t"], r["step"], r["code"]) for r in rows] == [
            (t, s, c) for t, s in ((0.0, 0), (0.8, 4)) for c in ("z", "h_mean")
        ]
        keys = ("pred", "pred_triv", "pred_ratio", "sigreg_zurich", "sigreg_heldin")
        keys += ("pred_ratio_calib", "participation", "z_var")
        keys += ("embed_drift", "block_drift")
        assert all(math.isfinite(r[k]) for r in rows for k in keys)
        for r in rows:  # the tokens' statistics of z alone
            assert ("token_r2" in r) == ("rho_in" in r) == (r["code"] == "z")
        assert rows[0]["embed_drift"] == rows[0]["block_drift"] == 0.0
        assert rows[2]["block_drift"] > 0
        config = json.loads((out / "config.json").read_text())
        assert config["span"] == SPAN and config["codes"] == ["z", "h_mean"]
        assert config["sigreg_on"] == "flat" and config["token_dim"] == 64
        probes = jsonl(out / "probes.jsonl")
        assert [(r["step"], r["code"], r["group"]) for r in probes] == [
            (s, c, g)
            for s in (0, 4)
            for c in ("z", "h_mean")
            for g in ("span", "after", "zurich")
        ]
        # two rows: the true last frames and the decoder's (no system's)
        for group in ("zurich", "span"):
            image = read_png(out / f"00000.8-z-{group}.png")
            assert image.shape[0] == 2 * (SIZE * 2 + 2)

    def test_the_z_of_arm_p16_is_v18s_code_of_the_default_codes(self, root, tmp_path):
        run = tiny_run(root, tmp_path)
        kw = {"root": root, "span": SPAN, "fit_steps": FIT, "spacing": EVERY}
        both = evaluate_run(run, probe_steps=(), heldin=HELDIN, **kw)
        a = [r for r in jsonl(both / "jepa.jsonl") if r["code"] == "z"]
        alone = evaluate_run(run, probe_steps=(), heldin=HELDIN, codes=("z",), **kw)
        assert a == jsonl(alone / "jepa.jsonl")

    def test_an_arm_m_run_rows_of_its_codes(self, root, tmp_path):
        run = tiny_run(root, tmp_path, "--patch", "2", "--sigreg-on", "mean")
        out = evaluate_run(
            run,
            root=root,
            span=SPAN,
            probe_steps=(),
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
            zpca_dim=4,
        )
        rows = jsonl(out / "jepa.jsonl")
        codes = ("zpca", "zbar", "zflat", "h_mean")
        assert [(r["step"], r["code"]) for r in rows] == [
            (s, c) for s in (0, 4) for c in codes
        ]
        by = {(r["step"], r["code"]): r for r in rows}
        assert by[4, "zpca"]["participation"] <= 4 + 1e-6
        assert "token_r2" in by[4, "zflat"] and "token_r2" not in by[4, "zbar"]
        assert all(math.isfinite(r["pred_ratio_calib"]) for r in rows)
        # every code of a checkpoint shares its pred and drift
        assert len({by[4, c]["pred"] for c in codes}) == 1
        config = json.loads((out / "config.json").read_text())
        assert config["sigreg_on"] == "mean" and config["expand"] == 2
        probes = jsonl(out / "probes.jsonl")
        assert [(r["step"], r["code"]) for r in probes] == [
            (4, c) for c in codes for _ in range(3)
        ]

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

    def test_the_yardsticks_of_a_finer_grid(self, root, tmp_path):
        conf = conf_of(patch=2)  # T = 64: arm F's d = 1024 / 64 = 16
        out = evaluate_yardsticks(
            conf,
            tmp_path / "y",
            root=root,
            span=SPAN,
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
            zpca_dim=4,
        )
        rows = {r["code"]: r for r in jsonl(out / "jepa.jsonl")}
        assert list(rows) == [
            "init_f",
            "init_m",
            "init_m_zbar",
            "init_m_zflat",
            "color2",
        ]
        for name, row in rows.items():
            tokens = name in ("init_f", "init_m_zflat", "color2")
            assert ("rho_in" in row) == ("token_r2" in row) == tokens, name
        assert rows["init_m"]["participation"] <= 4 + 1e-6
        config = json.loads((out / "config.json").read_text())
        assert config["token_dims"] == {"init_f": 16, "init_m": 64}
        probes = jsonl(out / "probes.jsonl")
        assert [r["code"] for r in probes] == [c for c in rows for _ in range(3)]


V18C = ("--patch", "2", "--token-dim", "16", "--proj", "fixed", "--sigreg-on", "mix")


class TestV18c:
    """V18c's runs (patch 2 at 16 px: the 8 x 8 grid of patch 8 at 64 px): the
    code $z = W \\mathrm{vec}(t)$ and the tokens $t_p = R h_p$."""

    def test_the_run_is_built_with_its_r_and_w_and_reads_its_checkpoint(
        self, root, tmp_path
    ):
        run = tiny_run(root, tmp_path, *V18C)
        conf = from_dict(yaml.safe_load((run / "config.yaml").read_text()))
        mods = build_run_jepa(conf, json.loads((run / "jepa.json").read_text()))
        assert isinstance(mods.proj, FixedProj) and isinstance(mods.mix, Mix)
        ckpt.load_eval(ckpt.eval_path(run, 0.8), mods.modules())
        x = random_stacks(3)
        with torch.no_grad():
            t = mods.tokens(x)
            assert t.shape == (3, 64, 16)
            assert torch.equal(mods.mu(x), t.flatten(1) @ mods.mix.weight.T)
        assert mods.mu(x).shape == (3, LATENT_DIM)

    def test_a_v18_run_without_the_new_settings_is_built_as_before(
        self, root, tmp_path
    ):
        run = tiny_run(root, tmp_path)
        settings = json.loads((run / "jepa.json").read_text())
        old = {k: v for k, v in settings.items() if k not in ("proj", "latent_dim")}
        conf = from_dict(yaml.safe_load((run / "config.yaml").read_text()))
        mods = build_run_jepa(conf, old)
        assert mods.mix is None and isinstance(mods.proj, nn.Linear)
        ckpt.load_eval(ckpt.eval_path(run, 0.8), mods.modules())

    def test_a_v18c_run_rows_of_z_and_h_mean(self, root, tmp_path):
        run = tiny_run(root, tmp_path, *V18C)
        out = evaluate_run(
            run,
            root=root,
            span=SPAN,
            probe_steps=(),
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
        )
        rows = jsonl(out / "jepa.jsonl")
        assert [(r["step"], r["code"]) for r in rows] == [
            (s, c) for s in (0, 4) for c in ("z", "h_mean")
        ]
        keys = ("pred", "pred_ratio_calib", "sigreg_heldin", "sigreg_zurich", "z_var")
        assert all(math.isfinite(r[k]) for r in rows for k in keys)
        for r in rows:  # token_r2 and rho_in of the tokens t of z alone
            assert ("token_r2" in r) == (r["code"] == "z")
        config = json.loads((out / "config.json").read_text())
        assert config["sigreg_on"] == "mix" and config["proj"] == "fixed"
        assert config["latent_dim"] == LATENT_DIM and config["token_dim"] == 16
        assert config["codes"] == ["z", "h_mean"] and config["expand"] == 2

    def test_the_yardsticks_of_v18c(self, root, tmp_path):
        out = evaluate_yardsticks(
            conf_of(patch=2),
            tmp_path / "y",
            root=root,
            span=SPAN,
            fit_steps=FIT,
            spacing=EVERY,
            heldin=HELDIN,
            token_dim=16,
            proj="fixed",
            sigreg_on="mix",
        )
        rows = {r["code"]: r for r in jsonl(out / "jepa.jsonl")}
        assert list(rows) == ["init_c", "init_c_h_mean", "color2"]
        for name, row in rows.items():
            tokens = name in ("init_c", "color2")
            assert ("rho_in" in row) == ("token_r2" in row) == tokens, name
        config = json.loads((out / "config.json").read_text())
        assert config["sigreg_on"] == "mix" and config["proj"] == "fixed"
        assert config["token_dim"] == 16 and config["latent_dim"] == LATENT_DIM
        # init_c is the step 0 of a V18c run of the same seed
        run = tiny_run(root, tmp_path, *V18C)
        first = jsonl(
            evaluate_run(
                run,
                root=root,
                span=SPAN,
                probe_steps=(),
                fit_steps=FIT,
                spacing=EVERY,
                heldin=HELDIN,
                codes=("z",),
            )
            / "jepa.jsonl"
        )[0]
        assert first["step"] == 0
        for k in ("sigreg_heldin", "sigreg_zurich", "token_r2", "participation"):
            assert rows["init_c"][k] == pytest.approx(first[k], rel=1e-6), k
