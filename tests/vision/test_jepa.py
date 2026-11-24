"""``exp.vision.jepa``: the mask family against a brute force of another
construction, its expansion to a finer grid against the pixels it hides, the
context side blind to the hidden patches, the loss by hand and its gradient
on the target side, SIGReg's values of the derivation (``plans/jepa.md``,
V18 and V18b) and its data point, activation checkpointing against the plain
forward, and muP's roles against the VAE's."""

import itertools
import math
from collections import Counter
from typing import override

import pytest
import torch
from torch import Tensor, nn

from exp.lib import regions
from exp.lib.sigreg import sigreg
from exp.vision.jepa import (
    COARSE_GRID,
    TOKEN_DIM,
    JEPAModules,
    Predictor,
    build_jepa,
    directions,
    draw_masks,
    expand_masks,
    hidden_count,
    jepa_mup_lr,
    jepa_mup_roles,
    jepa_terms,
    mask_expansion,
    mask_family,
    mask_index,
    predict,
    rho_in,
    sigreg_points,
    take,
)
from exp.vision.models import build_vae, early_fusion, mup_lr
from exp.vision.schema import DataConf, ModelConf, OptimConf, RunConf, ToursConf

GRID = 4
# |F_k|, those of one rectangle, and the counts by torus components (plan)
FAMILY = {
    4: (632, 24, {1: 280, 2: 352}),
    8: (436, 8, {1: 392, 2: 44}),
    12: (232, 8, {1: 232}),
}


def conf_of(
    width: int = 512, param: str = "mup", *, size: int = 16, patch: int = 4
) -> RunConf:
    """Stacks of 2 frames of 16 px, patch 4: the 4 x 4 grid of V18 (64 px and
    patch 4: the 16 x 16 grid of V18b)."""
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(size=size, frames=2)),
        model=ModelConf(width=width, patch=patch, param=param),
        optim=OptimConf(kind="rmsprop"),
    )


def stacks(b: int, seed: int = 0, size: int = 16) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(256, (b, 2, 3, size, size), generator=g, dtype=torch.uint8)


# -- the brute force: rectangles as products of cyclic intervals ----------------


def _intervals(g: int) -> set[frozenset[int]]:
    return {
        frozenset((s + i) % g for i in range(n))
        for s in range(g)
        for n in range(1, g + 1)
    }


def _rects(g: int) -> list[frozenset[tuple[int, int]]]:
    iv = _intervals(g)
    return [frozenset(itertools.product(r, c)) for r in iv for c in iv]


def _components(cells: frozenset[tuple[int, int]], g: int) -> int:
    seen: set[tuple[int, int]] = set()
    n = 0
    for c in cells:
        if c in seen:
            continue
        n += 1
        stack = [c]
        seen.add(c)
        while stack:
            r, q = stack.pop()
            for dr, dq in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = ((r + dr) % g, (q + dq) % g)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
    return n


def brute_family(k: int, g: int) -> set[frozenset[tuple[int, int]]]:
    """Every ``k``-subset of the grid that is the union of two rectangles it
    contains (a rectangle twice for one)."""
    rects = _rects(g)
    out = set()
    for subset in itertools.combinations(itertools.product(range(g), range(g)), k):
        s = frozenset(subset)
        inside = [r for r in rects if r <= s]
        if any(a | b == s for a in inside for b in inside):
            out.add(s)
    return out


def as_cells(mask: Tensor, g: int) -> frozenset[tuple[int, int]]:
    return frozenset(divmod(int(p), g) for p in mask.nonzero()[:, 0])


class TestMasks:
    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_the_family_is_the_brute_force_of_its_sizes_and_shapes(self, k):
        family = mask_family(k, GRID)
        got = {as_cells(m, GRID) for m in family}
        assert len(got) == len(family)  # distinct
        assert got == brute_family(k, GRID)
        size, single, comps = FAMILY[k]
        assert len(family) == size
        assert len(got & set(_rects(GRID))) == single
        assert Counter(_components(s, GRID) for s in got) == comps

    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_every_mask_hides_k_and_the_family_is_closed_under_translation(self, k):
        family = mask_family(k, GRID)
        assert (family.sum(1) == k).all()
        flat = {tuple(m.tolist()) for m in family}
        grids = family.reshape(-1, GRID, GRID)
        for di in range(GRID):
            for dj in range(GRID):
                moved = grids.roll((di, dj), dims=(1, 2)).flatten(1)
                assert all(tuple(m.tolist()) in flat for m in moved)

    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_each_position_is_covered_by_k_f_over_t_masks(self, k):
        family = mask_family(k, GRID)
        want = k * len(family) // (GRID * GRID)
        assert k * len(family) % (GRID * GRID) == 0
        assert family.sum(0).tolist() == [want] * (GRID * GRID)

    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_the_draws_hide_each_position_k_over_t_of_the_time(self, k):
        n, p = 100_000, k / (GRID * GRID)
        masks = draw_masks(mask_family(k, GRID), n, torch.Generator().manual_seed(k))
        assert (masks.sum(1) == k).all()
        freq = masks.double().sum(0)
        sd = math.sqrt(n * p * (1 - p))
        assert ((freq - n * p).abs() < 4 * sd).all()

    def test_the_visible_and_hidden_positions_in_order(self):
        mask = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]]).bool()
        keep, hide = mask_index(mask)
        assert keep.tolist() == [[1, 2], [0, 3]]
        assert hide.tolist() == [[0, 3], [1, 2]]

    @pytest.mark.parametrize(("r", "k"), [(0.25, 4), (0.5, 8), (0.75, 12)])
    def test_the_ratios_of_v18_hide_4_8_12(self, r, k):
        assert hidden_count(r, 16) == k

    @pytest.mark.parametrize("r", [0.3, 0.0, 1.0])
    def test_a_ratio_of_no_whole_count_in_range_is_refused(self, r):
        with pytest.raises(ValueError, match="mask ratio"):
            hidden_count(r, 16)


def pixel_mask(mask: Tensor, grid: int, side: int = 64) -> Tensor:
    """The pixels ``(side, side)`` hidden by ``mask`` ``(grid^2,)``, each
    position a square of ``side / grid`` px, by the pixels' coordinates."""
    cell = side // grid
    ys, xs = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    return mask[(ys // cell) * grid + xs // cell]


class TestExpandedMasks:
    """V18b: V18's masks of the 4 x 4 grid drawn and spread to the 16 x 16
    grid of patch 4 (``plans/jepa.md``, 「マスク」)."""

    FINE, S = 16, 4

    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_every_mask_hides_the_same_pixels_as_v18s(self, k):
        family = mask_family(k, GRID)
        fine = expand_masks(family, self.S)
        assert fine.shape == (len(family), self.FINE**2)
        assert (fine.sum(1) == self.S**2 * k).all()
        for coarse, spread in zip(family, fine, strict=True):
            assert torch.equal(pixel_mask(coarse, GRID), pixel_mask(spread, self.FINE))

    @pytest.mark.parametrize("k", sorted(FAMILY))
    def test_each_fine_position_is_covered_by_k_f_over_16_masks(self, k):
        family = mask_family(k, GRID)
        want = k * len(family) // (GRID * GRID)
        got = expand_masks(family, self.S).sum(0)
        assert got.tolist() == [want] * self.FINE**2  # P(hidden) = k / 16 exactly

    def test_the_same_draws_hide_the_same_pixels_at_patch_4_and_16(self):
        family = mask_family(8, GRID)
        coarse = draw_masks(family, 64, torch.Generator().manual_seed(5))
        fine = expand_masks(
            draw_masks(family, 64, torch.Generator().manual_seed(5)), self.S
        )
        for a, b in zip(expand_masks(coarse, 1), fine, strict=True):
            assert torch.equal(pixel_mask(a, GRID), pixel_mask(b, self.FINE))

    def test_one_is_the_identity(self):
        family = mask_family(4, GRID)
        assert torch.equal(expand_masks(family, 1), family)

    @pytest.mark.parametrize(("k", "visible"), [(4, 192), (8, 128), (12, 64)])
    def test_the_visible_and_hidden_positions_of_the_fine_grid(self, k, visible):
        mask = expand_masks(masks(3, k), self.S)
        keep, hide = mask_index(mask)
        assert keep.shape == (3, visible) and hide.shape == (3, 256 - visible)

    @pytest.mark.parametrize(("grid", "s"), [(4, 1), (8, 2), (16, 4)])
    def test_the_expansion_of_a_grid(self, grid, s):
        assert mask_expansion(grid) == s

    @pytest.mark.parametrize("grid", [2, 6, 10])
    def test_a_grid_of_no_whole_expansion_is_refused(self, grid):
        with pytest.raises(ValueError, match="grid"):
            mask_expansion(grid)


class TestModules:
    def test_the_encoder_is_build_vaes_without_the_head(self):
        conf = conf_of()
        mods = build_jepa(conf, seed=3)
        vae = build_vae(conf, seed=3).encoder.state_dict()
        want = {k: v for k, v in vae.items() if not k.startswith("parts.3.head")}
        got = mods.encoder.state_dict()
        assert set(got) == set(want) and len(want) < len(vae)
        assert all(torch.equal(got[k], want[k]) for k in want)

    def test_the_context_side_does_not_see_the_hidden_patches(self):
        mods = build_jepa(conf_of(width=64), seed=0)
        x = stacks(3)
        mask = draw_masks(mask_family(8, GRID), 3, torch.Generator().manual_seed(1))
        keep, _ = mask_index(mask)
        y = x.clone()
        for n in range(3):  # new pixels in every hidden patch (4 x 4 px)
            for p in mask[n].nonzero()[:, 0].tolist():
                i, j = divmod(p, GRID)
                y[n, :, :, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4] = (
                    255 - x[n, :, :, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4]
                )
        with torch.no_grad():
            a = mods.encoder(early_fusion(x), keep)
            b = mods.encoder(early_fusion(y), keep)
            full = mods.encoder(early_fusion(x)), mods.encoder(early_fusion(y))
        assert a.shape == (3, 8, 64)
        assert torch.equal(a, b)
        assert not torch.equal(*full)  # the target side does see them

    def test_at_patch_4_the_context_side_does_not_see_the_hidden_patches(self):
        mods = build_jepa(conf_of(width=64, size=64, patch=4), seed=0)
        x = stacks(3, size=64)
        mask = expand_masks(masks(3, 8, seed=1), COARSE_GRID)
        keep, _ = mask_index(mask)
        y = x.clone()
        for n in range(3):  # new pixels in every hidden patch (4 x 4 px)
            for p in mask[n].nonzero()[:, 0].tolist():
                i, j = divmod(p, 16)
                y[n, :, :, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4] = (
                    255 - x[n, :, :, 4 * i : 4 * i + 4, 4 * j : 4 * j + 4]
                )
        with torch.no_grad():
            a = mods.encoder(early_fusion(x), keep)
            b = mods.encoder(early_fusion(y), keep)
            full = mods.encoder(early_fusion(x)), mods.encoder(early_fusion(y))
        assert a.shape == (3, 128, 64)
        assert torch.equal(a, b)
        assert not torch.equal(*full)  # the target side does see them

    def test_the_arms_f_and_m_start_from_the_same_encoder(self):
        conf = conf_of(width=64, size=64, patch=4)
        f = build_jepa(conf, seed=2, token_dim=4)
        m = build_jepa(conf, seed=2, token_dim=64)
        got, want = f.encoder.state_dict(), m.encoder.state_dict()
        assert set(got) == set(want)
        assert all(torch.equal(got[k], want[k]) for k in want)
        assert f.proj.out_features == 4 and m.proj.out_features == 64
        assert isinstance(f.predictor, Predictor)
        assert f.predictor.out.out_features == 4
        assert f.predictor.pos.shape == (256, 256)  # sincos_2d((16, 16), 256)

    def test_checkpointing_the_blocks_gives_the_same_loss_and_gradients(self):
        x, mask = stacks(4), masks(4, 8)
        dirs = directions(16, 16 * TOKEN_DIM, torch.Generator().manual_seed(2))
        out = []
        for flag in (False, True):
            mods = build_jepa(conf_of(width=64), seed=0, checkpoint_blocks=flag)
            loss = jepa_terms(mods, x, mask, dirs, 0.5)["loss"]
            grads = torch.autograd.grad(loss, mods.parameters())
            out.append((loss, grads))
        (la, ga), (lb, gb) = out
        assert torch.equal(la, lb)
        assert all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True))

    def test_mup_draws_the_outputs_k_to_the_minus_half_of_sp(self):
        sp, mu = build_jepa(conf_of(param="sp"), seed=0), build_jepa(conf_of(), seed=0)
        k = 512 / 256
        assert torch.allclose(mu.proj.weight, sp.proj.weight * k**-0.5)
        assert isinstance(mu.predictor, Predictor) and isinstance(
            sp.predictor, Predictor
        )
        assert torch.allclose(
            mu.predictor.embed.weight, sp.predictor.embed.weight * k**-0.5
        )
        assert torch.equal(mu.predictor.out.weight, sp.predictor.out.weight)


class TestMuP:
    def test_the_roles_of_the_new_parameters(self):
        mods = build_jepa(conf_of(), seed=0)
        roles = jepa_mup_roles(mods)
        pred = mods.predictor
        assert isinstance(pred, Predictor)
        assert roles[mods.proj.weight] == "output"
        assert roles[pred.embed.weight] == roles[pred.out.weight] == "output"
        assert roles[pred.mask_token] == roles[mods.proj.bias] == "vector"
        block = pred.blocks[0]
        assert isinstance(block, nn.Module)
        assert {roles[p] for n, p in block.named_parameters() if "weight" in n} == {
            "hidden",
            "vector",  # the norms
        }
        assert Counter(roles.values())["input"] == 1

    def test_the_encoders_multipliers_are_the_vaes_and_the_rest_by_the_table(self):
        conf = conf_of()
        mods = build_jepa(conf, seed=0)
        lr = jepa_mup_lr(mods, conf.model.width)
        vae = build_vae(conf, seed=0)
        vae_lr = mup_lr(vae, conf.model)
        by_name = dict(vae.encoder.named_parameters())
        for name, p in mods.encoder.named_parameters():
            assert lr[p] == vae_lr[by_name[name]], name
        pred = mods.predictor
        assert isinstance(pred, Predictor)
        assert lr[mods.proj.weight] == lr[pred.embed.weight] == 0.5  # k_w = 2
        assert lr[mods.proj.bias] == lr[pred.embed.bias] == 1.0
        rest = [p for n, p in pred.named_parameters() if not n.startswith("embed.")]
        assert {lr[p] for p in rest} == {1.0}  # the predictor's k is 1
        assert len(lr) == len(mods.parameters())


class StackMean(nn.Module):
    """An encoder whose every token of ``tokens`` is the stack's mean pixel
    times a learned vector: constant within a stack."""

    def __init__(self, width: int, tokens: int = 16) -> None:
        super().__init__()
        self.v = nn.Parameter(
            torch.randn(width, generator=torch.Generator().manual_seed(0))
        )
        self._t = tokens

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        t = self._t if keep is None else keep.shape[1]
        m = images.mean((1, 2, 3))[:, None, None]
        return (m * self.v).expand(-1, t, -1)


class Copy(nn.Module):
    """A predictor that maps the mean visible token by ``proj`` to every
    hidden position."""

    def __init__(self, proj: nn.Linear) -> None:
        super().__init__()
        self._proj = [proj]

    @override
    def forward(self, context: Tensor, keep: Tensor, hide: Tensor) -> Tensor:
        z = self._proj[0](context.mean(1))
        return z[:, None].expand(-1, hide.shape[1], -1)


def masks(b: int, k: int, seed: int = 0) -> Tensor:
    return draw_masks(mask_family(k, GRID), b, torch.Generator().manual_seed(seed))


class TestLoss:
    def test_the_terms_by_hand(self):
        mods = build_jepa(conf_of(width=64), seed=0)
        x, mask = stacks(4), masks(4, 12)
        dirs = directions(32, 16 * TOKEN_DIM, torch.Generator().manual_seed(2))
        lam = 0.3
        with torch.no_grad():
            got = jepa_terms(mods, x, mask, dirs, lam)
            images = early_fusion(x)
            z = mods.proj(mods.encoder(images))  # (B, T, d)
            sq = 0.0
            for n in range(4):
                hidden = mask[n].nonzero()[:, 0]
                visible = (~mask[n]).nonzero()[:, 0]
                ctx = mods.encoder(images[n : n + 1], visible[None])
                z_hat = mods.predictor(ctx, visible[None], hidden[None])[0]
                sq += float((z_hat - z[n, hidden]).double().square().sum())
            s = sigreg(z.flatten(1), z.flatten(1)[:0], dirs)
        pred = sq / (4 * 12 * TOKEN_DIM)
        assert float(got["pred"]) == pytest.approx(pred, rel=1e-5)
        assert float(got["sigreg"]) == pytest.approx(float(s), rel=1e-6)
        assert float(got["loss"]) == pytest.approx(
            (1 - lam) * pred + lam * float(s), rel=1e-5
        )
        assert torch.allclose(got["tokens"], z)

    def test_the_gradient_reaches_the_target_side(self):
        mods = build_jepa(conf_of(width=64), seed=0)
        for m in mods.modules().values():
            m.double()
        x = stacks(4).double() / 255  # floats in [0, 1]: early_fusion keeps them
        mask = masks(4, 8)
        dirs = directions(8, 16 * TOKEN_DIM, torch.Generator().manual_seed(2)).double()
        pred = jepa_terms(mods, x, mask, dirs, 0.0)["pred"]
        (g_p,) = torch.autograd.grad(pred, [mods.proj.weight])
        # P is on the target side alone: dL/dP = 2/(Bkd) sum (P h + b - z_hat) h^T
        with torch.no_grad():
            images = early_fusion(x)
            h = mods.encoder(images)
            z_hat, hide = predict(mods, images, mask)
            hh = take(h, hide)
            e = mods.proj(hh) - z_hat
            want = 2 * torch.einsum("nkd,nkw->dw", e, hh) / (4 * 8 * TOKEN_DIM)
        assert torch.allclose(g_p, want, rtol=1e-9, atol=1e-12)
        assert g_p.abs().max() > 0
        # the encoder's gradient differs from that of a detached target
        enc = list(mods.encoder.parameters())
        g_full = torch.autograd.grad(jepa_terms(mods, x, mask, dirs, 0.0)["pred"], enc)
        z_hat, hide = predict(mods, early_fusion(x), mask)
        target = take(mods.proj(mods.encoder(early_fusion(x))), hide).detach()
        g_stop = torch.autograd.grad((z_hat - target).square().mean(), enc)
        assert any(
            not torch.allclose(a, b) for a, b in zip(g_full, g_stop, strict=True)
        )

    def test_the_predictions_are_in_the_order_of_the_hidden_positions(self):
        # blocks of zero output projections are the identity, so the output at
        # a hidden position p is out(norm(mask_token + pos[p])), by hand
        pred = Predictor(width_in=8, grid=GRID, dim=64, depth=2)
        with torch.no_grad():
            pred.mask_token.normal_(generator=torch.Generator().manual_seed(0))
            for block in pred.blocks:
                assert isinstance(block, nn.Module)
                for name, p in block.named_parameters():
                    if name.startswith(("proj.", "fc2.")):
                        p.zero_()
            mask = masks(3, 8, seed=4)
            keep, hide = mask_index(mask)
            context = torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(1))
            got = pred(context, keep, hide)
            want = pred.out(pred.norm(pred.mask_token + pred.pos))  # (T, d) by position
        assert got.shape == (3, 8, TOKEN_DIM)
        for n in range(3):
            for j, p in enumerate(mask[n].nonzero()[:, 0].tolist()):
                assert torch.allclose(got[n, j], want[p], atol=1e-6), (n, j, p)
        assert not torch.allclose(want[0], want[1])  # the positions differ

    def test_stack_constant_tokens_and_a_copying_predictor_predict_exactly(self):
        proj = nn.Linear(8, TOKEN_DIM)
        mods = JEPAModules(StackMean(8), proj, Copy(proj))
        dirs = directions(8, 16 * TOKEN_DIM, torch.Generator().manual_seed(0))
        with torch.no_grad():
            got = jepa_terms(mods, stacks(5), masks(5, 8), dirs, 0.5)
        assert float(got["pred"]) == pytest.approx(0.0, abs=1e-12)
        assert float(rho_in(got["tokens"])) == pytest.approx(0.0, abs=1e-12)


class TestSIGRegOfTheDerivation:
    """The values of ``plans/jepa.md`` at N = 128, D = 1024, M = 1024."""

    N, D, M = 128, 1024, 1024

    def mean_s(self, draw, reps: int = 4) -> float:
        g = torch.Generator().manual_seed(7)
        total = 0.0
        for _ in range(reps):
            z = draw(g)
            total += float(sigreg(z, z[:0], directions(self.M, self.D, g)))
        return total / reps

    def test_a_constant_is_0_409_n(self):
        s = self.mean_s(lambda g: torch.zeros(self.N, self.D), reps=1)
        exact = self.N * (
            math.sqrt(2 * math.pi) - 2 * math.sqrt(math.pi) + math.sqrt(2 * math.pi / 3)
        )
        assert exact == pytest.approx(52.3, abs=0.1)
        assert s == pytest.approx(exact, rel=0.01)  # the trapezoid rule's error

    def test_gaussian_samples_are_about_1_06(self):
        s = self.mean_s(lambda g: torch.randn(self.N, self.D, generator=g))
        assert s == pytest.approx(
            math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3), abs=0.1
        )

    def test_stack_constant_codes_are_about_1_55(self):
        def draw(g: torch.Generator) -> Tensor:
            h = torch.randn(self.N, 1, TOKEN_DIM, generator=g)
            return h.expand(-1, 16, -1).reshape(self.N, self.D)

        assert self.mean_s(draw) == pytest.approx(1.55, abs=0.15)


class Table(nn.Module):
    """An encoder whose ``tokens`` tokens of the $n$-th stack of a batch are
    all $h_n$, the $n$-th row of the table ``h`` ``(N, w)``: the stack-constant
    family K3 of any $h$."""

    def __init__(self, h: Tensor, tokens: int) -> None:
        super().__init__()
        self.h = nn.Parameter(h)
        self._t = tokens

    @override
    def forward(self, images: Tensor, keep: Tensor | None = None) -> Tensor:
        t = self._t if keep is None else keep.shape[1]
        return self.h[: len(images), None].expand(-1, t, -1)


class TestSIGRegDataPoint:
    """V18b's arms (``plans/jepa.md``, 「自明な解の導出」): SIGReg of the flat
    $z$ (F, P16) or of the mean token $\bar z$ (M), at N = 128, T = 256,
    M = 1024."""

    N, M, T = 128, 1024, 256
    HEALTHY = math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)  # N(0, I)'s

    def test_the_data_points_of_flat_and_mean(self):
        z = torch.randn(3, 5, 2, generator=torch.Generator().manual_seed(0))
        assert torch.equal(sigreg_points(z, "flat"), z.flatten(1))
        assert torch.allclose(sigreg_points(z, "mean"), z.mean(1))
        with pytest.raises(ValueError, match="sigreg_on"):
            sigreg_points(z, "max")

    def test_mean_takes_sigreg_of_the_mean_token_by_hand(self):
        mods = build_jepa(conf_of(width=64), seed=0)
        x, mask = stacks(4), masks(4, 8)
        dirs = directions(32, TOKEN_DIM, torch.Generator().manual_seed(2))
        flat_dirs = directions(32, 16 * TOKEN_DIM, torch.Generator().manual_seed(2))
        lam = 0.3
        with torch.no_grad():
            got = jepa_terms(mods, x, mask, dirs, lam, sigreg_on="mean")
            flat = jepa_terms(mods, x, mask, flat_dirs, lam)
            z_bar = mods.proj(mods.encoder(early_fusion(x))).mean(1)  # (B, d)
            s = float(sigreg(z_bar, z_bar[:0], dirs))
        assert float(got["sigreg"]) == pytest.approx(s, rel=1e-6)
        assert torch.equal(got["pred"], flat["pred"])  # the data point alone differs
        assert float(got["loss"]) == pytest.approx(
            (1 - lam) * float(got["pred"]) + lam * s, rel=1e-6
        )

    def test_the_gradient_of_mean_is_the_same_at_every_position(self):
        # S_M depends on z_bar alone: dS/dz_{n,p} = (1/T) dS/dz_bar_n for every p
        g = torch.Generator().manual_seed(0)
        z = torch.randn(16, 8, 4, generator=g, dtype=torch.float64, requires_grad=True)
        dirs = directions(32, 4, g).double()
        pts = sigreg_points(z, "mean")
        (gz,) = torch.autograd.grad(sigreg(pts, pts[:0], dirs), [z])
        zb = z.detach().mean(1).requires_grad_()
        (gb,) = torch.autograd.grad(sigreg(zb, zb[:0], dirs), [zb])
        assert torch.allclose(gz, (gb / 8)[:, None].expand_as(gz), rtol=1e-10)
        assert gz.abs().max() > 0

    def test_k3_predicts_exactly_and_meets_the_gaussian_value_under_mean(self):
        h = torch.randn(self.N, TOKEN_DIM, generator=torch.Generator().manual_seed(3))
        proj = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        with torch.no_grad():
            proj.weight.copy_(torch.eye(TOKEN_DIM))
            proj.bias.zero_()
        mods = JEPAModules(Table(h, self.T), proj, Copy(proj))
        mask = expand_masks(masks(self.N, 8), COARSE_GRID)
        dirs = directions(self.M, TOKEN_DIM, torch.Generator().manual_seed(4))
        with torch.no_grad():
            got = jepa_terms(mods, stacks(self.N), mask, dirs, 0.5, sigreg_on="mean")
            s_h = float(sigreg(h, h[:0], dirs))  # S of the Gaussian h itself
        assert float(got["pred"]) == pytest.approx(0.0, abs=1e-12)
        assert float(got["sigreg"]) == pytest.approx(s_h, rel=1e-5)
        assert float(got["loss"]) == pytest.approx(0.5 * s_h, rel=1e-5)

    def mean_s(self, draw, on: str, reps: int = 4) -> float:
        g = torch.Generator().manual_seed(7)
        total = 0.0
        for _ in range(reps):
            pts = sigreg_points(draw(g), on)
            total += float(sigreg(pts, pts[:0], directions(self.M, pts.shape[1], g)))
        return total / reps

    def k3(self, d: int):
        def draw(g: torch.Generator) -> Tensor:
            return torch.randn(self.N, 1, d, generator=g).expand(-1, self.T, -1)

        return draw

    def test_k3_under_mean_is_the_healthy_value(self):
        assert self.mean_s(self.k3(TOKEN_DIM), "mean") == pytest.approx(
            self.HEALTHY, abs=0.1
        )

    def test_k3_under_flat_at_d_4_is_about_8_4(self):
        # Sigma = (1 1^T) x I_4 of rank 4 in D = 1024: excess 7.3, penalized
        assert self.mean_s(self.k3(4), "flat") == pytest.approx(8.37, abs=0.3)

    def test_independent_tokens_under_mean_are_about_51_5(self):
        def draw(g: torch.Generator) -> Tensor:
            return torch.randn(self.N, self.T, TOKEN_DIM, generator=g)

        assert self.mean_s(draw, "mean") == pytest.approx(51.5, abs=1.0)


class TestRhoIn:
    def test_independent_tokens_of_equal_variance_give_t_minus_1_over_t(self):
        z = torch.randn(4096, 16, 8, generator=torch.Generator().manual_seed(0))
        assert float(rho_in(z)) == pytest.approx(15 / 16, abs=0.01)

    def test_a_term_of_the_position_plus_one_vector_a_stack_gives_zero(self):
        g = torch.Generator().manual_seed(0)
        pos = torch.randn(1, 16, 8, generator=g)
        z = pos + torch.randn(64, 1, 8, generator=g)  # position + one per stack
        assert float(rho_in(z)) == pytest.approx(0.0, abs=1e-12)


class TestEvalMasksOfTheFamily:
    """V18c's evaluation masks (``exp.lib.regions``) are of $F_k$ (Fable
    derived it; checked here) and spread to the same pixels at every grid."""

    @pytest.mark.parametrize("name", sorted(regions.EVAL_MASKS))
    def test_each_is_a_row_of_its_family(self, name):
        mask = regions.EVAL_MASKS[name]
        family = mask_family(int(mask.sum()), COARSE_GRID)
        assert bool((family == mask).all(1).any())

    @pytest.mark.parametrize("name", sorted(regions.EVAL_MASKS))
    def test_the_same_pixels_at_patch_16_8_and_4(self, name):
        mask = regions.EVAL_MASKS[name]
        want = regions.pixel_mask(mask, 64)
        assert torch.equal(pixel_mask(mask, COARSE_GRID, 64), want)
        for s in (1, 2, 4):
            fine = expand_masks(mask[None], s)[0]
            assert torch.equal(pixel_mask(fine, COARSE_GRID * s, 64), want)
