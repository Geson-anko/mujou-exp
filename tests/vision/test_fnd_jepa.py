"""``exp.vision.fnd_jepa`` (``plans/foundation.md`` 3) against theory's
oracles: the tubelets' boxes and positions against ``PatchEmbed`` and
``Predictor.st_positions`` (to the bit), the masks' law (the blocks' sides of
V-JEPA's rule, each block's length and place uniform, the law invariant
under the reversal of the times and the reflections of the grid, the tubes of
T0 = V-JEPA's multi-block, one time = the image multi-block), the spacings,
the attention with padding against ``Block`` and against the stack alone,
the hidden pixels' blindness (P4) and the open leak of the time side (P5),
the loss by hand of each stack alone, ``time_r2``'s two ends, SIGReg's K3
and time-constant penalties $0.241 N (1/q - 1/D)$ through the square $W$,
and the code of a checkpoint."""

import json
import math
from dataclasses import replace

import pytest
import torch
import yaml
from torch import Tensor

from exp.lib import ckpt
from exp.lib.randproj import gaussian_projection
from exp.lib.sigreg import expected_epps_pulley, sigreg
from exp.vision.compose import to_dict
from exp.vision.fnd_jepa import (
    ARMS,
    ASPECT,
    EMBED_SEED,
    LONG,
    SHORT,
    TOKEN_DIM,
    BlockKind,
    FndCode,
    attend,
    block_sides,
    build_fnd,
    complete,
    context_index,
    draw_blocks,
    draw_masks,
    draw_spacings,
    encode,
    fnd_terms,
    load_fnd,
    targets,
    time_r2,
    z_fn,
)
from exp.vision.iid import draw_lags
from exp.vision.jepa import (
    W_SEED,
    JEPAModules,
    Mix,
    Predictor,
    TokenEncoder,
    TubeletEmbed,
    build_jepa,
    directions,
    jepa_mup_lr,
    jepa_mup_roles,
    sincos_1d,
    sincos_3d,
)
from exp.vision.models import Block, PatchEmbed, early_fusion, sincos_2d
from exp.vision.schema import DataConf, ModelConf, OptimConf, RunConf, ToursConf
from tests.vision.test_iid import under_chi2

SIZE, FRAMES, PATCH = 16, 8, 4  # grid 4 x 4, 4 times: 64 tokens
TIMES, GRID = 4, 4


def conf_of(seed: int = 0) -> RunConf:
    return RunConf(
        name="test",
        seed=seed,
        device="cpu",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=FRAMES)),
        model=ModelConf(width=128, patch=PATCH, param="mup"),
        optim=OptimConf(kind="rmsprop"),
    )


def stacks(b: int, seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        256, (b, FRAMES, 3, SIZE, SIZE), generator=g, dtype=torch.uint8
    )


def vit(mods: JEPAModules) -> TokenEncoder:
    assert isinstance(mods.encoder, TokenEncoder)
    return mods.encoder


def block_of(mods: JEPAModules) -> Block:
    """The encoder's first block."""
    block = vit(mods).vit[0].blocks[0]
    assert isinstance(block, Block)
    return block


def masks(b: int, timed: bool = True, seed: int = 1) -> Tensor:
    gen = torch.Generator().manual_seed(seed)
    return draw_masks(b, TIMES, GRID, timed, gen)[0]


# -- tokens and positions -------------------------------------------------------


class TestTubelets:
    def test_one_time_is_the_patch_embedding_of_the_same_weight(self):
        torch.manual_seed(0)
        tub = TubeletEmbed(
            frames=FRAMES, tubelet=FRAMES, patch=PATCH, size=SIZE, dim=32
        )
        pe = PatchEmbed(channels=3 * FRAMES, patch=PATCH, size=SIZE, dim=32)
        pe.proj.load_state_dict(tub.proj.state_dict())
        tub.pos.zero_()  # the linear parts alone
        pe.pos.zero_()
        x = early_fusion(stacks(3))
        assert torch.equal(tub(x), pe(x))

    def test_a_token_is_the_linear_map_of_its_box(self):
        torch.manual_seed(1)
        tub = TubeletEmbed(frames=FRAMES, tubelet=2, patch=PATCH, size=SIZE, dim=32)
        x = stacks(2)
        out = tub(early_fusion(x)) - tub.pos  # (B, 64, 32)
        for j in range(TIMES):
            for r in range(GRID):
                for c in range(GRID):
                    box = x[
                        :, 2 * j : 2 * j + 2, :, 4 * r : 4 * r + 4, 4 * c : 4 * c + 4
                    ]
                    want = tub.proj(box.reshape(2, -1).float() / 255)
                    w = j * GRID * GRID + r * GRID + c  # time-major
                    torch.testing.assert_close(out[:, w], want, rtol=1e-5, atol=1e-5)

    def test_a_pixel_reaches_only_its_tubelet(self):
        torch.manual_seed(2)
        tub = TubeletEmbed(frames=FRAMES, tubelet=2, patch=PATCH, size=SIZE, dim=32)
        x = stacks(1)
        y = x.clone()
        y[0, 5, 1, 9, 2] ^= 0xFF  # frame 5 (time 2), row 9 col 2 (position 2 * 4 + 0)
        diff = (tub(early_fusion(x)) - tub(early_fusion(y))).abs().amax(-1)[0]
        changed = (diff > 0).nonzero().flatten().tolist()
        assert changed == [2 * 16 + 2 * 4 + 0]

    def test_no_whole_tubelets_are_refused(self):
        with pytest.raises(ValueError, match="whole tubelets"):
            TubeletEmbed(frames=7, tubelet=2, patch=PATCH, size=SIZE, dim=32)

    def test_the_positions_are_the_predictors_without_psi_to_the_bit(self):
        pred = Predictor(width_in=32, grid=GRID, dim=64)
        ell = torch.tensor([0.0, 2.5, 7.0])
        psi = sincos_1d(ell, 64).to(pred.pos)
        table = sincos_3d(TIMES, GRID, 64)
        assert torch.equal(table[None] + psi[:, None], pred.st_positions(TIMES, ell))

    def test_the_time_part_flips_its_sines_under_the_reversal(self):
        dim = 64
        e = sincos_3d(TIMES, GRID, dim).double().unflatten(0, (TIMES, -1))
        phi = e - sincos_2d((GRID, GRID), dim).double()  # phi(u_j) at every p
        torch.testing.assert_close(phi, phi[:, :1].expand_as(phi), atol=1e-6, rtol=0)
        rev = phi.flip(0)  # u -> -u
        half = dim // 2
        torch.testing.assert_close(rev[..., half:], phi[..., half:], atol=1e-6, rtol=0)
        torch.testing.assert_close(rev[..., :half], -phi[..., :half], atol=1e-6, rtol=0)


# -- masks ----------------------------------------------------------------------


ONE = BlockKind("one", 1, 0.15)  # one block of 15 %: its box read off the mask


def boxes(m: Tensor) -> dict[str, Tensor]:
    """The sides, length, first time, top and left of single-block masks
    ``(B, T, g, g)``."""
    t = m.any(3).any(2)  # (B, T)
    r = m.any(3).any(1)  # (B, g) rows
    c = m.any(2).any(1)  # (B, g) cols
    first = lambda a: a.float().argmax(1)  # noqa: E731
    return {
        "length": t.sum(1),
        "start": first(t),
        "h": r.sum(1),
        "top": first(r),
        "w": c.sum(1),
        "left": first(c),
    }


class TestBlocks:
    @pytest.mark.parametrize(
        ("kind", "aspect", "sides"),
        [
            (SHORT, 0.75, (3, 3)),  # n = 9: sqrt(6.75) = 2.60, sqrt(12) = 3.46
            (SHORT, 1.5, (4, 2)),  # sqrt(13.5) = 3.67, sqrt(6) = 2.45
            (LONG, 0.75, (6, 8)),  # n = 44: sqrt(33) = 5.74, sqrt(58.7) = 7.66
            (LONG, 1.5, (8, 5)),  # sqrt(66) = 8.12, sqrt(29.3) = 5.42
        ],
    )
    def test_the_sides_of_v_jepas_rule_by_hand(self, kind, aspect, sides):
        h, w = block_sides(kind, 8, torch.tensor([aspect]))
        assert (int(h), int(w)) == sides

    def test_the_sides_follow_the_uniform_aspect(self):
        # 8 x 8, n = 9: h = 4 iff 9 a >= 3.5^2, w = 2 iff 9 / a < 2.5^2
        m = draw_blocks(20000, ONE, 1, 8, True, torch.Generator().manual_seed(0))
        b = boxes(m)
        cell = (b["h"] - 3) * 2 + (3 - b["w"])  # (3,3) 0, (4,3) 2, (4,2) 3
        lo, hi = ASPECT
        a1, a2 = 3.5**2 / 9, 9 / 2.5**2
        probs = torch.tensor([(a1 - lo), 0.0, (a2 - a1), (hi - a2)]) / (hi - lo)
        counts = torch.bincount(cell, minlength=4)
        assert int(counts[1]) == 0
        assert under_chi2(counts[[0, 2, 3]], probs[[0, 2, 3]])

    def test_the_place_is_uniform_where_the_block_fits(self):
        m = draw_blocks(20000, ONE, 1, 8, True, torch.Generator().manual_seed(1))
        b = boxes(m)
        for side, at in (("h", "top"), ("w", "left")):
            for v in b[side].unique().tolist():
                on = b[side] == v
                counts = torch.bincount(b[at][on], minlength=8 - v + 1)
                assert len(counts) == 8 - v + 1
                assert under_chi2(counts, torch.full((8 - v + 1,), 1 / (8 - v + 1)))

    def test_the_length_and_the_first_time_are_uniform(self):
        m = draw_blocks(30000, ONE, TIMES, 8, True, torch.Generator().manual_seed(2))
        b = boxes(m)
        cells = [(n, s) for n in range(1, TIMES + 1) for s in range(TIMES - n + 1)]
        index = {c: i for i, c in enumerate(cells)}
        got = torch.tensor(
            [index[(int(n), int(s))] for n, s in zip(b["length"], b["start"])]
        )
        probs = torch.tensor([1 / TIMES / (TIMES - n + 1) for n, _ in cells])
        assert under_chi2(torch.bincount(got, minlength=len(cells)), probs)

    def test_each_time_is_hidden_by_the_closed_form_and_the_law_is_reversible(self):
        # one block: P(j) = (1/T) sum_n #{s: s <= j < s + n} / (T - n + 1)
        m = draw_blocks(40000, ONE, TIMES, 8, True, torch.Generator().manual_seed(3))
        p = [
            sum(
                sum(1 for s in range(TIMES - n + 1) if s <= j < s + n) / (TIMES - n + 1)
                for n in range(1, TIMES + 1)
            )
            / TIMES
            for j in range(TIMES)
        ]
        assert p == pytest.approx(p[::-1])  # the reversal j -> T - 1 - j
        rate = m.any(3).any(2).double().mean(0)
        se = (torch.tensor(p) * (1 - torch.tensor(p)) / len(m)).sqrt()
        assert ((rate - torch.tensor(p)).abs() < 5 * se).all()

    @pytest.mark.parametrize("kind", [SHORT, LONG])
    def test_the_masks_law_is_invariant_under_reversal_and_reflections(self, kind):
        m = draw_blocks(20000, kind, TIMES, 8, True, torch.Generator().manual_seed(4))
        rate = m.double().mean(0)  # (T, g, g)
        se = (rate * (1 - rate) / len(m)).sqrt().clamp(min=1e-3)
        for flipped in (rate.flip(0), rate.flip(1), rate.flip(2)):
            assert ((rate - flipped).abs() < 5 * math.sqrt(2) * se).all()

    def test_t0_hides_tubes_of_the_image_multi_block_v_jepas(self):
        # untimed: no time is drawn, so the draws are the image's (T = 1)
        tube = draw_blocks(
            500, SHORT, TIMES, 8, False, torch.Generator().manual_seed(5)
        )
        image = draw_blocks(500, SHORT, 1, 8, False, torch.Generator().manual_seed(5))
        assert torch.equal(tube, image.expand(-1, TIMES, -1, -1))

    def test_one_time_draws_the_image_multi_block_timed_or_not(self):
        for kind in (SHORT, LONG):
            timed = draw_blocks(500, kind, 1, 8, True, torch.Generator().manual_seed(6))
            plain = draw_blocks(
                500, kind, 1, 8, False, torch.Generator().manual_seed(6)
            )
            assert torch.equal(timed, plain)

    def test_the_image_multi_block_is_a_union_of_its_rectangles(self):
        # SHORT on 8 x 8: at most 8 blocks of 9 or 12 or 8 positions
        m = draw_blocks(2000, SHORT, 1, 8, False, torch.Generator().manual_seed(7))
        counts = m.flatten(1).sum(1)
        assert int(counts.min()) >= 8 and int(counts.max()) <= 8 * 12
        assert len(counts.unique()) > 10  # the count hidden is not fixed

    def test_no_mask_hides_every_token(self):
        # LONG on 2 x 2: n = 2 positions a block, often all four covered
        gen = torch.Generator().manual_seed(8)
        m = draw_blocks(5000, LONG, 1, 2, False, gen)
        assert not m.flatten(1).all(1).any()
        assert m.flatten(1).any(1).all()

    def test_a_batch_is_half_short_then_long(self):
        m, kind = draw_masks(9, TIMES, GRID, True, torch.Generator().manual_seed(9))
        assert m.shape == (9, TIMES * GRID * GRID)
        assert kind.tolist() == [0] * 5 + [1] * 4


class TestSpacings:
    def test_t0_and_s1_are_of_one_frame(self):
        gen = torch.Generator().manual_seed(0)
        for arm in ("T0", "S1"):
            assert torch.equal(
                draw_spacings(7, ARMS[arm], 128, gen), torch.ones(7).long()
            )

    def test_ss_is_the_rounded_log_uniform(self):
        a = draw_spacings(1000, ARMS["Ss"], 128, torch.Generator().manual_seed(1))
        b = draw_lags(1000, 1, 128, torch.Generator().manual_seed(1))
        assert torch.equal(a, b)


# -- modules and the loss -------------------------------------------------------


class TestModules:
    def test_the_blocks_and_the_predictor_are_v18cs_the_embedding_its_own(self):
        conf = conf_of(seed=3)
        mods = build_fnd(conf, seed=3)
        v18c = build_jepa(conf, seed=3, token_dim=TOKEN_DIM, proj="fixed")
        a, b = mods.encoder.state_dict(), v18c.encoder.state_dict()
        embed = {k for k in a if k.startswith("parts.0.embed.")}
        assert embed == {"parts.0.embed.proj.weight", "parts.0.embed.proj.bias"}
        assert all(torch.equal(a[k], b[k]) for k in a if k not in embed)
        pa, pb = mods.predictor.state_dict(), v18c.predictor.state_dict()
        assert all(torch.equal(pa[k], pb[k]) for k in pa)
        w = a["parts.0.embed.proj.weight"]  # LeCun's, truncated at 2 std
        std = 1 / math.sqrt(3 * 2 * PATCH * PATCH)
        assert float(w.abs().max()) <= 2 * std
        assert float(w.std()) == pytest.approx(0.8796 * std, rel=0.05)
        assert not a["parts.0.embed.proj.bias"].any()
        again = build_fnd(conf, seed=3).encoder.state_dict()[
            "parts.0.embed.proj.weight"
        ]
        other = build_fnd(conf, seed=4).encoder.state_dict()[
            "parts.0.embed.proj.weight"
        ]
        assert torch.equal(again, w) and not torch.equal(other, w)
        assert EMBED_SEED not in (0, 3)  # its own draw, not the blocks'

    def test_w_is_square_of_its_seed_and_no_parameter(self):
        mods = build_fnd(conf_of(), seed=0)
        assert isinstance(mods.mix, Mix)
        d = TIMES * GRID * GRID * TOKEN_DIM
        assert torch.equal(mods.mix.weight, gaussian_projection(d, d, W_SEED))
        held = {id(p) for p in mods.parameters()}
        assert id(mods.mix.weight) not in held and id(mods.proj.weight) not in held

    def test_the_embedding_is_mups_input(self):
        mods = build_fnd(conf_of(), seed=0)
        embed = vit(mods).vit[0].embed
        assert embed is not None
        w = embed.proj.weight
        assert jepa_mup_roles(mods)[w] == "input"
        assert jepa_mup_lr(mods, 128)[w] == 1.0

    def test_the_attention_is_the_blocks(self):
        mods = build_fnd(conf_of(), seed=0)
        block = block_of(mods)
        x = torch.randn(3, 10, 128, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(attend(block, x, 2), block(x), rtol=1e-5, atol=1e-5)

    def test_the_padding_is_no_key(self):
        mods = build_fnd(conf_of(), seed=0)
        block = block_of(mods)
        x = torch.randn(2, 10, 128, generator=torch.Generator().manual_seed(1))
        valid = torch.arange(10)[None] < torch.tensor([[6], [10]])
        out = attend(block, x, 2, valid)
        torch.testing.assert_close(
            out[0, :6], block(x[:1, :6])[0], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(out[1], block(x[1:])[0], rtol=1e-5, atol=1e-5)

    def test_the_context_index(self):
        hidden = masks(6)
        keep, valid, slot = context_index(hidden)
        for b in range(6):
            vis = (~hidden[b]).nonzero().flatten()
            assert torch.equal(keep[b, valid[b]], vis)
            assert torch.equal(keep[b][slot[b, vis]], vis)
            assert hidden[b, keep[b, ~valid[b]]].all()  # the padding is hidden


class TestLoss:
    def test_each_stacks_error_by_hand_alone(self):
        conf = conf_of()
        mods = build_fnd(conf, seed=0)
        x, hidden = stacks(4), masks(4)
        d = TIMES * GRID * GRID * TOKEN_DIM
        dirs = directions(64, d, torch.Generator().manual_seed(0))
        ell = torch.tensor([0.0, 1.0, 3.0, 7.0])
        terms = fnd_terms(mods, x, hidden, dirs, 0.3, heads=2, times=TIMES, ell=ell)
        with torch.no_grad():
            for b in range(4):
                img = early_fusion(x[b : b + 1])
                t = mods.proj(encode(mods, img, 2))
                vis = (~hidden[b]).nonzero().flatten()[None]
                ctx = encode(mods, img, 2, vis)  # no padding
                slot = context_index(hidden[b : b + 1])[2]
                t_hat = complete(
                    mods, ctx, slot, hidden[b : b + 1], TIMES, ell[b : b + 1]
                )
                err = (t_hat - t)[0, hidden[b]].square().mean()
                torch.testing.assert_close(terms["per"][b], err, rtol=1e-4, atol=1e-6)
        assert mods.mix is not None
        z = mods.mix(terms["tokens"].flatten(1))
        torch.testing.assert_close(terms["z"], z)
        s = sigreg(z, z[:0], dirs)
        torch.testing.assert_close(terms["sigreg"], s.detach())
        torch.testing.assert_close(terms["pred"], terms["per"].mean())
        want = 0.7 * terms["pred"] + 0.3 * terms["sigreg"]
        torch.testing.assert_close(terms["loss"], want)

    def test_the_gradient_reaches_the_target_side(self):
        # no stop-gradient: the hidden tubelets' pixels reach the loss only
        # through the targets (P4), and they get a gradient
        mods = build_fnd(conf_of(), seed=0)
        x = (stacks(2).float() / 255).requires_grad_()
        hidden = masks(2)
        dirs = directions(8, 1024, torch.Generator().manual_seed(0))
        fnd_terms(mods, x, hidden, dirs, 0.0, heads=2, times=TIMES)["loss"].backward()
        assert x.grad is not None
        per = hidden.reshape(2, TIMES, GRID, GRID).repeat_interleave(2, 1)
        pix = per.repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
        assert (
            x.grad.permute(0, 2, 1, 3, 4)[pix[:, None].expand(-1, 3, -1, -1, -1)]
            .abs()
            .sum()
            > 0
        )

    def test_checkpointing_gives_the_same_gradient(self):
        x, hidden = stacks(2), masks(2)
        dirs = directions(8, 1024, torch.Generator().manual_seed(0))
        grads = []
        for again in (False, True):
            mods = build_fnd(conf_of(), seed=0)
            t = fnd_terms(mods, x, hidden, dirs, 0.5, heads=2, times=TIMES, again=again)
            t["loss"].backward()
            grads.append(
                torch.cat(
                    [torch.zeros(0)]
                    + [
                        p.grad.flatten()
                        for p in mods.parameters()
                        if p.grad is not None
                    ]
                )
            )
        torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-7)

    def test_the_hidden_pixels_reach_no_prediction(self):
        # P4: the pixels of the hidden tubelets (both arms' masks) change nothing
        mods = build_fnd(conf_of(), seed=0)
        for timed in (False, True):
            x, hidden = stacks(4), masks(4, timed)
            y = x.clone()
            per = hidden.reshape(4, TIMES, GRID, GRID)
            pix = per.repeat_interleave(2, 1).repeat_interleave(PATCH, 2)
            pix = pix.repeat_interleave(PATCH, 3)[:, :, None].expand_as(x)
            y[pix] = 255 - y[pix]
            outs = []
            with torch.no_grad():
                for v in (x, y):
                    img = early_fusion(v)
                    keep, valid, slot = context_index(hidden)
                    ctx = encode(mods, img, 2, keep, valid)
                    outs.append(complete(mods, ctx, slot, hidden, TIMES))
            assert torch.equal(outs[0], outs[1])

    def test_the_time_sides_leak_is_open(self):
        # P5, a record: the target of a hidden time moves with a visible time
        mods = build_fnd(conf_of(), seed=0)
        x = stacks(1)
        y = x.clone()
        y[:, 0:2] = 255 - y[:, 0:2]  # time 0 only
        with torch.no_grad():
            ta = mods.proj(encode(mods, early_fusion(x), 2))
            tb = mods.proj(encode(mods, early_fusion(y), 2))
        later = slice(GRID * GRID, None)  # times 1..3
        assert (ta[:, later] - tb[:, later]).abs().amax() > 1e-4


# -- time_r2 and SIGReg's penalties ---------------------------------------------


class TestTimeR2:
    def test_tokens_constant_along_the_times_give_one(self):
        g = torch.Generator().manual_seed(0)
        t = torch.randn(50, 1, 16, 4, generator=g).expand(-1, TIMES, -1, -1)
        assert float(time_r2(t.flatten(1, 2), TIMES)) == pytest.approx(1.0, abs=1e-12)

    def test_independent_times_of_equal_variance_give_one_over_t(self):
        g = torch.Generator().manual_seed(1)
        t = torch.randn(4000, TIMES * 16, 4, generator=g)
        assert float(time_r2(t, TIMES)) == pytest.approx(1 / TIMES, abs=0.01)

    def test_it_is_blind_to_offsets_and_the_times_order(self):
        g = torch.Generator().manual_seed(2)
        t = torch.randn(30, TIMES, 16, 4, generator=g)
        t = t + t[:, :1]  # some share
        off = torch.randn(1, TIMES, 16, 4, generator=g)
        a = time_r2(t.flatten(1, 2), TIMES)
        torch.testing.assert_close(time_r2((t + off).flatten(1, 2), TIMES), a)
        torch.testing.assert_close(time_r2(t.flip(1).flatten(1, 2), TIMES), a)


class TestSigRegPenalties:
    """$z = W \\mathrm{vec}(t)$ of 128 stacks of 256 tokens of $d$ = 16 by the
    square $W$ (4096): the excess of SIGReg over isotropic tokens is about
    $0.241 N (1/q - 1/D)$ of the rank $q$ of vec(t) (``plans/foundation.md``
    3): 1.92 for K3 ($q$ = 16), 0.023 for tokens constant along the times
    ($q$ = 1024). The approximation is of second order in the spread of
    $a^\\top z$'s variance over the directions (its curvature $n \\cdot
    (3/16)\\sqrt\\pi\\,(2/3)^{5/2} = 0.1206\\,n$) and of the expected spread
    $2/q$; the 1024 directions' own spread is within 5 %."""

    N, D, M = 128, 4096, 1024

    def maps(self) -> tuple[Tensor, dict[str, Tensor]]:
        w = gaussian_projection(self.D, self.D, W_SEED)
        a = directions(self.M, self.D, torch.Generator().manual_seed(5))
        wa = (a.double() @ w.double()).reshape(self.M, TIMES, 64, TOKEN_DIM)
        var = {  # a^T z's variance of each direction, unit tokens
            "iso": wa.square().sum((1, 2, 3)),
            "k3": wa.sum((1, 2)).square().sum(1),  # t = c for every token
            "time": wa.sum(1).square().sum((1, 2)),  # t(j, p) = sigma(p)
        }
        return a, var

    def expected(self, var: Tensor) -> float:
        m = len(var)
        e = expected_epps_pulley(
            torch.zeros(m, dtype=torch.float64),
            var,
            torch.zeros(0, m, dtype=torch.float64),
            torch.zeros(0, dtype=torch.float64),
            self.N,
        )
        return float(e.mean())

    def test_the_penalties_of_the_derivation(self):
        _, var = self.maps()
        e = {k: self.expected(v) for k, v in var.items()}
        approx = {q: 0.241 * self.N * (1 / q - 1 / self.D) for q in (16, 1024)}
        assert approx[16] == pytest.approx(1.92, abs=0.005)
        assert approx[1024] == pytest.approx(0.0226, abs=0.0005)
        assert e["k3"] - e["iso"] == pytest.approx(approx[16], rel=0.1)
        assert e["time"] - e["iso"] == pytest.approx(approx[1024], rel=0.12)

    def test_sigreg_of_the_mixed_tokens_meets_its_expectation(self):
        a, var = self.maps()
        mix = Mix(gaussian_projection(self.D, self.D, W_SEED))
        g = torch.Generator().manual_seed(9)
        shapes = {"iso": (TIMES, 64), "k3": (1, 1), "time": (1, 64)}
        for kind, (tt, pp) in shapes.items():
            vals = []
            for _ in range(120):
                t = torch.randn(self.N, tt, pp, TOKEN_DIM, generator=g)
                z = mix(t.expand(-1, TIMES, 64, -1).reshape(self.N, -1))
                vals.append(float(sigreg(z, z[:0], a)))
            v = torch.tensor(vals, dtype=torch.float64)
            se = float(v.std() / math.sqrt(len(v)))
            assert abs(float(v.mean()) - self.expected(var[kind])) < 5 * se


# -- the code of a checkpoint ---------------------------------------------------


def test_the_code_of_a_checkpoint_is_its_target_side(tmp_path):
    conf = conf_of(seed=4)
    mods = build_fnd(conf, seed=4)
    with torch.no_grad():
        for p in mods.parameters():
            p.add_(0.01 * torch.randn_like(p))
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump(to_dict(replace(conf, device="cuda")))
    )
    (run / "jepa.json").write_text(json.dumps({"token_dim": TOKEN_DIM, "tubelet": 2}))
    path = ckpt.save_eval(
        run, program="fnd", group="iid", t=12.8, frame=128, modules=mods.modules()
    )
    x = stacks(5)
    _, _, loaded = load_fnd(path, "cpu")
    t, z = targets(mods, x, 2)
    torch.testing.assert_close(z_fn(str(path), "cpu")(x), z)
    torch.testing.assert_close(FndCode(loaded, 2).tokens(x), t)
    assert z.shape == (5, TIMES * GRID * GRID * TOKEN_DIM)
    assert t.shape == (5, TIMES * GRID * GRID, TOKEN_DIM)
