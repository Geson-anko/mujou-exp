"""The evaluation afterwards of an encoder of the packed layout (M17) on a
tiny run of a small Walking Tours: the files, the keys, the objective, the
checkpoints it reads, the resume, and the training windows."""

import dataclasses
import math
from pathlib import Path
from typing import override

import pytest
import torch
from torch import nn

from exp.lib.rundir import Exit
from exp.video import loop
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.main import local_job
from exp.video.measure import encoder_eval
from exp.video.measure.encoder_eval import (
    EncoderEval,
    _Code,
    _completion_objective,
    _objective,
    _Standardized,
    stream_windows,
)
from exp.video.measure.objective import FLOOR, PENALTY
from exp.video.measure.posthoc import Parent, as_float
from exp.video.measure.restore import last_slot, picture_of, token_decoder
from exp.video.programs.vision import DATA_ENV, PROGRAM
from tests.video.data.test_walking_tours import numbered
from tests.video.packed import packed_conf, smooth_tours, tiny

LENGTHS = {"A": 120, "B": 80, "H": 200}


@pytest.fixture(scope="module")
def parent(tmp_path_factory) -> tuple[Path, Path]:
    """A finished tiny run with two checkpoints, and the data's root."""
    tmp = tmp_path_factory.mktemp("m17")
    root = smooth_tours(tmp / "data" / "d", LENGTHS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_ENV, str(root.parent))
        mp.setattr(loop, "assert_free_threaded", lambda where: None)
        job = local_job(tiny(root.name), tmp / "run")
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
    return tmp / "run", root


def _eval(parent, out: Path, **kw) -> EncoderEval:
    run_dir, root = parent
    p = Parent.read(run_dir, device="cpu")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_ENV, str(root.parent))
        return EncoderEval(p, p.conf, out, root=root, windows=16, steps=3, **kw)


def _files(out: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


_RESTORE = [f"restore_{w}_{k}" for w in ("short", "long") for k in ("mask", "all")] + [
    "restore_target_short_mask",
    "restore_target_long_mask",
    "restore_target_all",
]


class _Constant(nn.Module):
    """A token decoder of every token to the patch of the constant ``value``."""

    def __init__(self, value: float, channels: int, patch: int) -> None:
        super().__init__()
        self._value, self._shape = value, (channels, patch, patch)

    @override
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.full((len(tokens), *self._shape), self._value)


class TestRestoreKeys:
    """The restoration keys of a line (``EncoderEval._restore``) of the ViT
    before it learns."""

    def _restore(self, e: EncoderEval, dec: nn.Module) -> tuple[dict, list]:
        v = e.conf.vision
        assert v is not None
        f0 = as_float(e._measures.windows["val"][:, v.frames - 1])
        line: dict = {}
        rows = e._restore(e._mods, "val", dec, f0, line)
        return line, rows

    def test_the_mean_image_is_one_on_every_key(self, parent, tmp_path):
        """A decoder that draws the mean image (a constant one here) restores
        and reaches the ceiling at exactly the mean image's error, on the
        masked pixels and the whole frame, blurred or not."""
        e = _eval(parent, tmp_path / "o")
        e._mean = torch.full_like(e._mean, 0.4)
        line, _ = self._restore(e, _Constant(0.4, 3, 4))
        assert set(line) == {*_RESTORE, *(f"{k}_blur" for k in _RESTORE)}
        for k, x in line.items():
            assert x == pytest.approx(1.0, rel=1e-6), k

    def test_the_ceiling_sees_no_predictor_and_the_rows(self, parent, tmp_path):
        """The ceiling is the token decoder on the target tokens of the whole
        stack: zeroing the predictor changes the restorations, not the
        ceiling's keys; the rows are the ceiling, then per view the context
        and its restoration."""
        e = _eval(parent, tmp_path / "o")
        torch.manual_seed(0)
        dec = token_decoder(16, 4, 3)
        line, rows = self._restore(e, dec)
        with torch.no_grad():
            for p in e._mods.predictor.parameters():
                p.zero_()
        held, _ = self._restore(e, dec)
        for k in line:
            same = held[k] == pytest.approx(line[k], rel=1e-6)
            assert same == k.startswith("restore_target"), k
        v = e.conf.vision
        assert v is not None
        x = e._measures.windows["val"][:, : v.frames]
        assert len(rows) == 1 + 2 * 2
        assert all(len(r) == min(encoder_eval.SHOWN, len(x)) for r in rows)
        with torch.no_grad():
            toks = last_slot(e._mods, e._mods.tokens(x))
            ceiling = picture_of(dec(toks.flatten(0, 1)), len(x), e._mods.grid)
        torch.testing.assert_close(rows[0], ceiling.clamp(0, 1)[: len(rows[0])])


class TestEncoderEval:
    def test_the_units_lines_and_pictures(self, parent, tmp_path):
        e = _eval(parent, tmp_path / "o")
        assert len(e) == 3  # the yardsticks, two checkpoints
        lines = [ln for _ in range(len(e)) for ln in e.step()]
        assert [ln["code"] for ln in lines] == ["pca"] * 2 + ["init"] * 2 + ["vit"] * 4
        vit = lines[-2]
        assert vit["split"] == "val" and vit["t"] == 1.0
        for k in (
            "recon",
            "recon_blur",
            "recon_fit",
            "dyn_1",
            "dyn_2",
            "copy_1",
            "dyn_1/copy_1",
            "dynw_1",
            "copyw_1",
            "img_dyn_1",
            "img_dyn_1/copy",
            "restore_short_mask",
            "restore_short_all",
            "restore_long_mask",
            "restore_long_all",
            "restore_target_short_mask",
            "restore_target_long_mask",
            "restore_target_all",
            *(f"{k}_blur" for k in _RESTORE),
            "rank",
            "z_std_min",
            "z_spread",
            "sens_dither",
            "sens_shift",
            "sens_dither_pix",
            "sens_shift_pix",
            "alpha",
            "objective",
        ):
            assert k in vit, k
        for ln in lines:  # every code of a split sees the same stacks and dither
            assert (
                ln["sens_dither_pix"]
                == lines[1 if ln["split"] == "test" else 0]["sens_dither_pix"]
            )
            assert ln["sens_dither"] > 0 and ln["sens_shift"] > 0
        assert "objective" not in lines[-1]  # test
        comp = {
            f"{c}_{k}{s}"
            for c in ("comp", "ncomp")
            for k in ("short", "long", "future")
            for s in ("", "_vis", "_mean", "_pos")
        }
        prefixed = {f"{p}_{k}" for p in ("calib", "fit") for k in comp}
        assert comp | prefixed <= set(vit)
        assert comp <= set(lines[-1]) and not prefixed & set(lines[-1])  # test
        init = lines[2]
        assert init["code"] == "init" and comp | prefixed <= set(init)
        pca = lines[0]
        assert {"recon_lin", "recon_lin_blur", "recon_blur"} <= set(pca)
        assert "restore_short_all" not in pca
        assert all(ln["recon_blur"] > 0 for ln in lines)
        assert not comp & set(pca)
        assert vit["objective"] == pytest.approx(_objective(vit, pca))
        names = set(_files(tmp_path / "o"))
        for stem in ("pca", "init", "00000.5", "00001.0"):
            assert {f"{stem}.png", f"{stem}-dyn.png"} <= names
        assert "init-restore.png" in names and "pca-restore.png" not in names
        final = e.final()
        assert final["decode"]["val"] == vit
        assert set(final["baselines"]["test"]) == {"pca", "init"}

    def test_the_encoder_is_held_bit_for_bit(self, parent, tmp_path, monkeypatch):
        """Every module of each code (the ViT before it learns, each
        checkpoint) has the same bytes after its evaluation (the decoders,
        the ridge and the token decoder fit) as before."""
        held = []
        evaluate = EncoderEval._evaluate

        def watched(self, code, *args, **kw):
            if code.mods is None:
                return evaluate(self, code, *args, **kw)
            before = {
                f"{n}.{k}": x.detach().clone()
                for n, m in code.mods.named().items()
                for k, x in m.state_dict().items()
            }
            out = evaluate(self, code, *args, **kw)
            after = {
                f"{n}.{k}": x
                for n, m in code.mods.named().items()
                for k, x in m.state_dict().items()
            }
            assert before.keys() == after.keys()
            for k, x in before.items():
                assert torch.equal(x, after[k]), k
            held.append(code.name)
            return out

        monkeypatch.setattr(EncoderEval, "_evaluate", watched)
        e = _eval(parent, tmp_path / "o")
        for _ in range(len(e)):
            e.step()
        assert held == ["init", "vit", "vit"]

    def test_ckpts_last_reads_the_last_checkpoint_only(self, parent, tmp_path):
        e = _eval(parent, tmp_path / "o", ckpts="last")
        assert len(e) == 2
        e.step()
        assert [ln["t"] for ln in e.step()] == [1.0, 1.0]

    def test_a_resume_writes_the_bytes_of_one_never_stopped(self, parent, tmp_path):
        full = _eval(parent, tmp_path / "a")
        for _ in range(len(full)):
            full.step()
        part = _eval(parent, tmp_path / "b")
        part.step()
        part.step()
        with (tmp_path / "b" / "metrics.jsonl").open("a") as f:
            f.write('{"cut": 1}\n')  # a line of the unit that was cut
        rest = _eval(parent, tmp_path / "b", done=2)
        rest.step()
        a, b = _files(tmp_path / "a"), _files(tmp_path / "b")
        del a["config.json"], b["config.json"]
        assert a == b


@pytest.fixture(scope="module")
def token_parent(tmp_path_factory) -> tuple[Path, Path]:
    """A finished tiny run of m17/tok_n16 (tokens of 8: a code of 32 x 8 =
    256 numbers, more than the training frames of 16 windows), and the
    data's root."""
    tmp = tmp_path_factory.mktemp("tok")
    root = smooth_tours(tmp / "data" / "d", LENGTHS)
    overrides = [
        "+experiment=m17/tok_n16" if o.startswith("+experiment") else o
        for o in tiny(root.name)
    ]
    overrides += ["vision.token_dim=8", "vision.token_buffer=24"]
    overrides += ["vision.token_insert=8"]  # of the 32 tokens of a stack
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DATA_ENV, str(root.parent))
        mp.setattr(loop, "assert_free_threaded", lambda where: None)
        job = local_job(overrides, tmp / "run")
        assert loop.run(PROGRAM, job, attempt=1) == Exit.OK
    return tmp / "run", root


class TestTokenCode:
    """measure.posthoc_code "tokens": a stack's code is its flattened target
    tokens, for the decoder of z, the linear dynamics and the statistics of
    z, of the ViT before it learns and of each checkpoint; the PCA of as many
    components as the training frames allow."""

    def test_the_code_is_the_flattened_tokens(self, token_parent, tmp_path):
        e = _eval(token_parent, tmp_path / "o", ckpts="last")
        mods = e._mods
        assert e.conf.measure.posthoc_code == "tokens"
        windows = e._measures.windows["val"][:3]
        z, _ = e._encode(_Code("vit", None, mods), windows)
        k = windows.shape[1] - mods.chunk[0] + 1
        stacks = torch.stack([windows[:, i : i + mods.chunk[0]] for i in range(k)], 1)
        with torch.no_grad():
            tokens = mods.tokens(stacks.flatten(0, 1))
        assert z.shape == (3, k, mods.tokens_per_view * 8) == (3, k, 256)
        torch.testing.assert_close(z.flatten(0, 1), tokens.flatten(1))

    def test_every_code_reaches_the_fits_at_its_dimension(
        self, token_parent, tmp_path, monkeypatch
    ):
        seen: dict[str, list[int]] = {"decoder": [], "dyn": []}
        fit, dyn = encoder_eval.fit_decoder, encoder_eval.linear_dyn.fit

        def fit_watched(z, *args, **kw):
            seen["decoder"].append(z.shape[1])
            return fit(z, *args, **kw)

        def dyn_watched(z0, z1, **kw):
            seen["dyn"].append(z0.shape[1])
            return dyn(z0, z1, **kw)

        monkeypatch.setattr(encoder_eval, "fit_decoder", fit_watched)
        monkeypatch.setattr(encoder_eval.linear_dyn, "fit", dyn_watched)
        e = _eval(token_parent, tmp_path / "o", ckpts="last")
        lines = [ln for _ in range(len(e)) for ln in e.step()]
        n = len(e._targets)
        assert n - 1 < 256  # the training frames allow fewer components
        assert [ln["code"] for ln in lines] == ["pca"] * 2 + ["init"] * 2 + ["vit"] * 2
        assert all(ln["pca_rank"] == n - 1 for ln in lines[:2])
        assert not any("pca_rank" in ln for ln in lines[2:])
        assert seen["decoder"] == [n - 1, 256, 256]
        assert set(seen["dyn"][-1:]) == {256} and n - 1 in seen["dyn"]
        assert all(ln["rank"] <= 256 for ln in lines)
        assert {"restore_short_all", "restore_target_all"} <= set(lines[-2])


def _with_target(e: EncoderEval) -> VisionModules:
    """The modules of ``e``'s run with I-JEPA's target encoder (vision.ema)
    whose weights differ from the online encoder's."""
    v = e.conf.vision
    assert v is not None
    conf = dataclasses.replace(
        e.conf, vision=dataclasses.replace(v, ema=0.9, token_buffer=0)
    )
    mods = build_vision(conf, seed=conf.seed)
    assert mods.target is not None
    with torch.no_grad():
        for p in mods.target.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return mods


class TestTheTargetEncoder:
    """With I-JEPA's target encoder, posthoc measures its tokens (the
    prediction's targets): the code of the decoder, the linear dynamics and
    the statistics, the token decoder's tokens, and the ceiling."""

    def test_the_code_and_the_slot_tokens_are_the_target_encoders(
        self, token_parent, tmp_path
    ):
        e = _eval(token_parent, tmp_path / "o", ckpts="last")
        mods = _with_target(e)
        windows = e._measures.windows["val"][:3]
        z, slot = e._encode(_Code("vit", None, mods), windows)
        assert slot is not None
        n = mods.chunk[0]
        k = windows.shape[1] - n + 1
        stacks = torch.stack([windows[:, i : i + n] for i in range(k)], 1)
        with torch.no_grad():
            target = mods.target_tokens(stacks.flatten(0, 1))
            online = mods.tokens(stacks.flatten(0, 1))
        assert not torch.allclose(target, online)
        torch.testing.assert_close(z.flatten(0, 1), target.flatten(1))
        torch.testing.assert_close(slot, last_slot(mods, target))

    def test_the_ceiling_decodes_the_target_encoders_tokens(
        self, token_parent, tmp_path
    ):
        e = _eval(token_parent, tmp_path / "o", ckpts="last")
        mods = _with_target(e)
        v = e.conf.vision
        assert v is not None
        x = e._measures.windows["val"][:, : v.frames]
        f0 = as_float(e._measures.windows["val"][:, v.frames - 1])
        torch.manual_seed(0)
        dec = token_decoder(8, v.patch, 3)
        rows = e._restore(mods, "val", dec, f0, {})
        with torch.no_grad():
            pictures = [
                picture_of(dec(last_slot(mods, t).flatten(0, 1)), len(x), mods.grid)
                for t in (mods.target_tokens(x), mods.tokens(x))
            ]
        want, online = (p.clamp(0, 1)[: len(rows[0])] for p in pictures)
        assert not torch.allclose(want, online)
        torch.testing.assert_close(rows[0], want)


def test_the_decoder_sees_z_centred_at_an_rms_std_of_one():
    """Components of std 10 and 0.1 about means 5 and -1, and a constant one,
    reach the decoder at mean 0, divided by one scale: the RMS of the stds,
    sqrt((100 + 0.01 + 0) / 3), so their stds keep the ratio 100 and the mean
    of their variances is 1; the constant one stays 0."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4096, 3, generator=g) * torch.tensor([10.0, 0.1, 0.0])
    z = z + torch.tensor([5.0, -1.0, 2.0])
    std = _Standardized(z).standardize(z)
    sd = z.std(0)
    scale = float((sd**2).mean().sqrt())
    torch.testing.assert_close(std.mean(0), torch.zeros(3), atol=1e-5, rtol=0)
    torch.testing.assert_close(std.std(0), sd / scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(float(std.var(0).mean()), 1.0, atol=1e-5, rtol=0)
    assert (std[:, 2] == 0).all()


def test_the_stream_windows_start_evenly_and_stay_within_a_video(tmp_path):
    """The loop A (0..29), B (30..49); window j starts at j 50 / 10 = 5 j, its
    frames are 2 apart, and a window of 4 + 3 frames (a span of 13) that
    would leave its video is left out: starts 0, 5, 10, 15 of A, 30, 35 of B."""
    root = numbered(tmp_path / "d", {"A": 30, "B": 20, "H": 10})
    conf = packed_conf(data_version=root.name, stride=2)
    w = stream_windows(conf, root, windows=10, steps=3)
    assert w.shape == (6, 7, 3, 2, 2)
    px = w[..., 0, 0].long()  # (6, 7, 3): video, j % 256, j // 256
    video, j = px[..., 0], px[..., 1] + 256 * px[..., 2]
    assert video[:, 0].tolist() == [0, 0, 0, 0, 1, 1]
    assert (video == video[:, :1]).all()
    assert j[:, 0].tolist() == [0, 5, 10, 15, 0, 5]
    assert (j[:, 1:] - j[:, :-1] == 2).all()


PCA = {"recon": 0.2, "dyn_1/copy_1": 0.5}


def _line(**kw) -> dict:
    return {"recon": 0.3, "dyn_1/copy_1": 0.5, "z_spread": 1.0} | kw


class TestObjective:
    """J = log max(recon / recon_PCA, r_1 / r_1_PCA), r_1 = dyn_1/copy_1, by
    hand on lines of round numbers."""

    def test_the_log_of_the_worse_ratio(self):
        assert _objective(_line(), PCA) == pytest.approx(math.log(1.5))
        line = _line(**{"recon": 0.1, "dyn_1/copy_1": 1.0})
        assert _objective(line, PCA) == pytest.approx(math.log(2.0))

    def test_better_than_the_pca_on_both_is_negative(self):
        line = _line(**{"recon": 0.1, "dyn_1/copy_1": 0.125})
        assert _objective(line, PCA) == pytest.approx(math.log(0.5))

    def test_ratios_up_to_ten_stay_apart_and_beyond_take_the_penalty(self):
        """Ratios 3.7 and 9 (the ViT before it learns is about 3.7) rank
        apart, below log 10; 20 is log 10 as a collapse."""
        values = [_objective(_line(recon=0.2 * r), PCA) for r in (3.7, 9.0, 20.0)]
        assert values[0] == pytest.approx(math.log(3.7))
        assert values[1] == pytest.approx(math.log(9.0))
        assert values[2] == PENALTY

    def test_a_collapse_is_the_penalty_however_good_the_ratios(self):
        line = _line(recon=0.01, z_spread=1e-4)
        assert _objective(line, PCA) == PENALTY
        assert _objective({"recon": 0.01, "dyn_1/copy_1": 0.1}, PCA) == PENALTY

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
    def test_a_value_that_is_not_finite_is_the_penalty(self, value):
        assert _objective(_line(recon=value), PCA) == PENALTY
        assert _objective(_line(), PCA | {"dyn_1/copy_1": value}) == PENALTY

    def test_the_pca_yardstick_is_its_better_decoder(self):
        """recon_lin 0.1 below recon 0.2: recon 0.3 is 3 times the PCA; a
        recon_lin above recon (0.4) leaves the yardstick at 0.2."""
        lin = PCA | {"recon_lin": 0.1}
        assert _objective(_line(), lin) == pytest.approx(math.log(3.0))
        worse = PCA | {"recon_lin": 0.4}
        assert _objective(_line(), worse) == pytest.approx(math.log(1.5))

    def test_a_perfect_code_counts_as_the_floor(self):
        line = _line(**{"recon": 0.0, "dyn_1/copy_1": 0.0})
        assert _objective(line, PCA) == pytest.approx(math.log(FLOOR))


class TestCompletionObjective:
    """The provisional "completion" objective: log of the mean comp_<kind>."""

    KINDS = ("short", "long", "future")

    def _line(self, **kw) -> dict:
        comp = {"comp_short": 0.5, "comp_long": 0.7, "comp_future": 0.9}
        return comp | {"z_spread": 1.0} | kw

    def test_the_log_of_the_mean(self):
        got = _completion_objective(self._line(), self.KINDS)
        assert got == pytest.approx(math.log(0.7))

    def test_a_collapse_is_the_penalty(self):
        line = self._line(z_spread=1e-4)
        assert _completion_objective(line, self.KINDS) == PENALTY

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
    def test_a_value_that_is_not_finite_is_the_penalty(self, value):
        line = self._line(comp_long=value)
        assert _completion_objective(line, self.KINDS) == PENALTY

    def test_at_most_the_penalty_and_a_perfect_one_is_the_floor(self):
        worse = self._line(comp_short=50.0, comp_long=50.0, comp_future=50.0)
        assert _completion_objective(worse, self.KINDS) == PENALTY
        perfect = self._line(comp_short=0.0, comp_long=0.0, comp_future=0.0)
        assert _completion_objective(perfect, self.KINDS) == pytest.approx(
            math.log(FLOOR)
        )

    def test_the_run_writes_it_when_chosen(self, parent, tmp_path):
        run_dir, root = parent
        p = Parent.read(run_dir, device="cpu")
        p.conf.measure.encoder_objective = "completion"
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv(DATA_ENV, str(root.parent))
            e = EncoderEval(
                p, p.conf, tmp_path / "o", root=root, windows=16, steps=3, ckpts="last"
            )
        lines = [ln for _ in range(len(e)) for ln in e.step()]
        val = next(ln for ln in lines if ln["code"] == "vit" and ln["split"] == "val")
        assert val["objective"] == pytest.approx(_completion_objective(val, self.KINDS))
