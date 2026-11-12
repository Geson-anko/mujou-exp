import json
import math
from pathlib import Path

import pytest
import torch

import exp.video.conf
from exp.video.compose import compose, from_dict, to_dict
from exp.video.jepa.loss import jepa_loss_packed
from exp.video.jepa.masks import expected_keep
from exp.video.jepa.nodes import draw_packed_directions
from exp.video.schema import RunConf

CONF = Path(exp.video.conf.__file__).parent
EXPERIMENTS = sorted(
    p.relative_to(CONF / "experiment").with_suffix("").as_posix()
    for p in CONF.glob("experiment/**/*.yaml")
)
GROUPS = ["dynamics", "vision", "learner", "optim", "phase"]
OPTIONS = [(g, p.stem) for g in GROUPS for p in sorted((CONF / g).glob("*.yaml"))]
# The options of the windows, applied under dynamics.window (not a default).
WINDOWS = [p.stem for p in sorted((CONF / "window").glob("*.yaml"))]
# The settings a group option needs to make a valid run: without dynamics,
# the program is the vision one, which needs an encoder and its optimizer.
NEEDS = {
    ("dynamics", "none"): [
        "program=vision",
        "vision=vit_t",
        "+optim@optim_vision=sgd",
    ],
}


def test_every_yaml_is_an_experiment_or_a_group_option() -> None:
    listed = {f"experiment/{e}" for e in EXPERIMENTS} | {f"{g}/{o}" for g, o in OPTIONS}
    listed |= {f"window/{w}" for w in WINDOWS}
    found = {
        p.relative_to(CONF).with_suffix("").as_posix() for p in CONF.glob("**/*.yaml")
    }
    assert found - {"run"} == listed


@pytest.mark.parametrize("experiment", EXPERIMENTS)
class TestExperiment:
    def test_composes_to_run_conf(self, experiment: str) -> None:
        conf = compose([f"+experiment={experiment}"])
        assert isinstance(conf, RunConf)
        assert conf.name == experiment

    def test_round_trips_through_json(self, experiment: str) -> None:
        conf = compose([f"+experiment={experiment}"])
        back = from_dict(json.loads(json.dumps(to_dict(conf))))
        assert isinstance(back, RunConf)
        assert back == conf


# One field per group that tells which option was applied.
WIDTHS = {"vit_t": 128, "vit_s": 256, "vit_m17": 256, "small": 256, "big": 512}


def _applied(conf: RunConf, group: str) -> object:
    match group:
        case "learner":
            return conf.learner.kind
        case "optim":
            return conf.optim.kind
        case "phase":
            return conf.phase
        case "vision":
            return "none" if conf.vision is None else conf.vision.dim
        case "dynamics":
            return "none" if conf.dynamics is None else conf.dynamics.d_s
        case _:
            raise AssertionError(f"no field for the group {group}")


@pytest.mark.parametrize(("group", "option"), OPTIONS)
def test_group_option_composes(group: str, option: str) -> None:
    extra = NEEDS.get((group, option), [])
    conf = compose(["+experiment=m14/s0", f"{group}={option}", *extra])
    assert _applied(conf, group) == WIDTHS.get(option, option)
    assert from_dict(to_dict(conf)) == conf


@pytest.mark.parametrize("option", WINDOWS)
def test_window_option_composes_under_the_dynamics(option: str) -> None:
    conf = compose(
        [
            "+experiment=m14/s0",
            f"+window@dynamics.window={option}",
            "dynamics.trace=false",
        ]
    )
    assert conf.dynamics is not None and conf.dynamics.window is not None
    assert from_dict(to_dict(conf)) == conf


class TestM14S0:
    """Stage 0 of M14: one ball, positions and velocities, the physics of M13."""

    def test_data(self) -> None:
        conf = compose(["+experiment=m14/s0"])
        assert conf.program == "dynamics"
        assert conf.data.streams == 128
        assert conf.data.tours is None
        balls = conf.data.balls
        assert balls is not None
        assert balls.radii == [0.10]
        assert balls.obs == "pos_vel"
        p = balls.physics
        assert (p.gamma0, p.gamma_amp, p.e0, p.e_amp, p.drag) == (
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )
        assert p.conserve
        assert p.substeps == 8

    def test_small_and_big_dynamics(self) -> None:
        small = compose(["+experiment=m14/s0"]).dynamics
        big = compose(["+experiment=m14/s0", "dynamics=big"]).dynamics
        assert small is not None and big is not None
        assert (small.d_s, small.nodes) == (256, 1)
        assert (big.d_s, big.nodes) == (512, 3)

    def test_time_and_optimizer(self) -> None:
        conf = compose(["+experiment=m14/s0"])
        assert conf.time.frame == pytest.approx(1 / 30)
        assert conf.time.learn_interval == 25e-3
        assert (conf.optim.kind, conf.optim.lr, conf.optim.warmup) == (
            "sgd",
            0.03,
            300.0,
        )
        assert conf.learner.kind == "bp"
        assert conf.phase == "explore"


M17 = [e for e in EXPERIMENTS if e.startswith("m17/")]


class TestM17:
    """M17's presets: the encoder on 128 x 128 Walking Tours at 30 fps, never a
    reconstruction term (the user's rule)."""

    @pytest.mark.parametrize("experiment", M17)
    def test_no_reconstruction_term_and_no_ema(self, experiment: str) -> None:
        conf = compose([f"+experiment={experiment}"])
        assert conf.vision is not None and conf.vision.recon == 0.0
        assert conf.vision.layout == "packed" and conf.vision.ema == 0.0

    def test_the_presets_are_n16_n8_and_n1_and_the_space_time_ones(self) -> None:
        assert M17 == [
            "m17/st_n16",
            "m17/st_tok_n16",
            "m17/tok_n16",
            "m17/wt_n1",
            "m17/wt_n16",
            "m17/wt_n8",
        ]

    @pytest.mark.parametrize("experiment", ["m17/st_n16", "m17/st_tok_n16"])
    def test_the_space_time_presets(self, experiment: str) -> None:
        """The stacks of wt_n16, three context views (the past one hides the
        last 2 of the 8 slots), the completion of u with SIGReg per view, and
        AdamW by steps."""
        conf = compose([f"+experiment={experiment}"])
        v = conf.vision
        assert v is not None and conf.optim_vision is not None
        assert (v.frames, v.tubelet, v.stride, v.views) == (16, 2, 3, 4)
        assert [m.future for m in v.context_masks] == [0, 0, 2]
        assert v.z_weight == 1.0 and v.u_width == 64
        assert v.sigreg_on in ("views", "views_tokens")
        assert conf.optim_vision.kind == "adamw_step"

    @pytest.mark.parametrize(
        ("experiment", "frames", "tubelet"),
        [("m17/wt_n16", 16, 2), ("m17/wt_n8", 8, 2), ("m17/wt_n1", 1, 1)],
    )
    def test_the_stacks_and_the_stream(
        self, experiment: str, frames: int, tubelet: int
    ) -> None:
        conf = compose([f"+experiment={experiment}"])
        v, t = conf.vision, conf.data.tours
        assert v is not None and t is not None
        assert (v.frames, v.tubelet, v.stride, v.patch) == (frames, tubelet, 3, 16)
        assert (t.size, t.fps, t.advance, t.held_out) == (128, 30.0, 3, "Zurich")
        assert [(m.blocks, m.scale) for m in v.context_masks] == [(8, 0.15), (2, 0.7)]
        assert v.views == 3 and v.z_width == 256 and not v.projected
        assert conf.measure.horizons == [1, 5, 10] and conf.measure.steps == 10
        assert conf.ctl.ckpt_every == 300.0  # explore keeps the last checkpoint

    def test_the_token_preset_is_wt_n16_with_the_tokens_as_the_code(self) -> None:
        """The tokens of 16 as the representation, SIGReg of them raw with
        8192 past tokens, the posthoc on the flattened tokens; everything
        else, the optimizer included, that of wt_n16 (2026-11-12)."""
        conf = compose(["+experiment=m17/tok_n16"])
        base = compose(["+experiment=m17/wt_n16"])
        v, w = conf.vision, base.vision
        assert v is not None and w is not None
        assert (v.token_dim, v.sigreg_on, v.token_sigreg) == (16, "tokens", "raw")
        assert (v.token_buffer, v.token_insert) == (8192, 64)
        assert v.z_dim is None and v.z_width == 16
        assert conf.measure.posthoc_code == "tokens"
        assert base.measure.posthoc_code == "z" and w.token_dim is None
        assert conf.optim_vision == base.optim_vision
        assert conf.data == base.data and conf.time == base.time
        changed = {"token_dim", "sigreg_on", "token_sigreg", "token_buffer"}
        changed |= {"token_insert", "sigreg"}
        for name in vars(w):
            if name not in changed:
                assert getattr(v, name) == getattr(w, name), name
        wide = compose(["+experiment=m17/tok_n16", "vision.token_dim=32"]).vision
        assert wide is not None and wide.z_width == 32
        with pytest.raises(ValueError, match="posthoc_code"):
            compose(["+experiment=m17/tok_n16", "measure.posthoc_code=pooled"])

    def test_an_entry_of_the_context_masks_takes_an_override(self) -> None:
        conf = compose(["+experiment=m17/wt_n16", "vision.context_masks.1.scale=0.5"])
        assert conf.vision is not None
        assert [m.scale for m in conf.vision.context_masks] == [0.15, 0.5]

    def test_horizons_beyond_the_steps_are_refused(self) -> None:
        with pytest.raises(ValueError, match="beyond measure.steps"):
            compose(["+experiment=m17/wt_n16", "measure.steps=5"])

    def test_the_objective_is_recon_dyn_unless_completion_is_chosen(self) -> None:
        conf = compose(["+experiment=m17/wt_n16"])
        assert conf.measure.encoder_objective == "recon_dyn"
        chosen = ["+experiment=m17/wt_n16", "measure.encoder_objective=completion"]
        assert compose(chosen).measure.encoder_objective == "completion"
        with pytest.raises(ValueError, match="encoder_objective"):
            compose(["+experiment=m17/wt_n16", "measure.encoder_objective=loss"])


def _m17_loss(conf: RunConf, target: torch.Tensor, predictor) -> float:
    """The loss of the packed layout of ``conf``'s vision for the target tokens
    ``target`` ``(B, T, D)`` (each context view's visible tokens taken from
    them, the masks of the preset's sizes), along directions of the preset's
    widths."""
    v = conf.vision
    assert v is not None
    b, t, d = target.shape
    slots = v.frames // v.tubelet
    side = math.isqrt(t // slots)
    keeps = [expected_keep((side, side), m) * slots for m in v.context_masks]
    g = torch.Generator().manual_seed(0)
    ys, parts = [], [target]
    for k in keeps:
        order = torch.rand(b, t, generator=g).argsort(1)
        y = torch.ones(b, t)
        y.scatter_(1, order[:, :k], 0.0)
        ys.append(y)
        visible = order[:, :k].sort(1).values
        parts.append(target.gather(1, visible[..., None].expand(-1, -1, d)))
    sigreg_on = {"views_tokens": "tokens"}.get(v.sigreg_on, v.sigreg_on)
    dz, dt = draw_packed_directions(
        32, v.z_width, v.token_width, tokens=sigreg_on != "z", generator=g
    )
    terms = jepa_loss_packed(
        torch.cat(parts, 1),
        torch.cat(ys, 1),
        torch.ones(b, dtype=torch.bool),
        dz,
        dt,
        predictor,
        keeps=keeps,
        proj=None,
        lam=v.sigreg,
        sg=v.sg,
        sigreg_on=sigreg_on,
        pred_loss=v.pred_loss,
        token_sigreg=v.token_sigreg,
        token_weight=v.token_weight,
        target_norm=v.target_norm,
    )
    return float(terms.loss)


@pytest.mark.parametrize("experiment", M17)
def test_m17_tokens_constant_within_a_stack_are_not_an_optimum(experiment):
    """The pilot's collapse (M17 audit, 2026-11-12): with SIGReg of z alone,
    tokens constant within each stack (z standard normal over the stacks),
    which the mean of the context predicts exactly, had a lower loss than
    i.i.d. standard normal tokens with their best constant prediction (0, an
    error of 1 per dimension). The loss of a preset must prefer the latter
    (its token terms: the completion of u, of the space-time presets, does
    not see the tokens within a stack)."""
    conf = compose([f"+experiment={experiment}"])
    v = conf.vision
    assert v is not None
    if v.token_weight == 0:
        pytest.skip("no token is predicted")

    def norm(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, x.shape[-1:]) if v.target_norm else x

    b, t = conf.data.streams, (v.frames // v.tubelet) * (128 // v.patch) ** 2
    d = v.token_width
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        clip = torch.randn(b, 1, d, generator=g).expand(b, t, d)
        collapsed = _m17_loss(
            conf,
            clip.contiguous(),
            lambda ctx, ctx_ids, tgt_ids: norm(ctx.mean(1, keepdim=True)).expand(
                -1, tgt_ids.shape[1], -1
            ),
        )
        spread = _m17_loss(
            conf,
            torch.randn(b, t, d, generator=g),
            lambda ctx, ctx_ids, tgt_ids: ctx.new_zeros(*tgt_ids.shape, ctx.shape[2]),
        )
    assert collapsed > spread
