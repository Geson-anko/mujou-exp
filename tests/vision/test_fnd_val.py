"""``exp.vision.fnd_val`` on videos whose frames carry their video and their
number (``test_iid.numbered``): the validation video is dyn/copy's, never
Zurich's role (the held-out); its stacks are of the arm's spacings and fit
in it; the masks are the arm's of ``VAL_SEED``; ``val_rel`` and the collapse
numbers by hand from the JEPA's own outputs."""

import pytest
import torch

from exp.lib.metrics import effective_rank
from exp.lib.tours.walking_tours import videos
from exp.vision.dyn_eval import DynConf, split_videos
from exp.vision.fnd_jepa import (
    ARMS,
    build_fnd,
    complete,
    context_index,
    draw_masks,
    draw_spacings,
    encode,
    targets,
    time_r2,
    uniform_int,
)
from exp.vision.fnd_val import VAL_SEED, val_rows, val_set
from exp.vision.jepa_eval import token_r2
from exp.vision.models import early_fusion
from tests.vision.test_fnd_jepa import GRID, TIMES, conf_of
from tests.vision.test_iid import CODES, LENGTHS, frames_of, numbered  # noqa: F401

COUNT, HELD = 24, 40


@pytest.fixture
def vs(numbered):  # noqa: F811
    return val_set(
        conf_of(), numbered, ARMS["Ss"], s_max=4, times=TIMES, count=COUNT, held=HELD
    )


class TestValSet:
    def test_the_video_is_dyn_copys_validation_video(self, numbered, vs):  # noqa: F811
        names = [v for v in videos(numbered) if v != "H"]
        assert vs.video == split_videos(names, DynConf().split_seed)["val"][0]
        video, _ = frames_of(vs.stacks)
        assert (video == CODES[vs.video]).all()
        video, _ = frames_of(vs.held)
        assert (video == CODES[vs.video]).all()

    def test_the_stacks_are_the_arms_spacings_and_masks(self, vs):
        gen = torch.Generator().manual_seed(VAL_SEED)
        s = draw_spacings(COUNT, ARMS["Ss"], 4, gen)
        assert torch.equal(vs.spacing, s)
        uniform_int(torch.full((COUNT,), 1), gen)  # the last frames' draws
        masks, kinds = draw_masks(COUNT, TIMES, GRID, True, gen)
        assert torch.equal(vs.masks, masks) and torch.equal(vs.kinds, kinds)

    def test_the_frames_are_s_apart_within_the_video(self, vs):
        _, frame = frames_of(vs.stacks)  # (V, 8)
        steps = torch.arange(8)[None] * vs.spacing[:, None]
        assert torch.equal(frame, frame[:, :1] + steps)
        assert int(frame.min()) >= 0 and int(frame.max()) < LENGTHS[vs.video]
        _, held = frames_of(vs.held)
        assert torch.equal(held, held[:, :1] + torch.arange(8)[None])
        assert int(held[0, 0]) == 0 and int(held[-1, -1]) == LENGTHS[vs.video] - 1


def test_the_numbers_by_hand(vs):
    mods = build_fnd(conf_of(), seed=0)
    row = val_rows(mods, vs, heads=2, times=TIMES, scaled=True)
    with torch.no_grad():
        img = early_fusion(vs.stacks)
        t = mods.proj(encode(mods, img, 2)).double()
        keep, valid, slot = context_index(vs.masks)
        ctx = encode(mods, img, 2, keep, valid)
        ell = vs.spacing.double().log2()
        t_hat = complete(mods, ctx, slot, vs.masks, TIMES, ell).double()
    h = vs.masks
    sq = ((t_hat - t).square().sum(-1) * h).sum()
    dev = ((t - t.mean(0)).square().sum(-1) * h).sum()
    assert row["val_rel"] == pytest.approx(float(sq / dev), rel=1e-5)
    per = ((t_hat - t).square().mean(-1) * h).sum(1) / h.sum(1)
    assert row["val_pred"] == pytest.approx(float(per.mean()), rel=1e-5)
    for i, name in enumerate(("short", "long")):
        on = vs.kinds == i
        part = ((t_hat - t).square().sum(-1) * h)[on].sum()
        base = ((t - t.mean(0)).square().sum(-1) * h)[on].sum()
        assert row[f"val_rel_{name}"] == pytest.approx(float(part / base), rel=1e-5)
    th, zh = targets(mods, vs.held, 2)
    assert row["rank"] == pytest.approx(effective_rank(zh), rel=1e-6)
    assert row["time_r2"] == pytest.approx(float(time_r2(th, TIMES)), rel=1e-6)
    assert row["token_r2"] == pytest.approx(
        float(token_r2(th[0::2], th[1::2])), rel=1e-6
    )
    assert row["sigreg_gauss"] > 0 and row["sigreg_heldin"] > 0
