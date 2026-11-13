r"""DIAGNOSIS ONLY (M17): an adversarial check of the completion error of the
masked tokens ($E^{\mathrm n}$, ``measure.completion.Unexplained.within``,
the ``ncomp_<kind>`` of ``exp/video/results/m17.md``): the trained predictor
against trivial predictors that know the position of a masked token.

For the target tokens $s(x, p)$ of the held-out Zurich val stacks and
$\mu(p)$, the mean target token of position $p$ (slot and spatial position)
over the held-in stacks (``PackedMeasures.held_in``):

- ``pos``: $\hat s(x, p) = \mu(p)$
- ``pos_off``: $\hat s(x, p) = \mu(p) + \frac{1}{|C|} \sum_{c \in C} (s(x, c) -
  \mu(c))$, $C$ the visible tokens: the position mean shifted by the stack's
  mean offset seen in the context
- ``copy``: for the ``future`` kind, the token of the same spatial position in
  the last visible slot; else ``pos_off``

``pos_R2_within_val`` is the share of the spread of the val tokens about their
stack's mean that the position means explain: $1 - \sum \lVert (s - \bar s_x)
- (\mu - \bar\mu) \rVert^2 / \sum \lVert s - \bar s_x \rVert^2$. A token of
the position alone (P, $s(x, p) = g(p)$) gives ``pos`` an error of 0, and a
token of the position plus a stack's constant gives ``pos_off`` an error of 0
(``trivial_predictions``). It is the precursor of ``comp_<kind>_pos`` (the
same check built into the evaluation, 「既存の 6 本の P の確認」 of
「ストリームごとのバッファ」); its own numbers are not quoted in the results.

Run on a GPU host, from the snapshot the runs were made with (it was run in
``code/dfaa437098ff``), read-only (it loads checkpoints)::

    python -m exp.video.diag.posbase <batch>:<run_id>:<ckpt> ... [--runs /nas/geson/mujou-exp/runs]

``<ckpt>`` is a checkpoint's name under ``ckpt/`` without ``.pt`` (e.g.
``00600.0``), or ``init`` for the untrained network. One JSON line per run.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from exp.lib import ckpt
from exp.video.compose import from_dict
from exp.video.jepa.vit import build_vision
from exp.video.measure.completion import FUTURE, Unexplained, complete, targets
from exp.video.programs.vision import PackedMeasures

RUNS = Path("/nas/geson/mujou-exp/runs")


def trivial_predictions(
    s: torch.Tensor,
    pos: torch.Tensor,
    masked: torch.Tensor,
    keep: int,
    *,
    future: bool,
    slots: int,
    per: int,
) -> dict[str, torch.Tensor]:
    """``pos``, ``pos_off`` and ``copy`` (module docstring) ``(n, M, D)`` of
    the masked tokens of the target tokens ``s`` ``(n, T, D)`` (ids in slot
    order: ``per`` spatial positions a slot), with the position means ``pos``
    ``(T, D)`` and the mask ``masked`` ``(n, T)`` (1: masked, ``keep``
    visible a row; ``future``: the masked tokens are the last slots whole)."""
    order = masked.sort(dim=1, stable=True).indices
    ctx_ids, tgt_ids = order[:, :keep], order[:, keep:]
    p_tgt = pos[tgt_ids]  # (n, M, D)
    p_ctx = pos[ctx_ids]
    vis = s.gather(1, ctx_ids[..., None].expand(-1, -1, s.shape[2]))
    off = (vis - p_ctx).mean(1, keepdim=True)
    preds = {"pos": p_tgt, "pos_off": p_tgt + off}
    if future:
        f = slots - (keep // per)  # the masked slots
        src = tgt_ids - per * (tgt_ids // per - (slots - f - 1))
        preds["copy"] = s.gather(1, src[..., None].expand(-1, -1, s.shape[2]))
    else:
        preds["copy"] = p_tgt + off
    return preds


def check(
    spec: str, runs: Path, cache: dict[tuple[int, int], PackedMeasures]
) -> dict[str, Any]:
    """The row of ``<batch>:<run_id>:<ckpt>``."""
    batch, rid, which = spec.split(":")
    job = json.loads((runs / batch / rid / "job.json").read_text())
    cfg = dict(job["config"])
    cfg["device"] = "cuda"
    conf = from_dict(cfg)
    v = conf.vision
    assert v is not None
    mods = build_vision(conf, seed=conf.seed)
    if which != "init":
        ckpt.load_eval(runs / batch / rid / "ckpt" / f"{which}.pt", mods.named())
    for m in mods.named().values():
        m.eval()
    key = (v.frames, v.stride)
    if key not in cache:
        cache[key] = PackedMeasures(conf)
    meas = cache[key]
    dev = torch.device("cuda")
    slots = meas._slots
    per = meas.grid[0] * meas.grid[1]
    norm = v.target_norm

    @torch.no_grad()
    def tok(x: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, len(x), 32):
            out.append(targets(mods, x[i : i + 32].to(dev), normalize=norm).cpu())
        return torch.cat(out)

    held = tok(meas.held_in)  # (n, T, D)
    pos = held.mean(0)  # (T, D) per-position mean, held-in
    xval = meas.windows["val"][:, : v.frames]
    sval = tok(xval)
    # fraction of the within-stack spread explained by the held-in position mean
    dev_w = sval - sval.mean(1, keepdim=True)
    dev_p = pos - pos.mean(0, keepdim=True)
    r2_pos = 1 - float(((dev_w - dev_p) ** 2).sum()) / float((dev_w**2).sum())
    row: dict[str, Any] = {
        "run": rid,
        "ckpt": which,
        "pos_R2_within_val": round(r2_pos, 4),
    }
    for kind, (masked, keep) in meas._kinds("val").items():
        acc = Unexplained(["pred", "pos", "pos_off", "copy"])
        for i in range(0, len(xval), 32):
            x = xval[i : i + 32].to(dev)
            s = sval[i : i + 32].to(dev)
            m = masked[i : i + 32].to(dev)
            pred, goal, _ = complete(mods, x, s, m, keep)
            preds = trivial_predictions(
                s, pos.to(dev), m, keep, future=kind == FUTURE, slots=slots, per=per
            )
            acc.add(goal, {"pred": pred, **preds})
        for k, val in acc.within().items():
            row[f"ncomp_{kind}_{k}"] = round(val, 4)
    return row


def main(argv: Sequence[str] | None = None) -> None:
    """One JSON line per ``<batch>:<run_id>:<ckpt>`` (module docstring)."""
    p = argparse.ArgumentParser(description="Trivial predictors vs ncomp (M17).")
    p.add_argument("specs", nargs="+")
    p.add_argument("--runs", type=Path, default=RUNS)
    args = p.parse_args(argv)
    cache: dict[tuple[int, int], PackedMeasures] = {}
    for spec in args.specs:
        print(json.dumps(check(spec, args.runs, cache)), flush=True)


if __name__ == "__main__":
    main()
