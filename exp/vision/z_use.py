r"""Does the system's decoder use its code? (V12) ``span`` < 1 says the
system beats the mean stack, not that its decoder $g$ reads the code: a
decoder that tracks the present scene with its weights beats it too.

On the stacks $x_i$ of the run's span every ``every`` seconds (the span
positions of ``stream_eval``, the units the stream marks not ready or reset
left out), with $m$ their mean stack and $b_i = \lVert x_i/255 - m\rVert^2$,
a checkpoint's row holds three ratios $\sum_i e_i / \sum_i b_i$:

- ``span``: $e_i = \lVert x_i/255 - g(\mu(x_i))\rVert^2$, ``stream_eval``'s
  span on these positions
- ``span_shifted``: $e_i = \lVert x_i/255 - g(\mu(x_{\pi(i)}))\rVert^2$, the
  code of the stack half the span away (``shifted``; 900 s in an 1800 s run,
  another scene)
- ``span_mean_z``: $e_i = \lVert x_i/255 - g(\bar\mu)\rVert^2$, $\bar\mu$ the
  mean code of the positions

A decoder that ignores its code has the three equal; ``span_shifted -
span`` is what the code adds, ``span_mean_z`` what the decoder makes of no
code in particular.

``evaluate_z_use`` writes ``<run_dir>/posthoc_stream/z_use.jsonl`` (a row per
checkpoint of ``at``, default ``stream_eval.PROBE_AT`` and the last)::

    python -m exp.vision.z_use <run_dir> [--root <data>]
"""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from exp.lib import ckpt
from exp.lib.provenance import configure_determinism
from exp.lib.tours.stacks import tours_root
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, VaeCode, stacks_at
from exp.vision.schema import RunConf
from exp.vision.stream_eval import OUT, PAST_EVERY, PROBE_AT, stream_positions

__all__ = ["ZUse", "evaluate_z_use", "main", "shifted"]


def shifted(n: int) -> list[int]:
    """$\\pi(i) = (i + \\lfloor n/2 \\rfloor) \\bmod n$."""
    return [(i + n // 2) % n for i in range(n)]


class ZUse:
    """The rows (module docstring) of the run ``conf`` on the data at
    ``root``, its span every ``every`` seconds."""

    def __init__(self, conf: RunConf, root: Path, *, every: float = PAST_EVERY):
        self._dev = torch.device(conf.device)
        span = round(conf.measure.duration / conf.time.frame)
        ks = [k for k in stream_positions(conf, past_every=every) if k < span]
        self.ks, self._x = stacks_at(conf, root, ks)  # u8, CPU
        self._m = self._x.double().mean(0).div(255).float().to(self._dev)
        self._base = sum(self._errors(self._x, lambda _: self._m[None]))

    def _errors(self, x: Tensor, recon: Callable[[int], Tensor]) -> list[float]:
        """$\\lVert x_i/255 - r_i\\rVert^2$ of the stacks ``x`` u8, ``recon(i)``
        the reconstructions ``(B, ...)`` of the batch starting at ``i``."""
        out: list[float] = []
        for i in range(0, len(x), BATCH):
            xb = x[i : i + BATCH].to(self._dev).float() / 255
            r = recon(i).reshape(-1, *xb.shape[1:]).expand_as(xb)
            out += ((xb - r) ** 2).double().flatten(1).sum(1).tolist()
        return out

    @torch.no_grad()
    def row(self, code: VaeCode) -> dict[str, Any]:
        """``n`` and the three ratios of the VAE ``code``."""
        mu = torch.cat(
            [code.mu(c.to(self._dev)).float() for c in self._x.split(BATCH)]
        )  # (n, D)
        pi = torch.tensor(shifted(len(mu)), device=mu.device)
        moved, mean = mu[pi], mu.mean(0, keepdim=True)

        def ratio(z_of: Callable[[int], Tensor]) -> float:
            return sum(self._errors(self._x, lambda i: code.decode(z_of(i)))) / (
                self._base
            )

        return {
            "n": len(mu),
            "span": ratio(lambda i: mu[i : i + BATCH]),
            "span_shifted": ratio(lambda i: moved[i : i + BATCH]),
            "span_mean_z": ratio(lambda _: mean),
        }


def evaluate_z_use(
    run_dir: Path,
    build: Callable[[RunConf], VaeCode],
    *,
    root: Path | None = None,
    every: float = PAST_EVERY,
    at: Sequence[float] | None = None,
) -> Path:
    """The rows of the checkpoints ``ckpt/<t>.pt`` of the run in ``run_dir``
    whose ``t`` is in ``at`` (default ``PROBE_AT`` and the last), in the
    order of ``t``, under ``provenance.configure_determinism``, into
    ``posthoc_stream/z_use.jsonl``. Returns its path.

    Raises:
        FileNotFoundError: The run has no checkpoint.
    """
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    configure_determinism(conf.device)
    paths = sorted(run_dir.glob("ckpt/*.pt"), key=lambda p: float(p.stem))
    if not paths:
        raise FileNotFoundError(f"no checkpoint under {run_dir / 'ckpt'}")
    last = float(paths[-1].stem)
    wanted = [*PROBE_AT, last] if at is None else list(at)
    paths = [p for p in paths if any(abs(float(p.stem) - a) < 1e-6 for a in wanted)]
    root = tours_root(conf.data.tours.data_version) if root is None else root
    z = ZUse(conf, root, every=every)
    code = build(conf)
    rows = []
    for path in paths:
        head = ckpt.load_eval(path, code.modules())
        rows.append({"t": head["t"], "frame": int(head["frame"])} | z.row(code))
    out = run_dir / OUT / "z_use.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.z_use``: ``evaluate_z_use`` of a run of the
    program ``"vae"``. Returns the path of its rows."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.z_use",
        description="Does the system's decoder use its code? (span positions)",
    )
    ap.add_argument("run_dir", type=Path, help="the run's directory")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    args = ap.parse_args(argv)
    from exp.vision.models import build_vae  # the model, only for the shell

    return evaluate_z_use(
        args.run_dir, lambda conf: build_vae(conf, seed=conf.seed), root=args.root
    )


if __name__ == "__main__":
    print(main())
