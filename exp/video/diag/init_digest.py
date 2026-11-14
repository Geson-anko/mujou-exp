"""DIAGNOSIS ONLY (M17): the SHA-256 of the initial weights of the vision
modules, to check that two checkouts build the same untrained network (the
yardstick of one batch reused by another). The digest covers the name, dtype,
shape and bytes of every parameter and buffer of ``build_vision(conf,
seed=seed)``, on the CPU.

Run from the checkout to compare (``python -m exp.video.diag.init_digest
--seed 100 +experiment=m17/ssig_t128 learner=bp ...``), or with another
checkout first on ``sys.path``.
"""

import argparse
import hashlib
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from exp.video.compose import compose
from exp.video.jepa.vit import build_vision


def digest(modules: Mapping[str, nn.Module]) -> str:
    """The SHA-256 of the parameters and buffers of ``modules``, in order."""
    h = hashlib.sha256()
    for key, module in modules.items():
        tensors = [*module.named_parameters(), *module.named_buffers()]
        for name, t in tensors:
            c = t.detach().cpu().contiguous()
            h.update(f"{key}.{name} {c.dtype} {tuple(c.shape)}".encode())
            h.update(bytes(c.flatten().view(torch.uint8).tolist()))
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="the digest of the initial weights")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args(argv)
    conf = compose([*args.overrides, "device=cpu"])
    print(digest(build_vision(conf, seed=args.seed).named()))


if __name__ == "__main__":
    main()
