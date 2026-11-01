"""The baselines of backpropagation: the same modules learned by autograd inside
one node.

A ``pc.Output`` differentiates its module by autograd from the loss, so a chain
of one ``nn.Sequential`` of all the modules is BP in one node, with the same
``LearnClock`` and the same optimizers as the chain of PC-ALM.
"""

from collections.abc import Callable
from typing import Any

import mujou.pc as pc
import torch
from torch import nn

__all__ = ["bp_chain"]


def bp_chain(
    *modules: nn.Module,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    **kwargs: Any,
) -> pc.Chain:
    """``pc.Chain(nn.Sequential(*modules), ...)``: ``Input -> Output``, the
    output learning all of ``modules`` by BP. ``kwargs`` are those of
    ``pc.Chain``."""
    return pc.Chain(nn.Sequential(*modules), loss=loss, **kwargs)
