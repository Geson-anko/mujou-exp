"""The network of PLAN.md, sections 3-4 and 7: ``Input -> G1 -> Z``, and from
``Z`` the ``Dynamics -> Compare`` of the prediction and ``SIGReg``."""

from typing import Any

import mujou.pc as pc
import torch
from mujou.pc.optim import Optimizer
from torch import nn

import mujou
from experiments.continuous_stream.config import Config
from experiments.continuous_stream.nodes import Compare, Dynamics, MinGRU, SIGReg
from experiments.continuous_stream.optim import Correlation, from_config
from experiments.continuous_stream.video import Video
from mujou import TensorSpec

__all__ = ["Network", "encoder_modules", "standardize_"]


def encoder_modules(size: int, d_z: int) -> tuple[nn.Module, nn.Module]:
    """``G1`` and ``Z`` of the CNN (section 7): three convolutions of stride 2
    and a linear map to ``d_z``. They take ``(1, size, size)`` or a batch of
    them."""
    g1 = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.GELU(),
    )
    side = size // 8
    z = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Flatten(-3),
        nn.Linear(128 * side * side, d_z),
    )
    return g1, z


@torch.no_grad()
def standardize_(g1: nn.Module, z: nn.Module, size: int, seed: int) -> None:
    """Scale the weights of each convolution and linear map of ``g1`` and ``z``
    in place, in order, so that its output has standard deviation 1 over 256
    frames of a video of its own (``seed``), as LSUV does; then shift the bias
    of the last one so that each dimension of $z$ has mean 0.

    Without it, $z$ starts near 0 (the frames are mostly black), where the
    gradient of SIGReg vanishes with $z$ and the prediction error collapses it.
    Scaling only the last map would amplify the credit sent back to ``G1``
    by the same factor.
    """
    video = Video(size=size, seed=seed)
    images = torch.stack([video.next().image for _ in range(256)])
    encoder = nn.Sequential(g1, z)
    layers = [m for m in encoder.modules() if isinstance(m, nn.Conv2d | nn.Linear)]
    for layer in layers:
        seen: list[Any] = []  # the output of the layer
        handle = layer.register_forward_hook(lambda _m, _i, out: seen.append(out))
        encoder(images)
        handle.remove()
        std = seen[0].std()
        layer.weight.div_(std)
        assert layer.bias is not None
        layer.bias.div_(std)
    last = layers[-1]
    assert last.bias is not None
    last.bias.sub_(encoder(images).mean(0))


class Network(mujou.Network):
    """The world model, made from ``config`` after ``torch.manual_seed(config.seed)``.

    The host writes a ``pc.msg.Sample`` (``x``: the image ``(1, size, size)``,
    ``y``: unused zeros ``(1,)``) to ``prev`` and reads the prequential
    prediction of $z_k$ from ``next`` (``Compare.next``).
    """

    def __init__(self, config: Config, *, correlation: bool = False) -> None:
        super().__init__()
        m, t, d = config.model, config.time, config.data
        tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        self.tau = tau
        # With correlation, each node records the correlation of its gradients.
        self.correlations: dict[str, Correlation] = {}

        def optimizer_of(name: str) -> Optimizer:
            optimizer = from_config(config.optim)
            if correlation:
                optimizer = self.correlations[name] = Correlation(optimizer)
            return optimizer

        li = t.learn_interval
        torch.manual_seed(config.seed)
        g1, z = encoder_modules(d.size, m.d_z)
        standardize_(g1, z, d.size, seed=config.seed + 10_000)
        step, readout = MinGRU(m.d_z, m.d_s), nn.Linear(m.d_s, m.d_z)
        # Made on the CPU, so that the initial weights do not depend on the device.
        dev = config.device
        g1, z, step, readout = g1.to(dev), z.to(dev), step.to(dev), readout.to(dev)
        self.input = pc.Input(
            x=TensorSpec((1, d.size, d.size)),
            y=TensorSpec((1,)),
            period=tau.period,
            device=dev,
        )
        self.g1 = pc.Layer(
            g1,
            act_spec=self.input.act_spec,
            tau=tau,
            optimizer=optimizer_of("g1"),
            learn_interval=li,
            device=dev,
        )
        self.z = pc.Layer(
            z,
            act_spec=self.g1.act_spec,
            tau=tau,
            optimizer=optimizer_of("z"),
            learn_interval=li,
            device=dev,
        )
        self.dynamics = Dynamics(
            step,
            act_spec=self.z.act_spec,
            tau=tau,
            optimizer=optimizer_of("dynamics"),
            learn_interval=li,
            device=dev,
        )
        self.sigreg = SIGReg(
            act_spec=self.z.act_spec,
            kappa=m.kappa,
            directions=m.directions,
            capacity=m.buffer,
            tau=tau,
            seed=config.seed,
            device=dev,
        )
        self.compare = Compare(
            readout,
            s_spec=self.dynamics.act_spec,
            z_spec=self.z.act_spec,
            kappa=m.kappa,
            tau=tau,
            optimizer=optimizer_of("compare"),
            learn_interval=li,
            device=dev,
        )
        self.input.next.connect(self.g1.prev)
        self.g1.next.connect(self.z.prev)
        self.z.next.connect(self.dynamics.z)
        self.z.next.connect(self.sigreg.prev)
        self.z.next.connect(self.compare.z)
        self.dynamics.next.connect(self.compare.s)
        self.prev, self.next = self.input.prev, self.compare.next
