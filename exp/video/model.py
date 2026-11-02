"""The network of PLAN.md, sections 3-4 and 7: ``Input -> G1 -> Z``, and from
``Z`` the ``Dynamics -> Compare`` of the prediction and ``SIGReg``; with
``ModelConfig.recon``, also ``Reconstruct`` of the frame from ``Z`` (M13).

The encoder and the dynamics can be cut into more nodes (``ModelConfig.enc_nodes``
and ``dyn_nodes``): the encoder of EgoDex (``egodex_stages``) into up to three
``pc.Layer``, the dynamics into a ``Dynamics`` and ``DynamicsLayer`` nodes on
top of it (``DynamicsStack`` is the same computation as one module).
"""

import math
from collections.abc import Sequence
from typing import Any, override

import mujou.pc as pc
import torch
from mujou.pc.optim import Optimizer
from torch import nn
from torch.nn import functional as F

import mujou
from exp.lib.mingru import MinGRU, ResidualMinGRU
from exp.lib.optim import Correlation
from exp.lib.sigreg import SIGReg
from exp.video.balls import Video
from exp.video.config import Config, DataConfig, ModelConfig, make_optimizer
from exp.video.nodes import Compare, Dynamics, DynamicsLayer, Reconstruct
from mujou import TensorSpec

__all__ = [
    "DynamicsStack",
    "GaussianBlur",
    "Network",
    "Top",
    "calibration_images",
    "decoder",
    "dynamics_modules",
    "egodex_stages",
    "encoder_modules",
    "encoder_stages",
    "image_shape",
    "standardize_",
    "target_weight",
]


class GaussianBlur(nn.Module):
    """A fixed Gaussian blur of standard deviation ``sigma`` pixels of grey
    images ``(1, H, W)`` or a batch of them, of the same size (zero padding).
    It has no parameters."""

    kernel: torch.Tensor

    def __init__(self, sigma: float) -> None:
        super().__init__()
        r = math.ceil(3 * sigma)
        x = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-(x**2) / (2 * sigma**2))
        k = k / k.sum()
        self.register_buffer("kernel", (k[:, None] * k)[None, None])  # (1, 1, K, K)
        self.r = r

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=self.r)


def encoder_modules(
    size: int, d_z: int, *, z_norm: bool = False, blur: float = 0.0, width: int = 32
) -> tuple[nn.Module, nn.Module]:
    """``G1`` and ``Z`` of the CNN (section 7): three convolutions of stride 2
    (``width``, twice and four times ``width`` channels) and a linear map to
    ``d_z``, scaled onto the sphere of radius
    $\\sqrt{d_z}$ with ``z_norm``; ``G1`` blurs the frame first with
    ``blur`` > 0 (``GaussianBlur``). They take ``(1, size, size)`` or a batch
    of them."""
    g1 = nn.Sequential(
        *([GaussianBlur(blur)] if blur > 0 else []),
        nn.Conv2d(1, width, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(width, 2 * width, 3, stride=2, padding=1),
        nn.GELU(),
    )
    side = size // 8
    z = nn.Sequential(
        nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Flatten(-3),
        nn.Linear(4 * width * side * side, d_z),
    )
    if z_norm:
        z.append(nn.RMSNorm(d_z, elementwise_affine=False))
    return g1, z


def egodex_stages(
    size: int, d_z: int, *, z_norm: bool = False
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """The three stages of the CNN of EgoDex (``(3, size, size)``, size 64):
    four convolutions of stride 2 and one of stride 1, to ``size / 16`` by
    ``size / 16`` by 256, and a linear map to ``d_z`` (with ``z_norm`` as in
    ``encoder_modules``)."""
    e1 = nn.Sequential(
        nn.Conv2d(3, 32, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.GELU(),
    )
    e2 = nn.Sequential(
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.GELU(),
    )
    side = size // 16
    e3 = nn.Sequential(
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Flatten(-3),
        nn.Linear(256 * side * side, d_z),
    )
    if z_norm:
        e3.append(nn.RMSNorm(d_z, elementwise_affine=False))
    return e1, e2, e3


def encoder_stages(config: Config) -> list[nn.Module]:
    """The modules of the encoder's nodes, from the input to $z$: ``G1`` and
    ``Z`` on the synthetic video, the stages of EgoDex grouped into
    ``model.enc_nodes`` nodes (3: one each; 2: the first two together)."""
    m, d = config.model, config.data
    if d.kind != "egodex":
        if m.enc_nodes != 2:
            raise ValueError("the encoder of the synthetic video has two nodes")
        return list(
            encoder_modules(d.size, m.d_z, z_norm=m.z_norm, blur=m.blur, width=m.width)
        )
    if m.blur > 0 or m.width != 32:
        raise ValueError("blur and width are for the synthetic video")
    e1, e2, e3 = egodex_stages(d.size, m.d_z, z_norm=m.z_norm)
    if m.enc_nodes == 3:
        return [e1, e2, e3]
    if m.enc_nodes == 2:
        return [nn.Sequential(e1, e2), e3]
    if m.enc_nodes == 1:
        return [nn.Sequential(e1, e2, e3)]
    raise ValueError(f"enc_nodes must be 1, 2 or 3, not {m.enc_nodes}")


def image_shape(d: DataConfig) -> tuple[int, int, int]:
    """The shape of a frame: grey on the synthetic video, RGB on EgoDex."""
    return (3 if d.kind == "egodex" else 1, d.size, d.size)


def calibration_images(config: Config) -> torch.Tensor:
    """256 frames to standardize the encoder on, apart from the stream: of a
    synthetic video of its own seed, or spread over the stream of EgoDex."""
    d, seed = config.data, config.seed + 10_000
    if d.kind == "egodex":
        from exp.video.egodex import calibration_images as egodex_images

        return egodex_images(d.path, n=256, seed=seed)
    video = Video(size=d.size, seed=seed)
    return torch.stack([video.next().image for _ in range(256)])


@torch.no_grad()
def standardize_(encoder: nn.Module, images: torch.Tensor) -> None:
    """Scale the weights of each convolution and linear map of ``encoder`` in
    place, in order, so that its output has standard deviation 1 over
    ``images``, as LSUV does; then shift the bias of the last one so that each
    of its outputs has mean 0.

    Without it, $z$ starts near 0 (the frames are mostly black), where the
    gradient of SIGReg vanishes with $z$ and the prediction error collapses it.
    Scaling only the last map would amplify the credit sent back to ``G1``
    by the same factor.
    """
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
    seen: list[Any] = []  # the last map's output, before any normalization of z
    handle = last.register_forward_hook(lambda _m, _i, out: seen.append(out))
    encoder(images)
    handle.remove()
    last.bias.sub_(seen[0].mean(0))


def decoder(d_z: int, shape: tuple[int, ...]) -> nn.Module:
    """$z$ ``(n, d_z)`` to frames ``(n, *shape)``, ``shape = (C, H, W)`` with
    ``H = W`` a power of 2 from 8: a linear map to 4 x 4, then transposed
    convolutions that double the size."""
    c, h, _ = shape
    ups = int(math.log2(h // 4))
    width = 32 * 2**ups
    layers: list[nn.Module] = [
        nn.Linear(d_z, width * 16),
        nn.Unflatten(1, (width, 4, 4)),
    ]
    for _ in range(ups):
        layers += [nn.ReLU(), nn.ConvTranspose2d(width, width // 2, 4, 2, 1)]
        width //= 2
    layers += [nn.ReLU(), nn.Conv2d(width, c, 3, padding=1)]
    return nn.Sequential(*layers)


def target_weight(m: ModelConfig) -> float:
    """The weight of the credit the prediction error sends to its target $z_k$."""
    return m.target_weight if m.stop_target else 1.0


def dynamics_modules(m: ModelConfig) -> tuple[MinGRU, list[ResidualMinGRU]]:
    """The modules of the dynamics' nodes: the minGRU of ``Dynamics`` and
    ``dyn_nodes - 1`` layers on top of it."""
    return MinGRU(m.d_z, m.d_s), [ResidualMinGRU(m.d_s) for _ in range(m.dyn_nodes - 1)]


class DynamicsStack(nn.Module):
    r"""The nodes of the dynamics as one module: $s_k$ from $z_{k-1}$ and the
    state $(s_{1}, h_{2}, \dots, h_{L})_{k-1}$ (concatenated, ``L d_s``) of the
    ``Dynamics`` and the ``DynamicsLayer`` nodes, to the new state. With one
    node, it is the minGRU (the state is $s$)."""

    def __init__(self, first: MinGRU, rest: Sequence[ResidualMinGRU]) -> None:
        super().__init__()
        self.first = first
        self.rest = nn.ModuleList(rest)
        self.d_s = first.gate.out_features

    @property
    def d_state(self) -> int:
        return self.d_s * (1 + len(self.rest))

    @override
    def forward(self, z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        parts = state.split(self.d_s, -1)
        y = self.first(z, parts[0])
        out = [y]
        for layer, h in zip(self.rest, parts[1:], strict=True):
            y_next = layer(y, h)
            out.append(y_next - y)
            y = y_next
        return torch.cat(out, -1)


class Top(nn.Module):
    """The output $y_L = s_1 + h_2 + \\dots + h_L$ of the top node of the
    dynamics, from the state of ``DynamicsStack``."""

    def __init__(self, d_s: int) -> None:
        super().__init__()
        self.d_s = d_s

    @override
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.stack(state.split(self.d_s, -1)).sum(0)


class Network(mujou.Network):
    """The world model, made from ``config`` after ``torch.manual_seed(config.seed)``.

    The host writes a ``pc.msg.Sample`` (``x``: the image of ``image_shape``,
    ``y``: unused zeros ``(1,)``; with ``data.streams`` = B > 1, each with a
    leading B, and ``mask`` and ``reset`` of ``(B,)``) to ``prev`` and reads the
    prequential
    prediction of $z_k$ from ``next`` (``Compare.next``). The nodes of the
    encoder are ``encoder_names`` (the last is ``z``), those of the dynamics
    ``dynamics_names`` (the first is ``dynamics``); with ``model.recon`` > 0,
    ``reconstruct`` decodes the frame from $z$.
    """

    # Set by name: those of the layout of the config only.
    g1: pc.Layer
    g2: pc.Layer
    z: pc.Layer
    dynamics2: DynamicsLayer
    dynamics3: DynamicsLayer
    reconstruct: Reconstruct

    def __init__(self, config: Config, *, correlation: bool = False) -> None:
        super().__init__()
        m, t, d = config.model, config.time, config.data
        tau = pc.TimeConstants(h=t.h, lam=t.lam, r=t.r)
        self.tau = tau
        # With correlation, each node records the correlation of its gradients.
        self.correlations: dict[str, Correlation] = {}

        def optimizer_of(name: str) -> Optimizer:
            optimizer = make_optimizer(config.optim)
            if correlation:
                optimizer = self.correlations[name] = Correlation(optimizer)
            return optimizer

        li = t.learn_interval
        torch.manual_seed(config.seed)
        stages = encoder_stages(config)
        standardize_(nn.Sequential(*stages), calibration_images(config))
        step, layers = dynamics_modules(m)
        readout = nn.Linear(m.d_s, m.d_z)
        if m.residual:  # the prediction starts as z_{k-1}, the change as 0
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)
        dec = decoder(m.d_z, image_shape(d)) if m.recon > 0 else None
        # Made on the CPU, so that the initial weights do not depend on the device.
        dev = config.device
        stages = [s.to(dev) for s in stages]
        step, readout = step.to(dev), readout.to(dev)
        layers = [layer.to(dev) for layer in layers]
        self.input = pc.Input(
            x=TensorSpec((*d.batch, *image_shape(d))),
            y=TensorSpec((*d.batch, 1)),
            period=tau.period,
            batch=d.batch,
            device=dev,
        )
        n = len(stages)
        self.encoder_names = tuple([f"g{i + 1}" for i in range(n - 1)] + ["z"])
        prev: Any = self.input
        for name, stage in zip(self.encoder_names, stages, strict=True):
            node = pc.Layer(
                stage,
                act_spec=prev.act_spec,
                tau=tau,
                optimizer=optimizer_of(name),
                learn_interval=li,
                device=dev,
            )
            setattr(self, name, node)
            prev.next.connect(node.prev)
            prev = node
        self.dynamics = Dynamics(
            step,
            act_spec=self.z.act_spec,
            tau=tau,
            optimizer=optimizer_of("dynamics"),
            learn_interval=li,
            open_loop=m.open_loop,
            open_cycle=m.open_cycle,
            residual=m.residual,
            device=dev,
        )
        self.dynamics_names = ("dynamics",) + tuple(
            f"dynamics{i + 2}" for i in range(len(layers))
        )
        top: Any = self.dynamics
        for name, layer in zip(self.dynamics_names[1:], layers, strict=True):
            node = DynamicsLayer(
                layer,
                act_spec=top.act_spec,
                tau=tau,
                optimizer=optimizer_of(name),
                learn_interval=li,
                device=dev,
            )
            setattr(self, name, node)
            top.next.connect(node.prev)
            top = node
        self.sigreg = SIGReg(
            act_spec=self.z.act_spec,
            kappa=m.kappa,
            directions=m.directions,
            capacity=m.buffer,
            every=m.buffer_every,
            tau=tau,
            seed=config.seed,
            device=dev,
        )
        self.compare = Compare(
            readout,
            s_spec=top.act_spec,
            z_spec=self.z.act_spec,
            kappa=m.kappa,
            tau=tau,
            optimizer=optimizer_of("compare"),
            learn_interval=li,
            target_weight=target_weight(m),
            residual=m.residual,
            device=dev,
        )
        if dec is not None:
            self.reconstruct = Reconstruct(
                dec.to(dev),
                x_spec=self.input.act_spec,
                z_spec=self.z.act_spec,
                weight=m.recon,
                tau=tau,
                optimizer=optimizer_of("reconstruct"),
                learn_interval=li,
                device=dev,
            )
            self.input.next.connect(self.reconstruct.x)
            self.z.next.connect(self.reconstruct.z)
        self.z.next.connect(self.dynamics.z)
        self.z.next.connect(self.sigreg.prev)
        self.z.next.connect(self.compare.z)
        top.next.connect(self.compare.s)
        if m.open_loop > 0:
            self.compare.next.connect(self.dynamics.z_hat)
        if m.residual:
            self.dynamics.base.connect(self.compare.base)
        self.residual = m.residual
        self.prev, self.next = self.input.prev, self.compare.next
