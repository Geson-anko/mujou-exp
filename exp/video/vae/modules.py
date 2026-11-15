r"""The modules of the autoencoder of M18: the encoder $f$ of M17 (the ViT of
``vit.build_vision``, its tokens the code) and the decoder $g$, the encoder
turned around (``TokenDecoder``); and of the VAE (stage 2, ``VAEModules``):
the same with the encoder's last map widened to the posterior's mean and log
variance, and the likelihood's $\ln \sigma$ (``LogSigma``).

A stack $x$ (u8 ``(n, N, C, H, W)``) is coded by its $T$ tokens of $d$
dimensions (``VisionModules.tokens``: the encoder's last stage maps each token
to ``vision.token_dim``), $D = T d$ numbers; $g$ maps them back to the stack,
the target $y = x / 256$ (``loss``). $g$'s constants $\mu$, $\sigma$ are the
encoder's, of $x / 255$: 0.4 % off the target's units at the start, which its
last map learns away.
"""

import math
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from exp.video.jepa.vit import (
    INIT_STD,
    Block,
    Stage,
    VisionModules,
    build_vision,
    image_shape,
    sincos_3d,
)
from exp.video.schema.run import RunConf

__all__ = [
    "DECODER_SEED",
    "AEModules",
    "LogSigma",
    "TokenDecoder",
    "VAEModules",
    "build_ae",
    "build_modules",
    "build_vae",
    "unpatchify_tubelets",
]

# Added to the run's seed: the decoder's initial weights, drawn after (and
# apart from) the encoder's, so that the encoder starts as M17's of the seed.
DECODER_SEED = 11


def unpatchify_tubelets(
    tokens: Tensor, grid3: tuple[int, int, int], patch: int, channels: int
) -> Tensor:
    r"""The stacks ``(n, N, C, H, W)`` of the tubelets ``(n, T, \tau C p^2)``
    in the order of ``vit.patchify_tubelets`` (time slot, row, column), of
    the grid ``grid3`` = (slots, rows, columns): its inverse."""
    n = len(tokens)
    gt, gh, gw = grid3
    tau = tokens.shape[2] // (channels * patch * patch)
    x = tokens.reshape(n, gt, gh, gw, tau, channels, patch, patch)
    x = x.permute(0, 1, 4, 5, 2, 6, 3, 7)  # (n, gt, tau, C, gh, p, gw, p)
    return x.reshape(n, gt * tau, channels, gh * patch, gw * patch)


class TokenDecoder(nn.Module):
    r"""$g$: the tokens ``(n, T, d)`` of a stack to the stack ``(n, N, C, H,
    W)``, the encoder of M17 turned around. Each token goes to the width by a
    linear map (``embed``) and gets its fixed 3-D position (``sincos_3d``, as
    in the encoder); ``stage`` is blocks, a ``LayerNorm`` and a linear map of
    each token to its tubelet's $\tau C p^2$ pixels (``vit.Stage``), whose
    outputs are standardized pixels: $\hat y = \mu + \sigma\,
    \mathrm{out}$ with the data's constants ``mean`` and ``std`` (those of
    the encoder's ``PackedEmbed``).

    The weights follow the encoder's rule (``vit``'s initialization): the
    maps into and out of the width from outside it (``embed``, the last map)
    LeCun's, the blocks' std ``INIT_STD``, unit ``LayerNorm``\ s, zero
    biases."""

    def __init__(
        self,
        *,
        token_dim: int,
        dim: int,
        heads: int,
        depth: int,
        grid3: tuple[int, int, int],
        patch: int,
        tubelet: int,
        channels: int,
        mean: float,
        std: float,
    ) -> None:
        super().__init__()
        self._grid3, self._patch, self._channels = grid3, patch, channels
        self._mean, self._std = mean, std
        self.embed = nn.Linear(token_dim, dim)
        self.register_buffer("pos", sincos_3d(grid3, dim), persistent=False)
        self.pos: Tensor
        blocks = [Block(dim, heads) for _ in range(depth)]
        self.stage = Stage(blocks, dim, norm=True, out=tubelet * channels * patch**2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                outside = m is self.embed or m is self.stage.out
                s = 1 / math.sqrt(m.in_features) if outside else INIT_STD
                nn.init.trunc_normal_(m.weight, std=s, a=-2 * s, b=2 * s)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        out = self.stage(self.embed(tokens) + self.pos)  # (n, T, tau C p^2)
        stacks = unpatchify_tubelets(out, self._grid3, self._patch, self._channels)
        return self._mean + self._std * stacks


@dataclass
class AEModules:
    """The encoder (``vision``: the modules of ``vit.build_vision``, of which
    the autoencoder uses the encoder only) and the decoder of the
    autoencoder."""

    vision: VisionModules
    decoder: TokenDecoder

    def named(self) -> dict[str, nn.Module]:
        """The learned modules by the names of the checkpoints."""
        return {"encoder": self.vision.encoder, "decoder": self.decoder}

    def parameters(self) -> list[nn.Parameter]:
        """The parameters the autoencoder learns: the encoder's, then the
        decoder's."""
        return [*self.vision.encoder.parameters(), *self.decoder.parameters()]

    def code(self, stacks: Tensor) -> Tensor:
        """The tokens ``(n, T, d)`` of the stacks ``(n, N, C, H, W)`` (u8, or
        floats in ``[0, 1]``) at the weights as they are now."""
        return self.vision.tokens(stacks)

    def __call__(self, stacks: Tensor) -> Tensor:
        r"""$\hat y = g(f(x))$ ``(n, N, C, H, W)`` of the stacks ``(n, N, C,
        H, W)``."""
        return self.decoder(self.code(stacks))


def build_ae(conf: RunConf, *, seed: int) -> AEModules:
    """The modules of ``conf``: the vision of ``vit.build_vision`` (``seed``:
    the encoder of M17's run of the same vision and seed), then the decoder
    of ``conf.ae``, built on the CPU after seeding the CPU's global RNG with
    ``seed + DECODER_SEED`` (restored afterwards), moved to ``conf.device``.

    Raises:
        ValueError: ``conf`` has no ``ae`` or no vision of the packed layout
            with ``vision.token_dim``.
    """
    v, a = conf.vision, conf.ae
    if a is None or v is None or v.layout != "packed" or v.token_dim is None:
        raise ValueError(
            f"the run {conf.name!r} needs ae and a packed vision with token_dim"
        )
    vision = build_vision(conf, seed=seed)
    channels, _ = image_shape(conf)
    with torch.random.fork_rng(devices=[]):  # the global RNG is left as it was
        torch.default_generator.manual_seed(seed + DECODER_SEED)
        decoder = TokenDecoder(
            token_dim=v.token_dim,
            dim=v.dim,
            heads=v.heads,
            depth=a.decoder_depth,
            grid3=vision.grid3,
            patch=v.patch,
            tubelet=v.tubelet,
            channels=channels,
            mean=v.pixel_mean,
            std=v.pixel_std,
        )
    return AEModules(vision, decoder.to(conf.device))


class LogSigma(nn.Module):
    r"""The likelihood's one learned $\ln \sigma$ (``log_sigma``, 0-d), held
    by this module itself: its fan-in is 1 (``exp.lib.optim.fan_ins``), so
    that ``scale="fan_in"`` steps it by $\eta$ as a vector."""

    def __init__(self, sigma_sq: float) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(0.5 * math.log(sigma_sq)))


@dataclass
class VAEModules:
    r"""The VAE of stage 2: the encoder (``vision``, its last map to $2d$ a
    token: ``build_vae``), the decoder and the likelihood's $\ln \sigma$.

    A stack's $T$ tokens of $2d$ are the posterior $q(z \mid x) =
    N(\mu, \mathrm{diag}\, s^2)$ of its code of $D = T d$: the first $d$ of
    each token $\mu$, the last $d$ $\ln s^2$. The code of the evaluations is
    $\mu$ (``code``), and ``__call__`` is the decoder at it, $g(\mu)$, as
    ``AEModules`` (``eval.AEEval``, ``measures.stack_errors``)."""

    vision: VisionModules
    decoder: TokenDecoder
    likelihood: LogSigma

    def named(self) -> dict[str, nn.Module]:
        """The learned modules by the names of the checkpoints."""
        return {
            "encoder": self.vision.encoder,
            "decoder": self.decoder,
            "likelihood": self.likelihood,
        }

    def parameters(self) -> list[nn.Parameter]:
        r"""The parameters the VAE learns: the encoder's, the decoder's, then
        $\ln \sigma$."""
        return [
            *self.vision.encoder.parameters(),
            *self.decoder.parameters(),
            *self.likelihood.parameters(),
        ]

    @property
    def log_sigma(self) -> Tensor:
        r"""$\ln \sigma$ (0-d)."""
        return self.likelihood.log_sigma

    @property
    def head(self) -> nn.Linear:
        r"""The encoder's last map, to $\mu$ and $\ln s^2$ (``build_vae``)."""
        stage = self.vision.encoder.stages[-1]
        assert isinstance(stage, Stage) and stage.out is not None
        return stage.out

    @property
    def code_shape(self) -> tuple[int, int]:
        r"""$(T, d)$ of a stack's code (and of its noise $\varepsilon$)."""
        return self.vision.tokens_per_view, self.head.out_features // 2

    def posterior(self, stacks: Tensor) -> tuple[Tensor, Tensor]:
        r"""$\mu$ and $\ln s^2$ ``(n, T, d)`` of the stacks ``(n, N, C, H,
        W)`` (u8, or floats in ``[0, 1]``) at the weights as they are now."""
        tokens = self.vision.tokens(stacks)
        d = tokens.shape[-1] // 2
        return tokens[..., :d], tokens[..., d:]

    def code(self, stacks: Tensor) -> Tensor:
        r"""$\mu$ ``(n, T, d)`` of the stacks."""
        return self.posterior(stacks)[0]

    def __call__(self, stacks: Tensor) -> Tensor:
        r"""$g(\mu)$ ``(n, N, C, H, W)`` of the stacks."""
        return self.decoder(self.code(stacks))


def build_vae(conf: RunConf, *, seed: int) -> VAEModules:
    r"""The VAE of ``conf``: the modules of ``build_ae`` (the same initial
    weights), with the encoder's last map (``Stage.out``, LeCun, $d$ outputs)
    widened to $2d$: its rows of $\mu$ are those it had, those of $\ln s^2$
    are 0 (weights and bias), and $\ln \sigma$ starts at $\frac12 \ln$
    ``ae.sigma0_sq``.

    So the untrained VAE's $\mu$ is the untrained autoencoder's code of the
    same seed and depth (the yardstick of U, ``EncoderEval``'s ViT before it
    learns, is the same) and its decoder the same, and its posterior variance
    $s^2 = 1$ is the prior's for every stack: the one scale the generative
    model gives before $g$ learns (the plan's LeCun would draw a log variance
    of unit spread around it, a scale the model does not state). The rows of
    $\ln s^2$ learn from 0 (the gradient of a weight is the output's times
    the input's, not 0).

    Raises:
        ValueError: ``conf`` has no ``ae.sigma0_sq`` or ``build_ae``
            raises.
    """
    if conf.ae is None or conf.ae.sigma0_sq is None:
        raise ValueError(f"the run {conf.name!r} needs ae.sigma0_sq")
    ae = build_ae(conf, seed=seed)
    stage = ae.vision.encoder.stages[-1]
    assert isinstance(stage, Stage) and stage.out is not None  # build_ae checks
    old = stage.out
    d, width = old.out_features, old.in_features
    # built empty (no draw of any RNG), every value then set
    head = nn.Linear(width, 2 * d, device="meta").to_empty(device=old.weight.device)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[:d] = old.weight
        head.bias[:d] = old.bias
    stage.out = head
    sigma = LogSigma(conf.ae.sigma0_sq).to(conf.device)
    return VAEModules(ae.vision, ae.decoder, sigma)


def build_modules(conf: RunConf, *, seed: int) -> AEModules | VAEModules:
    """``build_vae`` for the program ``"vae"``, else ``build_ae``."""
    if conf.program == "vae":
        return build_vae(conf, seed=seed)
    return build_ae(conf, seed=seed)
