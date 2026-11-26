r"""A light decoder of images conditioned on a frozen code, learned by flow
matching (rectified flow), to look at what a code keeps: it draws what the
code does not hold as plausible detail instead of a blur (V18c's G,
``exp/vision/plans/jepa.md``; the decoder of ``exp/vision/PLAN.md``'s
「評価の拡張: 軽量の拡散デコーダ」).

Objective: with the noise $x_0 \sim N(0, I)$, the data $x_1$, $t \sim U(0,
1)$ and $x_t = (1 - t) x_0 + t x_1$,

$$
L = \mathbb{E}\, \frac{1}{\dim x} \lVert v_\theta(x_t, t, c) - (x_1 - x_0)
\rVert^2 ,
$$

whose minimizer is $v^*(x, t, c) = \mathbb{E}[x_1 - x_0 \mid x_t = x, c]$.
Samples follow the ODE $\dot x = v_\theta(x, t, c)$ from $x_0$ at $t = 0$ to
$t = 1$ by Euler's method (``euler_sample``).

The net (``FlowUNet``) is a small U-Net of ADM's shape: at each resolution
two ``ResBlock``\ s down and three up (each joined to a skip of the way
down), attention at the resolutions of ``attn_levels`` and in the middle
(``ResBlock``, attention, ``ResBlock``), down by a 3 x 3 convolution of
stride 2, up by the nearest 2 x enlargement and a 3 x 3 convolution. The
time ($1000 t$ in a sinusoidal embedding) and the code (through an MLP) are
added into one embedding $e$, from which each ``ResBlock`` makes its FiLM
$(\gamma, \beta)$. The code's MLP is built last, so the other weights of the
same seed do not depend on the code's dimension.
"""

import math
from collections.abc import Callable, Sequence
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "CHANNELS",
    "EMBED",
    "FlowUNet",
    "TIME_DIM",
    "euler_sample",
    "fit_flow",
    "flow_loss",
    "lr_at",
]

CHANNELS = (32, 64, 128, 256)  # of the resolutions 64, 32, 16, 8
ATTN_LEVELS = (2, 3)  # the levels of CHANNELS with attention: 16 and 8
GROUPS, HEAD_DIM = 32, 32
EMBED, TIME_DIM = 512, 128

type Velocity = Callable[[Tensor, Tensor, Tensor], Tensor]


def _norm(c: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


def time_embedding(t: Tensor, dim: int = TIME_DIM) -> Tensor:
    r"""The sinusoidal embedding ``(B, dim)`` of $1000 t$ (``t`` ``(B,)``):
    $\cos(1000 t f_k)$ then $\sin(1000 t f_k)$, $f_k = 10000^{-k / (dim /
    2)}$."""
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = 1000 * t.float()[:, None] * f
    return torch.cat([a.cos(), a.sin()], 1)


class ResBlock(nn.Module):
    r"""$h = \mathrm{conv}(\mathrm{SiLU}(\mathrm{GN}(x)))$, $h =
    \mathrm{GN}(h) (1 + \gamma) + \beta$ of $(\gamma, \beta)$ =
    ``film(SiLU(e))``, $h = \mathrm{conv}(\mathrm{SiLU}(h))$; the output $h$
    + skip($x$) (1 x 1 when the channels change)."""

    def __init__(self, c_in: int, c_out: int, *, groups: int, embed: int) -> None:
        super().__init__()
        self.norm1 = _norm(c_in, groups)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.film = nn.Linear(embed, 2 * c_out)
        self.norm2 = _norm(c_out, groups)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    @override
    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.film(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.norm2(h) * (1 + gamma) + beta
        return self.conv2(F.silu(h)) + self.skip(x)


class Attention(nn.Module):
    r"""Self-attention over the positions of ``(B, C, H, W)`` with heads of
    ``head_dim`` (``GN``, 1 x 1 ``qkv``, softmax of $q k^\top / \sqrt{d}$,
    1 x 1 ``out``), added to the input. Written out (no fused kernel), so
    its backward is deterministic."""

    def __init__(self, c: int, *, groups: int, head_dim: int) -> None:
        super().__init__()
        self.heads = max(1, c // head_dim)
        self.norm = _norm(c, groups)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.out = nn.Conv2d(c, c, 1)

    @override
    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, -1, h * w).unbind(1)
        a = torch.softmax(q.transpose(2, 3) @ k / math.sqrt(q.shape[2]), -1)
        y = (v @ a.transpose(2, 3)).reshape(b, c, h, w)  # (B, heads, d, HW)
        return x + self.out(y)


class Down(nn.Module):
    """A 3 x 3 convolution of stride 2."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    @override
    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        return self.conv(x)


class Up(nn.Module):
    """The nearest 2 x enlargement (by ``expand``, whose backward is a sum:
    deterministic) and a 3 x 3 convolution."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    @override
    def forward(self, x: Tensor, e: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = (
            x[:, :, :, None, :, None]
            .expand(b, c, h, 2, w, 2)
            .reshape(b, c, 2 * h, 2 * w)
        )
        return self.conv(x)


class FlowUNet(nn.Module):
    r"""$v_\theta(x_t, t, c)$: images ``(B, 3, S, S)`` ($S$ divisible by
    $2^{L-1}$, $L$ = ``len(channels)``), times ``(B,)`` and codes ``(B,
    code_dim)`` to velocities ``(B, 3, S, S)`` (module docstring).

    Args:
        code_dim: $D$, the code's dimension.
        channels: The channels of each resolution, from $S$ down.
        attn_levels: The indices of ``channels`` with attention.
        groups: The groups of each ``GroupNorm`` (fewer where the channels
            are fewer).
        embed: The width of the embedding $e$.
        time_dim: The sinusoidal embedding of the time.
        head_dim: The width of an attention head.
    """

    def __init__(
        self,
        code_dim: int,
        *,
        channels: Sequence[int] = CHANNELS,
        attn_levels: Sequence[int] = ATTN_LEVELS,
        groups: int = GROUPS,
        embed: int = EMBED,
        time_dim: int = TIME_DIM,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self._time_dim = time_dim
        self.time = nn.Sequential(
            nn.Linear(time_dim, embed), nn.SiLU(), nn.Linear(embed, embed)
        )
        kw = {"groups": groups, "embed": embed}
        attn = {"groups": groups, "head_dim": head_dim}
        self.inp = nn.Conv2d(3, channels[0], 3, padding=1)
        # each a layer: its modules, run in order (a ResBlock first joins a skip)
        down: list[list[nn.Module]] = []
        skips, c = [channels[0]], channels[0]
        for i, ch in enumerate(channels):
            for _ in range(2):
                down.append([ResBlock(c, ch, **kw)])
                if i in attn_levels:
                    down[-1].append(Attention(ch, **attn))
                c = ch
                skips.append(c)
            if i < len(channels) - 1:
                down.append([Down(c)])
                skips.append(c)
        self.mid = nn.ModuleList(
            [ResBlock(c, c, **kw), Attention(c, **attn), ResBlock(c, c, **kw)]
        )
        up: list[list[nn.Module]] = []
        for i, ch in reversed(list(enumerate(channels))):
            for _ in range(3):
                up.append([ResBlock(c + skips.pop(), ch, **kw)])
                if i in attn_levels:
                    up[-1].append(Attention(ch, **attn))
                c = ch
            if i > 0:
                up.append([Up(c)])
        self.down = nn.ModuleList(nn.ModuleList(layer) for layer in down)
        self.up = nn.ModuleList(nn.ModuleList(layer) for layer in up)
        self._down, self._up = down, up  # the same modules, typed
        self.out = nn.Sequential(
            _norm(c, groups), nn.SiLU(), nn.Conv2d(c, 3, 3, padding=1)
        )
        # last: the other weights of a seed do not depend on code_dim
        self.code = nn.Sequential(
            nn.Linear(code_dim, embed), nn.SiLU(), nn.Linear(embed, embed)
        )

    @override
    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        e = self.time(time_embedding(t, self._time_dim)) + self.code(c)
        h = self.inp(x)
        hs = [h]
        for layer in self._down:
            for m in layer:
                h = m(h, e)
            hs.append(h)
        for m in self.mid:
            h = m(h, e)
        for layer in self._up:
            if isinstance(layer[0], ResBlock):
                h = torch.cat([h, hs.pop()], 1)
            for m in layer:
                h = m(h, e)
        return self.out(h)


def _times(t: Tensor, x: Tensor) -> Tensor:
    """``t`` ``(B,)`` shaped to broadcast against ``x`` ``(B, ...)``."""
    return t.reshape(-1, *([1] * (x.dim() - 1))).to(x.dtype)


def flow_loss(v: Velocity, x1: Tensor, c: Tensor, *, t: Tensor, x0: Tensor) -> Tensor:
    r"""$L$ (module docstring) of the velocity ``v`` on the data ``x1``
    ``(B, ...)`` of the codes ``c``, at the times ``t`` ``(B,)`` and the
    noise ``x0`` (as ``x1``): the mean over the batch and the coordinates of
    $(v(x_t, t, c) - (x_1 - x_0))^2$."""
    s = _times(t, x1)
    xt = (1 - s) * x0 + s * x1
    return (v(xt, t, c) - (x1 - x0)).square().mean()


@torch.no_grad()
def euler_sample(v: Velocity, x0: Tensor, c: Tensor, steps: int) -> Tensor:
    r"""$x_1$ of the ODE $\dot x = v(x, t, c)$ from ``x0`` at $t = 0$ by
    Euler's method of ``steps`` steps: $x_{k+1} = x_k + v(x_k, t_k, c) / K$,
    $t_k = k / K$."""
    x = x0
    for k in range(steps):
        t = torch.full((len(x),), k / steps, dtype=x.dtype, device=x.device)
        x = x + v(x, t, c) / steps
    return x


def lr_at(step: int, *, lr: float, warmup: int, steps: int) -> float:
    """``lr`` after a linear warmup of ``warmup`` steps (from ``lr /
    warmup`` at step 0), then a cosine to 0 at ``steps``."""
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, steps - warmup)
    return 0.5 * lr * (1 + math.cos(math.pi * frac))


def fit_flow(
    net: nn.Module,
    codes: Tensor,
    frames: Tensor,
    *,
    steps: int,
    batch: int,
    lr: float,
    warmup: int,
    gen: torch.Generator,
) -> Tensor:
    r"""Learn ``net`` (on its device) by AdamW (betas 0.9, 0.999, eps
    $10^{-8}$, no weight decay, no clipping) at ``lr_at`` on $L$ of the
    frames u8 ``frames`` ``(N, 3, S, S)`` (CPU; $x_1 = 2 x / 255 - 1$) of the
    codes ``codes`` ``(N, D)`` (on the net's device). Each step draws from
    ``gen`` (a CPU generator) the ``batch`` indices uniformly with
    replacement, then $t \sim U(0, 1)$ and $x_0$. Returns the loss of each
    step ``(steps,)`` float64 (CPU).

    Raises:
        FloatingPointError: A loss is not finite.
    """
    dev = codes.device
    opt = torch.optim.AdamW(
        net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    losses = torch.zeros(steps, dtype=torch.float64, device=dev)
    net.train()
    for step in range(steps):
        idx = torch.randint(len(frames), (batch,), generator=gen)
        t = torch.rand(batch, generator=gen)
        x0 = torch.randn(batch, *frames.shape[1:], generator=gen)
        x1 = frames[idx].to(dev).float() * (2 / 255) - 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step, lr=lr, warmup=warmup, steps=steps)
        loss = flow_loss(net, x1, codes[idx.to(dev)], t=t.to(dev), x0=x0.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses[step] = loss.detach()
    net.eval()
    out = losses.cpu()
    if not bool(out.isfinite().all()):
        raise FloatingPointError(f"the flow's loss is not finite: {out[-5:].tolist()}")
    return out
