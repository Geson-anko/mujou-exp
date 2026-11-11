r"""The restoration through the predictor (M17, posthoc): a masked context view
through the encoder and the predictor, drawn back to pixels.

A token decoder (``token_decoder``, learned afterwards with the encoder held:
``fit_token_decoder``) maps a token of the target view's last time slot to the
patch of the stack's last frame it covers. ``restore`` shows a stack's context
view (the visible tokens only, as in training) to the encoder, predicts the
masked tokens with the predictor, and draws the last time slot: the visible
positions from the context's own tokens, the masked ones from the predictions.
It reads neither the target view nor the masked pixels.
"""

import torch
from torch import Tensor, nn

from exp.lib.optim import adam
from exp.video.jepa.loss import view_ids
from exp.video.jepa.vit import VisionModules, patchify_tubelets
from exp.video.measure.posthoc import FIT_BATCH, LR, STEPS, as_float

__all__ = [
    "HIDDEN",
    "TOKEN_BATCH",
    "fit_token_decoder",
    "last_slot",
    "patches_of",
    "picture_of",
    "restore",
    "token_decoder",
]

HIDDEN = 512
TOKEN_BATCH = 16 * FIT_BATCH  # tokens per step of the token decoder's fit


def token_decoder(
    dim: int, patch: int, channels: int, hidden: int = HIDDEN
) -> nn.Module:
    """A token ``(n, dim)`` to its patch ``(n, C, p, p)``:
    ``Linear - GELU - Linear - GELU - Linear`` (PyTorch's default init)."""
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, channels * patch * patch),
        nn.Unflatten(1, (channels, patch, patch)),
    )


def patches_of(frames: Tensor, patch: int) -> Tensor:
    """The patches ``(n gh gw, C, p, p)`` of ``frames`` ``(n, C, H, W)`` in
    the order (frame, row, column) of the tokens of a time slot."""
    n, c, h, w = frames.shape
    x = frames.reshape(n, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(-1, c, patch, patch)


def picture_of(patches: Tensor, n: int, grid: tuple[int, int]) -> Tensor:
    """The frames ``(n, C, H, W)`` of ``patches`` ``(n gh gw, C, p, p)``
    (``patches_of``'s inverse)."""
    gh, gw = grid
    _, c, p, _ = patches.shape
    x = patches.reshape(n, gh, gw, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(n, c, gh * p, gw * p)


def fit_token_decoder(
    tokens: Tensor,
    patches: Tensor,
    *,
    steps: int = STEPS,
    batch: int = TOKEN_BATCH,
    lr: float = LR,
    gen: torch.Generator,
) -> nn.Module:
    """A ``token_decoder`` of ``tokens`` ``(n, D)`` to ``patches`` ``(n, C, p,
    p)`` (u8 or in ``[0, 1]``, on the device of ``tokens``), fit as
    ``posthoc.fit_decoder`` fits its decoder: ``steps`` steps of ``adam(lr)``
    on the mean squared error of ``batch`` pairs drawn with replacement, the
    initial weights' seed and then the batches drawn from ``gen`` (CPU)."""
    seed = int(torch.randint(2**62, (1,), generator=gen))
    c, p = patches.shape[1], patches.shape[2]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dec = token_decoder(tokens.shape[1], p, c)
    dec = dec.to(tokens.device)
    params = list(dec.parameters())
    optimizer = adam(lr).bind(params)
    tokens = tokens.detach()
    for _ in range(steps):
        idx = torch.randint(len(tokens), (batch,), generator=gen).to(tokens.device)
        with torch.enable_grad():
            loss = ((dec(tokens[idx]) - as_float(patches[idx])) ** 2).mean()
            grads = torch.autograd.grad(loss, params)
        optimizer.step_(grads, 1.0)
    return dec


def last_slot(mods: VisionModules, tokens: Tensor) -> Tensor:
    """The tokens ``(n, gh gw, D)`` of the last time slot of ``tokens`` ``(n,
    T, D)`` (in the order of the ids)."""
    s = mods.grid[0] * mods.grid[1]
    return tokens[:, -s:]


@torch.no_grad()
def restore(
    mods: VisionModules, chunks: Tensor, mask: Tensor, dec: nn.Module, *, keep: int
) -> tuple[Tensor, Tensor]:
    """The last frames of the stacks ``chunks`` ``(n, N, C, H, W)`` (u8 or in
    ``[0, 1]``) restored from their context view of the spatial ``mask``
    ``(n, gh, gw)`` (True: masked, ``keep`` visible patches, a tube) through
    the encoder and the predictor (module docstring).

    Returns:
        The restored frames ``(n, C, H, W)`` and the masked pixels ``(n, H,
        W)`` bool.
    """
    x = as_float(chunks)
    n = len(x)
    slots = mods.grid3[0]
    tokens = patchify_tubelets(x, mods.patch, mods.tubelet)  # (n, T, P)
    t, k = tokens.shape[1], keep * slots
    masked = mask.flatten(1).repeat(1, slots).float()  # (n, T), the tube
    ctx_ids, tgt_ids = view_ids(masked, k)
    picked = tokens.gather(1, ctx_ids[..., None].expand(-1, -1, tokens.shape[2]))
    h = torch.cat([picked, ctx_ids[..., None].to(picked)], 2)
    ctx = mods.encoder(h, (k,))  # (n, K, D)
    pred = mods.predictor(ctx, ctx_ids, tgt_ids)  # (n, T - K, D)
    d = ctx.shape[2]
    full = ctx.new_zeros(n, t, d)
    full.scatter_(1, ctx_ids[..., None].expand(-1, -1, d), ctx)
    full.scatter_(1, tgt_ids[..., None].expand(-1, -1, d), pred)
    images = picture_of(dec(last_slot(mods, full).flatten(0, 1)), n, mods.grid)
    p = mods.patch
    pixels = mask.repeat_interleave(p, 1).repeat_interleave(p, 2)
    return images, pixels
