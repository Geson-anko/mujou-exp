r"""Direct prediction of whitened codes by a minGRU that reads the whole
history (``exp/vision/plans/foundation.md``, 4.1; the decisions of
2026-11-27 in ``exp/vision/results/foundation/README.md``).

**The sets** (``SeqSet``): sequences of codes $w_t \in \mathbb{R}^k$ one
after another (``w`` ``(N, k)``, ``bounds`` the sequences' starts and the
end), the ``anchors`` $t$ whose predictions are scored and each anchor's
block (for the block bootstrap). ``seq_set`` makes the anchors of a margin
$M$: $t - M$ and $t + M$ in $t$'s sequence (so the whole window before $t$
and every target $t \pm \Delta$ are there, in both directions), and those
``keep`` allows (a cut between $t - M$ and $t + M$ on the real data). The
past is the same thing on the reversed sequences (``SeqSet.reversed``: each
sequence flipped in place, every anchor moved with it, its block kept), so
$t + \Delta$ there is $t - \Delta$ here.

**The predictor** (``GRUPredictor``): of each lag $\Delta$ of ``lags`` a
two-point linear plus a residual read from a minGRU, $\hat w_{t+\Delta} =
\hat w^{\mathrm{lin}}_{t+\Delta} + A_\Delta y_t + b_\Delta$ (direct, no
rollout): $\hat w^{\mathrm{lin}} = M_\Delta [w_t; w_{t-\Delta}] +
c_\Delta$, $y_t$ the top of ``MinGRUStack`` (``layers`` layers of width
``width``) over the inputs $w$. At first $(M_\Delta, c_\Delta)$ are
dyn/copy's two-point linear (``dynpred.fit_linear`` on the training set,
$\alpha$ chosen on the validation set) and the heads $(A_\Delta,
b_\Delta)$ are 0 (dyn/copy's residual MLP's form): the epoch 0 is the
linear itself, so the family holds the lower bound and the early stopping
never ends above it on the validation set. All of them are learned. The
initial weights are drawn after ``torch.manual_seed(seed)`` on the CPU
(the global RNG restored after). (Value-seen fixes, 2026-11-27: without the
linear, a minGRU's top is a gated running mean of its inputs and stayed
above the one-point linear at 0.8 s on ``pca``; with the linear fixed, the
optimum left the family wherever it does not weigh $w_{t-\Delta}$ as the
two-point linear does, since a minGRU's state is a sum of decaying
exponentials of its inputs and holds no delay: so $M_\Delta$, $c_\Delta$
are learned from there.)

**The training** (``fit_gru``): AdamW (``lr``, ``weight_decay``) on the mean
squared error $\frac{1}{|J| L k} \sum_{j \in J} \sum_\Delta \lVert \hat
w_{a+j+\Delta} - w_{a+j+\Delta} \rVert^2$ of windows $w_a .. w_{a+W-1}$
($W$ = ``window``, $J = \{$``burn``$, .., W - 1\}$; the first ``burn``
positions are not scored), every state starting at 0. An epoch is one
pass over the training sequences: the windows ``window - burn`` apart
from a phase drawn uniformly in $\{0, .., W - \mathrm{burn} - 1\}$ by the
seed's generator (each time scored once; the first row $a$ past the
sequence's first $\max(0, \Delta_{\max} - \mathrm{burn})$ steps, so
$w_{t - \Delta}$ of the linear is in the sequence, and every target in
it), shuffled by the same generator into batches of ``batch`` windows.
(Value-seen fix, 2026-11-27: an epoch of every window, each time scored
$W - \mathrm{burn}$ times, was that many passes, coarser than the
overfitting: the validation loss was least at the first check and the
two-layer minGRU stayed above the bound of the ``kalman`` check.)
The validation loss is the same loss of the
validation sequences tiled by windows ``window - burn`` apart from their
first row (``tiled_loss``: each time scored once), at epoch 0 (the initial
weights) and after each epoch; the weights of the least (the earliest on a
tie) are kept, and the training stops ``patience`` epochs after it (or at
``max_epochs``).

**The scores** (``gru_sums``): at each anchor $t$ the predictions from the
window of the $W$ codes ending at $t$ (its last position), and of each lag
the blocks' sums of the numerator $\lVert \hat w_{t+\Delta} - w_{t+\Delta}
\rVert^2$ (``num_b``) and of the two denominators, the copy's $\lVert w_t -
w_{t+\Delta} \rVert^2$ (``copy_b``) and the mean's $\lVert \bar w -
w_{t+\Delta} \rVert^2$ (``clim_b``, $\bar w$ the training set's mean), and
the counts ``n_b``; with the shifted blocks also ``*_s``. Their ratios are
``exp.lib.dynpred``'s $R_{\mathrm{copy}}$ and $R_{\mathrm{clim}}$.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from exp.lib.dynpred import Triples, block_sums, fit_linear
from exp.lib.mingru import MinGRUStack

__all__ = [
    "LAGS",
    "GRUConf",
    "GRUFit",
    "GRUPredictor",
    "SeqSet",
    "best_window",
    "block_terms",
    "fit_gru",
    "gru_sums",
    "predict_at",
    "seq_set",
    "step_blocks",
    "tiled_loss",
]

LAGS = (1, 2, 4, 8, 16, 32)  # of stacks 0.8 s apart: 0.8 .. 25.6 s
_F64 = torch.float64


@dataclass(frozen=True)
class GRUConf:
    """The minGRU's settings (module docstring), by default the decided
    ones."""

    width: int = 256
    layers: int = 1
    window: int = 64
    burn: int = 8
    lags: tuple[int, ...] = LAGS
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch: int = 64
    patience: int = 10
    max_epochs: int = 1000
    chunk: int = 1024  # windows evaluated at a time

    @property
    def margin(self) -> int:
        """The anchors' margin $M$: the window before and the largest lag."""
        return max(self.window - 1, max(self.lags))


# -- the sets -----------------------------------------------------------------------


@dataclass(frozen=True)
class SeqSet:
    """Sequences and their anchors (module docstring): ``w`` ``(N, k)``,
    ``bounds`` ``(V + 1,)``, ``anchors`` ``(A,)`` (indices into ``w``),
    ``blocks`` ``(A,)`` of ``n_blocks`` and optionally the shifted blocks
    ``blocks_s`` of ``n_blocks_s``."""

    w: Tensor
    bounds: Tensor
    anchors: Tensor
    blocks: Tensor
    n_blocks: int
    blocks_s: Tensor | None = None
    n_blocks_s: int = 0

    def _flip(self) -> Tensor:
        """The index ``(N,)`` of each row's place in its flipped sequence."""
        idx = torch.arange(len(self.w))
        seq = torch.bucketize(idx, self.bounds[1:], right=True)
        return self.bounds[seq] + self.bounds[seq + 1] - 1 - idx

    def reversed(self) -> SeqSet:
        """The set of the reversed sequences (module docstring)."""
        flip = self._flip()
        w = torch.empty_like(self.w)
        w[flip] = self.w
        return SeqSet(w, self.bounds, flip[self.anchors], self.blocks, self.n_blocks,
                      self.blocks_s, self.n_blocks_s)  # fmt: skip

    def triples(self) -> Triples:
        """The anchors as ``dynpred.Triples`` (the two-point linear)."""
        return Triples(self.w.double(), self.anchors, self.blocks, self.n_blocks,
                       self.blocks_s, self.n_blocks_s)  # fmt: skip


def step_blocks(lengths: list[int], block: int, shift: int) -> Tensor:
    """Each row's block ``(N,)``: of each sequence the blocks of ``block``
    steps from its step ``shift`` on, the steps before it joined to the
    first and the remainder to the last (``metrics.contiguous_blocks``),
    numbered over the sequences."""
    out, at = [], 0
    for n in lengths:
        count = max((n - shift) // block, 1)
        q = (torch.arange(n) - shift).clamp(min=0) // block
        out.append(q.clamp(max=count - 1) + at)
        at += count
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def _renumber(b: Tensor) -> tuple[Tensor, int]:
    """``b`` numbered over its distinct values (in order) and their count."""
    uniq, inv = torch.unique(b, return_inverse=True)
    return inv, len(uniq)


def seq_set(
    w: Tensor,
    lengths: list[int],
    margin: int,
    block: int,
    *,
    keep: Tensor | None = None,
    shifted: bool = False,
) -> SeqSet:
    r"""The set of the sequences of ``lengths`` (``w`` ``(sum, k)``): the
    anchors of ``margin`` that ``keep`` ``(N,)`` (bool, all if None) allows,
    each sequence cut into blocks of ``block`` steps from its start (the
    remainder joined to the block before it), numbered over the blocks with
    anchors; with ``shifted`` also the blocks from the step ``block // 2``
    on (``*_s``).

    Raises:
        ValueError: ``lengths`` do not sum to the rows of ``w``.
    """
    if sum(lengths) != len(w):
        raise ValueError(f"the lengths sum to {sum(lengths)}, w has {len(w)} rows")
    bounds = torch.tensor(
        [0, *torch.tensor(lengths, dtype=torch.long).cumsum(0).tolist()]
    )
    seg = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])
    idx = torch.arange(len(w))
    ok = (idx - margin >= bounds[seg]) & (idx + margin < bounds[seg + 1])
    if keep is not None:
        ok &= keep
    anchors = idx[ok]
    blocks, nb = _renumber(step_blocks(lengths, block, 0)[anchors])
    if not shifted:
        return SeqSet(w, bounds, anchors, blocks, nb)
    blocks_s, nbs = _renumber(step_blocks(lengths, block, block // 2)[anchors])
    return SeqSet(w, bounds, anchors, blocks, nb, blocks_s, nbs)


# -- the predictor ------------------------------------------------------------------


class GRUPredictor(nn.Module):
    """The minGRU, its heads and the linear (module docstring): the linear's
    $M$ ``base_m`` ``(L, k, 2k)`` and $c$ ``base_c`` ``(L, k)`` are
    parameters, 0 until ``set_linear``."""

    lags: Tensor

    def __init__(self, k: int, conf: GRUConf) -> None:
        super().__init__()
        self.stack = MinGRUStack(k, conf.width, conf.layers)
        self.head = nn.Linear(conf.width, len(conf.lags) * k)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._k, self._lags = k, len(conf.lags)
        self.register_buffer("lags", torch.tensor(conf.lags))
        self.base_m = nn.Parameter(torch.zeros(len(conf.lags), k, 2 * k))
        self.base_c = nn.Parameter(torch.zeros(len(conf.lags), k))

    def set_linear(self, m: Tensor, c: Tensor) -> None:
        """Sets the linear's $M$ ``(L, k, 2k)`` and $c$ ``(L, k)``."""
        with torch.no_grad():
            self.base_m.copy_(m)
            self.base_c.copy_(c)

    def linear(self, w: Tensor, t: Tensor) -> Tensor:
        r"""$\hat w^{\mathrm{lin}}_{t+\Delta}$ ``(*t, L, k)`` of the rows
        ``t`` of the sequences ``w`` ``(N, k)``."""
        lags = self.lags
        x = torch.cat([w[t][..., None, :].expand(*t.shape, len(lags), -1),
                       w[t[..., None] - lags]], -1)  # (*t, L, 2k)  # fmt: skip
        return torch.einsum("lij,...lj->...li", self.base_m, x) + self.base_c

    @classmethod
    def build(cls, k: int, conf: GRUConf, seed: int, dev: torch.device) -> GRUPredictor:
        """The predictor of ``seed``'s initial weights (on the CPU, then
        moved to ``dev``)."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = cls(k, conf)
        return model.to(dev)

    @override
    def forward(self, x: Tensor) -> Tensor:
        r"""The residuals $A_\Delta y_t + b_\Delta$ ``(B, T, L, k)`` of the
        windows ``x`` ``(B, T, k)`` at every position."""
        y = self.head(self.stack.scan(x))
        return y.unflatten(-1, (self._lags, self._k))


# -- the training -------------------------------------------------------------------


def _tile_starts(s: SeqSet, conf: GRUConf, step: int, offset: int = 0) -> Tensor:
    """The windows' first rows ``(n,)`` of each sequence ``step`` apart from
    ``offset`` on (module docstring), every target in it."""
    head = max(0, max(conf.lags) - conf.burn)  # t - max lag in the sequence
    out = []
    for v in range(len(s.bounds) - 1):
        a, e = int(s.bounds[v]), int(s.bounds[v + 1])
        last = e - conf.window - max(conf.lags)  # a + W - 1 + max lag <= e - 1
        out.append(torch.arange(a + head + offset, last + 1, step))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long)


def _window_loss(
    model: GRUPredictor, w: Tensor, starts: Tensor, conf: GRUConf
) -> Tensor:
    """The loss (module docstring) of the windows of ``starts`` ``(B,)`` of
    ``w`` (on the model's device)."""
    pos = torch.arange(conf.window, device=w.device)
    x = w[starts[:, None] + pos]  # (B, W, k)
    t = starts[:, None] + pos[conf.burn :]  # (B, J)
    out = model(x)[:, conf.burn :] + model.linear(w, t)  # (B, J, L, k)
    lags = torch.tensor(conf.lags, device=w.device)
    tgt = w[t[..., None] + lags]  # (B, J, L, k)
    return (out - tgt).square().mean()


@torch.no_grad()
def tiled_loss(
    model: GRUPredictor, s: SeqSet, conf: GRUConf, dev: torch.device
) -> float:
    """The loss of ``s``'s sequences tiled from offset 0 (the windows'
    mean of equal weight: every window scores the same count)."""
    starts = _tile_starts(s, conf, conf.window - conf.burn)
    w = s.w.float().to(dev)
    total = 0.0
    for c in starts.split(conf.chunk):
        total += float(_window_loss(model, w, c.to(dev), conf)) * len(c)
    return total / max(len(starts), 1)


@dataclass(frozen=True)
class GRUFit:
    """A training's record: the validation loss of each epoch ``val`` (epoch
    0 the initial weights), the ``best`` epoch (whose weights the model
    keeps) and the ``epochs`` run."""

    val: tuple[float, ...]
    best: int
    epochs: int


def fit_gru(
    train: SeqSet, val: SeqSet, conf: GRUConf, seed: int, dev: torch.device
) -> tuple[GRUPredictor, GRUFit]:
    """The predictor of ``seed`` learned on ``train``, stopped on ``val``
    (module docstring), with the best epoch's weights.

    Raises:
        FloatingPointError: A training loss is not finite.
    """
    k = train.w.shape[1]
    model = GRUPredictor.build(k, conf, seed, dev)
    tr, va = train.triples(), val.triples()
    fits = [fit_linear(tr, va, lag) for lag in conf.lags]
    model.set_linear(torch.stack([f.a for f in fits]).float().to(dev),
                     torch.stack([f.b for f in fits]).float().to(dev))  # fmt: skip
    opt = torch.optim.AdamW(
        model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay
    )
    gen = torch.Generator().manual_seed(seed)
    w = train.w.float().to(dev)
    step = conf.window - conf.burn
    vals = [tiled_loss(model, val, conf, dev)]
    best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
    best = 0
    epoch = 0
    while epoch < conf.max_epochs and epoch - best < conf.patience:
        epoch += 1
        offset = int(torch.randint(step, (1,), generator=gen))
        tiles = _tile_starts(train, conf, step, offset)
        starts = tiles[torch.randperm(len(tiles), generator=gen)]
        model.train()
        for b in starts.split(conf.batch):
            loss = _window_loss(model, w, b.to(dev), conf)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(
                    f"the training loss is {loss.item()} at epoch {epoch}"
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        vals.append(tiled_loss(model, val, conf, dev))
        if vals[-1] < vals[best]:
            best = epoch
            best_state = {n: p.detach().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, GRUFit(tuple(vals), best, epoch)


# -- the oracle ---------------------------------------------------------------------


def best_window(
    gamma: Callable[[int], Tensor], window: int, lags: Sequence[int]
) -> Tensor:
    r"""The least error ``(L,)`` float64 of any linear map of the window
    $x = [w_{t-W+1}; ..; w_t]$ to $w_{t+\Delta}$ of a stationary process of
    mean 0 and autocovariance $\Gamma(\ell) = E[w_{t+\ell} w_t^\top]$
    (``gamma``, ``(k, k)``; $\Gamma(-\ell) = \Gamma(\ell)^\top$):
    $\mathrm{tr}(\Gamma(0) - K \Sigma^+ K^\top)$, $\Sigma_{ij} = \Gamma(i -
    j)$, $K_j = \Gamma(\Delta + W - 1 - j)$. Of a Gaussian process it is
    the least error of any predictor that sees the window (what
    ``predict_at`` sees)."""
    g = {d: gamma(d).double() for d in range(-(window - 1), window)}
    sigma = torch.cat([torch.cat([g[i - j] for j in range(window)], 1)
                       for i in range(window)], 0)  # fmt: skip
    pinv = torch.linalg.pinv((sigma + sigma.T) / 2, hermitian=True)
    out = []
    for lag in lags:
        k = torch.cat([gamma(lag + window - 1 - j).double() for j in range(window)], 1)
        out.append(float((g[0] - k @ pinv @ k.T).trace()))
    return torch.tensor(out, dtype=_F64)


# -- the scores ---------------------------------------------------------------------


@torch.no_grad()
def predict_at(
    model: GRUPredictor, s: SeqSet, conf: GRUConf, dev: torch.device
) -> Tensor:
    r"""$\hat w_{t+\Delta}$ ``(A, L, k)`` float64 (CPU) at the anchors, each
    from the window of the ``window`` codes ending at it."""
    w = s.w.float().to(dev)
    back = torch.arange(-conf.window + 1, 1, device=dev)
    out = []
    for c in s.anchors.split(conf.chunk):
        c = c.to(dev)
        x = w[c[:, None] + back]
        out.append((model(x)[:, -1] + model.linear(w, c)).double().cpu())
    k = s.w.shape[1]
    return torch.cat(out) if out else torch.zeros(0, len(conf.lags), k, dtype=_F64)


def gru_sums(
    model: GRUPredictor, s: SeqSet, mean: Tensor, conf: GRUConf, dev: torch.device
) -> dict[str, Tensor]:
    r"""The blocks' sums (module docstring) of ``s``: ``num_b``, ``copy_b``,
    ``clim_b`` ``(L, B)``, ``n_b`` ``(B,)`` float64 (and ``*_s``), $\bar w$
    = ``mean`` ``(k,)``."""
    pred = predict_at(model, s, conf, dev)
    w = s.w.double()
    now = w[s.anchors]
    lags = torch.tensor(conf.lags)
    tgt = w[s.anchors[:, None] + lags]  # (A, L, k)
    terms = {
        "num": (pred - tgt).square().sum(-1).T,  # (L, A)
        "copy": (now[:, None] - tgt).square().sum(-1).T,
        "clim": (mean.double() - tgt).square().sum(-1).T,
    }
    return block_terms(terms, s)


def block_terms(terms: Mapping[str, Tensor], s: SeqSet) -> dict[str, Tensor]:
    """The blocks' sums ``<name>_b`` ``(L, B)`` of each term ``(L, A)`` of
    the anchors, and the counts ``n_b`` (with the shifted blocks also
    ``*_b_s``)."""
    out: dict[str, Tensor] = {}
    ones = torch.ones(len(s.anchors), dtype=_F64)
    for blocks, nb, tail in (
        (s.blocks, s.n_blocks, ""),
        (s.blocks_s, s.n_blocks_s, "_s"),
    ):
        if blocks is None:
            continue
        for name, v in terms.items():
            out[f"{name}_b{tail}"] = block_sums(v, blocks, nb)
        out[f"n_b{tail}"] = block_sums(ones, blocks, nb)
    return out
