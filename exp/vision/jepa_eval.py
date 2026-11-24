r"""The evaluation of V18, V18b and V18c (``plans/jepa.md``, 「評価」 of V18,
V18b and V18c): the JEPA's runs and the yardsticks, each code a function from
a stack to a vector, through the same probes and the same statistics.

Probes (``stream_eval.StreamEvaluator.probes`` on the span ``[0, span)``,
``SPAN`` = 6000 s as V17 by default): the decoders learned afterwards and the linear
dynamics fit on the pairs of the span's odd 10-second blocks; ``recon_blur``,
``dyn_1/copy_1`` and ``rank`` of span, after and zurich. Pictures
``<name>-zurich.png`` and ``<name>-span.png`` (the true last frames and the
decoder's of seed 0).

The codes of a run's checkpoint (``arm_codes``, by default ``ARM_CODES`` of
its SIGReg's data point): ``z`` (or ``zflat``), the target side's code
(``JEPAModules.mu``; the modules are the code): the flat $z$, or of V18c's
$W \mathrm{vec}(t)$ of its tokens $t_p = R h_p$ (``build_run_jepa`` reads
its $P$ and $W$ from ``jepa.json``); ``zpca``, the PCA of $z$ on
the probes' fit stacks to ``ZPCA_DIM`` dimensions (``eval.PCA_SEED``; V18b's
arm M, whose $z$ has 16384); and the codes of ``TokenCode``: ``zbar``
($\bar z$), and of $h$, the encoder's tokens after its last ``LayerNorm``,
``h_mean``, ``h_flat`` and ``h_pool`` (their means over each block of the
coarse grid, ``jepa.COARSE_GRID``).

Statistics of a code (``CodeStats.of``), on the first stack of each of the
held-out video's val windows ($N$ = 1024) unless said:

- ``sigreg_zurich``: $S$ (``exp.lib.sigreg.sigreg``) of their codes along
  ``EVAL_DIRECTIONS`` fixed directions (``EVAL_DIRECTION_SEED``), beside
  that of as many $N(0, I)$ samples (``_gauss``) and of a constant code
  (``_const``) along the same directions;
- ``sigreg_heldin``: the mean $S$ of ``HELDIN_BATCHES`` batches of
  ``HELDIN_BATCH`` stacks drawn uniformly from the training units
  (``iid.Units`` of ``"all"``, ``HELDIN_SEED``; the same stacks for every
  code), the scale of the training's $S$, with ``_gauss`` and ``_const``;
- ``z_var``: the mean over the coordinates of their variance;
- ``participation``: the participation ratio $r_2$ of their covariance
  (``participation``), the $r_2$ of SIGReg's excess (``plans/jepa.md``);
- of a code made of tokens, ``rho_in`` (``jepa.rho_in``) and ``token_r2``
  (``token_r2``, fit on the probes' fit stacks): 1 for every code of the
  derivation's stack-constant family $z_{n,p} = A_p h(x_n)$, whatever the
  invertible $A_p$, where ``rho_in`` is 0 for $A_p = I$ only. Of a JEPA they
  are of its tokens ``JEPAModules.tokens`` ($z_p$, V18c's $t_p$).

Of a run's checkpoint also, in the row of each of its codes:
(``CodeStats.pred``) ``pred``, $L_{\mathrm{pred}}$ under ``EVAL_MASKS`` masks
of $F_k$ a stack (``EVAL_MASK_SEED``) spread to the run's grid
(``jepa.expand_masks``) in the space of the tokens ($t_p$ of V18c's), and
``pred_triv``, the same with each hidden $z_p$
guessed by its position's mean over the probes' fit stacks; ``pred_ratio`` =
``pred`` / ``pred_triv``; ``pred_triv_calib`` and ``pred_ratio_calib``, the
same with the means over the held-out video's calib windows (no shift of
domain); (``drift``) ``embed_drift`` and ``block_drift``, how far the
encoder's matrices moved from the run's step 0.

The yardsticks (``evaluate_yardsticks``) of V18's grid: ``init``, the JEPA
at step 0; ``init<latent>``, the PCA of ``init``'s $z$ to the VAE's
``model.latent`` (256) dimensions; ``color``, the mean of each patch's pixels
of each channel ($T \times 3n$ = 384; ``ColorCode``); ``pca``, the pixel PCA
of ``PCA_COMPONENTS`` components of the fit stacks (flattened $3n \times H
\times W$, ``eval.PCA_SEED``); ``v16`` and ``v17``, the $\mu$ of VAE
checkpoints (``--v16``, ``--v17``; their run's ``config.yaml`` two levels
up). Of a finer grid (``--patch 4``, V18b's): ``init_f``, arm F's step 0 (its
$z$ of V18's $D$, $d = G^2 \cdot$ ``TOKEN_DIM`` / $T$); ``init_m``,
``init_m_zbar`` and ``init_m_zflat``, arm M's ``zpca``, ``zbar`` and ``z`` at
step 0; ``color<patch>``; and ``v16``, ``v17`` given. Of V18c's
(``sigreg_on`` ``"mix"``): ``init_c``, its JEPA at step 0 (the code $W
\mathrm{vec}(t)$, the tokens $t$), ``init_c_h_mean``, its ``h_mean``, and
``color<patch>``.

``evaluate_run`` writes ``<run_dir>/posthoc_jepa/`` (``config.json``,
``jepa.jsonl`` a row per checkpoint and code, ``probes.jsonl`` a row per
probed checkpoint, code and group, the pictures ``<t>-<code>-*.png``);
``evaluate_yardsticks`` writes ``<out>/posthoc_jepa/`` of the same files, a
row per code (``code``)::

    python -m exp.vision.jepa_eval <run_dir> [--probe-steps 2000,6000] \
        [--codes z,h_mean]
    python -m exp.vision.jepa_eval --yardsticks [--config <config.yaml>] \
        [--patch 4] --v16 <ckpt> --v17 <ckpt> --out runs/v18-jepa/yardsticks
    python -m exp.vision.jepa_eval --yardsticks --patch 8 --token-dim 16 \
        --proj fixed --sigreg-on mix --out runs/v18c-baselines
"""

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Self

import torch
import yaml
from torch import Tensor, nn

from exp.lib import ckpt, pca
from exp.lib.provenance import configure_determinism
from exp.lib.rundir import write_json
from exp.lib.sigreg import sigreg
from exp.lib.tours.stacks import tours_root
from exp.vision.compose import from_dict
from exp.vision.eval import BATCH, FIT_STEPS, PCA_SEED
from exp.vision.iid import Units
from exp.vision.jepa import (
    COARSE_GRID,
    LATENT_DIM,
    PROJ_KINDS,
    SIGREG_ON,
    TOKEN_DIM,
    JEPAModules,
    build_jepa,
    directions,
    draw_masks,
    expand_masks,
    mask_expansion,
    mask_family,
    predict,
    rho_in,
    sigreg_points,
    take,
)
from exp.vision.jepa_iid import v16_config
from exp.vision.models import Block, PatchEmbed, build_vae, early_fusion
from exp.vision.schema import RunConf
from exp.vision.stream_eval import Code, StreamEvaluator, append_jsonl, spanned

__all__ = [
    "ARM_CODES",
    "CODES",
    "EVAL_DIRECTIONS",
    "EVAL_DIRECTION_SEED",
    "EVAL_MASKS",
    "EVAL_MASK_SEED",
    "HELDIN_BATCH",
    "HELDIN_BATCHES",
    "HELDIN_SEED",
    "OUT",
    "PCA_COMPONENTS",
    "PICTURES",
    "PROBE_STEPS",
    "SPAN",
    "TOKEN_POOLS",
    "TOKEN_RIDGE",
    "ZPCA_DIM",
    "CodeStats",
    "ColorCode",
    "PCACode",
    "TokenCode",
    "arm_codes",
    "build_run_jepa",
    "drift",
    "evaluate_run",
    "evaluate_yardsticks",
    "main",
    "participation",
    "pixels",
    "token_r2",
    "vae_at",
]

OUT = "posthoc_jepa"
PICTURES = ("zurich", "span")
PROBE_STEPS = (2000, 6000)  # the steps a run is probed at (and its last)
SPAN = 6000.0  # seconds of the probes' span: about 6000 fit pairs (V17's)
EVAL_MASKS, EVAL_MASK_SEED = 8, 37  # masks a stack for pred, from seed + this
EVAL_DIRECTIONS, EVAL_DIRECTION_SEED = 1024, 41
HELDIN_BATCHES, HELDIN_BATCH, HELDIN_SEED = 16, 128, 43
GAUSS_SEED = 47  # added to the seed: the N(0, I) samples of the references
PCA_COMPONENTS = 1024
TOKEN_RIDGE = 1e-3  # token_r2's ridge, relative to the mean variance of z_p
ZPCA_DIM = 1024  # zpca's dimensions: the D of V18's z, of the arms' codes
TOKEN_POOLS = ("z", "zbar", "h_mean", "h_flat", "h_pool")  # of TokenCode
CODES = ("zflat", "zpca", *TOKEN_POOLS)  # of arm_codes
# a run's codes by its SIGReg's data point: the judged code first (V18b)
ARM_CODES = {
    "flat": ("z", "h_mean"),
    "mean": ("zpca", "zbar", "zflat", "h_mean"),
    "mix": ("z", "h_mean"),  # V18c's: z = W vec(t)
}


class ColorCode:
    """The mean of each ``patch`` x ``patch`` patch's pixels of each of the
    $3n$ channels of the stacks (``early_fusion``, ``x / 255``): tokens
    ``(B, T, 3n)`` in row-major positions, $z$ ``(B, T 3n)``."""

    def __init__(self, patch: int) -> None:
        self._patch = patch

    def modules(self) -> Mapping[str, nn.Module]:
        return {}

    def tokens(self, x: Tensor) -> Tensor:
        images = early_fusion(x)
        b, c, h, w = images.shape
        p = self._patch
        means = images.reshape(b, c, h // p, p, w // p, p).mean((3, 5))
        return means.flatten(2).transpose(1, 2)

    def mu(self, x: Tensor) -> Tensor:
        return self.tokens(x).flatten(1)


class TokenCode:
    r"""The code of the stacks by the JEPA ``mods`` of ``pool``
    (``TOKEN_POOLS``): ``"z"`` the target side's $z$ (``JEPAModules.mu``),
    ``"zbar"`` its mean token $\bar z$ ``(B, d)``; of $h$ ``(B, T, w)``, the
    encoder's tokens after its last ``LayerNorm`` (the input of the
    projection), ``"h_mean"`` their mean over the positions ``(B, w)``,
    ``"h_flat"`` them flattened ``(B, T w)`` and ``"h_pool"`` the means of
    each $s \times s$ block of the grid $g = s G$ (``jepa.mask_expansion``),
    the coarse grid's tokens flattened ``(B, G^2 w)`` (``h_flat`` for $s$ =
    1).

    Raises:
        ValueError: ``pool`` is not of ``TOKEN_POOLS``.
    """

    def __init__(self, mods: JEPAModules, pool: str) -> None:
        if pool not in TOKEN_POOLS:
            raise ValueError(f"pool is one of {TOKEN_POOLS}, not {pool!r}")
        self._mods, self._pool = mods, pool

    def modules(self) -> Mapping[str, nn.Module]:
        return self._mods.modules()

    def mu(self, x: Tensor) -> Tensor:
        if self._pool == "z":
            return self._mods.mu(x)
        if self._pool == "zbar":
            return sigreg_points(self._mods.tokens(x), "mean")
        h = self._mods.encoder(early_fusion(x))  # (B, T, w)
        if self._pool == "h_mean":
            return h.mean(1)
        if self._pool == "h_flat":
            return h.flatten(1)
        b, t, w = h.shape
        g = COARSE_GRID
        s = mask_expansion(math.isqrt(t))
        return h.reshape(b, g, s, g, s, w).mean((2, 4)).flatten(1)


def pixels(x: Tensor) -> Tensor:
    """The stacks u8 ``(B, n, 3, H, W)`` flattened in ``x / 255``: ``(B, n 3
    H W)``."""
    return x.flatten(1).float() / 255


class PCACode:
    """The PCA (``exp.lib.pca``) of features $f$ of the stacks (``pixels``,
    or another code's ``mu``): $z = (f(x) - \\mu) W^\\top$."""

    def __init__(
        self, features: Callable[[Tensor], Tensor], mean: Tensor, comps: Tensor
    ) -> None:
        self._features, self._mean, self._comps = features, mean, comps

    @classmethod
    @torch.no_grad()
    def fit(
        cls,
        features: Callable[[Tensor], Tensor],
        stacks: Tensor,
        d: int,
        *,
        gen: torch.Generator,
        device: str,
    ) -> Self:
        """The PCA of ``d`` components (``pca.fit_pca``) of ``features`` of
        the stacks u8 ``(m, n, 3, H, W)``, on ``device``."""
        f = _encode(features, stacks, torch.device(device))
        return cls(features, *pca.fit_pca(f, d, gen=gen))

    def modules(self) -> Mapping[str, nn.Module]:
        return {}

    def mu(self, x: Tensor) -> Tensor:
        return pca.encode(self._features(x), self._mean, self._comps)


def _encode(fn: Callable[[Tensor], Tensor], x: Tensor, dev: torch.device) -> Tensor:
    """``fn`` of the stacks u8 ``x`` (CPU), ``BATCH`` at a time on ``dev``."""
    return torch.cat([fn(c.to(dev)).float() for c in x.split(BATCH)])


def participation(z: Tensor) -> Tensor:
    r"""The participation ratio $r_2 = (\mathrm{tr}\,\Sigma)^2 /
    \mathrm{tr}\,\Sigma^2$ of the codes ``(N, D)``, $\Sigma$ their sample
    covariance: $D$ for isotropic codes, 1 for codes on a line. The sample's
    is biased low, about $N D / (N + D)$ for $N(0, I_D)$. A 0-d float64
    tensor."""
    u = z.double() - z.double().mean(0)
    g = u @ u.T  # (N, N): tr Sigma and tr Sigma^2 up to the same factors
    return g.trace().square() / g.square().sum()


def token_r2(fit: Tensor, held: Tensor, ridge: float = TOKEN_RIDGE) -> Tensor:
    r"""The mean over the ordered pairs of positions $p \ne q$ of the $R^2$ on
    the tokens ``held`` ``(N, T, d)`` of the ridge regression of $z_q$ on
    $z_p$ (with an intercept) fit on the tokens ``fit`` ``(m, T, d)``.

    With $x_p$ the fit's $z_p$ less their mean $\bar z_p$, $W_{pq} = (X_p^\top
    X_p + \alpha_p I)^{-1} X_p^\top X_q$, $\alpha_p$ = ``ridge`` x
    $\mathrm{tr}(X_p^\top X_p) / d$, $\hat z_q = (z_p - \bar z_p) W_{pq} +
    \bar z_q$, and $R^2_{pq} = 1 - \sum_n \lVert z_{n,q} - \hat z_{n,q}
    \rVert^2 / \sum_n \lVert z_{n,q} - \langle z_q \rangle \rVert^2$ (the
    held tokens' mean). About 1 when every token is a linear function of
    any other, as $z_{n,p} = A_p h(x_n)$ for any invertible $A_p$; about 0
    for tokens independent of each other. The solve of each $p$ is made
    once and applied to every $q$ at once. A 0-d float64 tensor."""
    f, h = fit.double(), held.double()
    t, d = f.shape[1:]
    mean = f.mean(0)  # (T, d)
    x = f - mean  # (m, T, d)
    xf = x.flatten(1)  # (m, T d)
    u = h - mean  # (N, T, d): z less the fit's means
    total = (h - h.mean(0)).square().sum((0, 2))  # (q,)
    eye = torch.eye(d, dtype=f.dtype, device=f.device)
    r2 = torch.zeros((), dtype=f.dtype, device=f.device)
    for p in range(t):
        xp = x[:, p]  # (m, d)
        gram = xp.T @ xp
        alpha = ridge * gram.trace() / d
        w = torch.linalg.solve(gram + alpha * eye, xp.T @ xf)  # (d, T d): W_pq
        err = (u[:, p] @ w - u.flatten(1)).square()  # (N, T d)
        r = 1 - err.reshape(-1, t, d).sum((0, 2)) / total  # (q,)
        r2 = r2 + r.sum() - r[p]
    return r2 / (t * (t - 1))


def drift(init: nn.Module, now: nn.Module) -> dict[str, float]:
    r"""How far the encoder ``now`` moved from ``init`` (the same modules):
    $\lVert W - W_0 \rVert_F / \lVert W_0 \rVert_F$ of the patch embedding's
    matrix (``embed_drift``) and of the matrices of the ``Block``\ s joined
    (``block_drift``)."""

    def matrices(enc: nn.Module) -> tuple[list[Tensor], list[Tensor]]:
        embed = [m.proj.weight for m in enc.modules() if isinstance(m, PatchEmbed)]
        blocks = [
            m.weight
            for b in enc.modules()
            if isinstance(b, Block)
            for m in b.modules()
            if isinstance(m, nn.Linear)
        ]
        return embed, blocks

    def rel(w0: Sequence[Tensor], w: Sequence[Tensor]) -> float:
        num = sum(
            float((b - a).double().square().sum()) for a, b in zip(w0, w, strict=True)
        )
        den = sum(float(a.double().square().sum()) for a in w0)
        return math.sqrt(num / den)

    (e0, b0), (e1, b1) = matrices(init), matrices(now)
    with torch.no_grad():
        return {"embed_drift": rel(e0, e1), "block_drift": rel(b0, b1)}


def build_run_jepa(conf: RunConf, settings: Mapping[str, Any]) -> JEPAModules:
    """The JEPA of a run of ``conf`` and its ``jepa.json`` ``settings`` (its
    $d$, $P$ and $W$; V18's ``TOKEN_DIM``, ``"learned"`` and none where not
    recorded), at its step 0."""
    return build_jepa(
        conf,
        seed=conf.seed,
        token_dim=settings.get("token_dim", TOKEN_DIM),
        proj=settings.get("proj", "learned"),
        latent_dim=settings.get("latent_dim"),
    )


def arm_codes(
    mods: JEPAModules,
    names: Sequence[str],
    fit: Tensor,
    *,
    seed: int,
    device: str,
    zpca_dim: int = ZPCA_DIM,
) -> dict[str, Code]:
    r"""The codes ``names`` (``CODES``; module docstring) of the JEPA
    ``mods``: ``z`` and ``zflat`` the modules themselves, ``zpca`` the PCA of
    ``zpca_dim`` components of their $z$ on the stacks ``fit`` (from ``seed +
    eval.PCA_SEED``), the others ``TokenCode``\ s.

    Raises:
        ValueError: A name is not of ``CODES``.
    """
    unknown = [n for n in names if n not in CODES]
    if unknown:
        raise ValueError(f"the codes are of {CODES}, not {unknown}")
    codes: dict[str, Code] = {}
    for name in names:
        if name in ("z", "zflat"):
            codes[name] = mods
        elif name == "zpca":
            gen = torch.Generator().manual_seed(seed + PCA_SEED)
            codes[name] = PCACode.fit(mods.mu, fit, zpca_dim, gen=gen, device=device)
        else:
            codes[name] = TokenCode(mods, name)
    return codes


class CodeStats:
    """The statistics of codes (module docstring) for the run ``conf`` on the
    stacks of the evaluator ``ev`` and of the training units at ``root``.

    Args:
        conf: The run's config (its seed and device).
        root: The prepared Walking Tours.
        ev: The probes' evaluator, whose fit and zurich stacks are used.
        heldin_batches, heldin_batch: The held-in batches and their stacks.
    """

    def __init__(
        self,
        conf: RunConf,
        root: Path,
        ev: StreamEvaluator,
        *,
        heldin_batches: int = HELDIN_BATCHES,
        heldin_batch: int = HELDIN_BATCH,
    ) -> None:
        self._seed, self._dev = conf.seed, torch.device(conf.device)
        self._zx, self._fit = ev.zurich_stacks, ev.fit_stacks
        self._calib = ev.zurich_calib_stacks
        units = Units(conf, root, "all", seed=conf.seed + HELDIN_SEED)
        self._heldin = [units.draw(heldin_batch) for _ in range(heldin_batches)]
        self._refs: dict[int, dict[str, float]] = {}

    def _dirs(self, d: int) -> Tensor:
        gen = torch.Generator().manual_seed(self._seed + EVAL_DIRECTION_SEED)
        return directions(EVAL_DIRECTIONS, d, gen).to(self._dev)

    def _references(self, d: int) -> dict[str, float]:
        """$S$ of $N(0, I)$ samples and of a constant code, as many as the
        zurich stacks and as the held-in batches, along ``_dirs(d)``."""
        if d not in self._refs:
            a = self._dirs(d)
            gen = torch.Generator().manual_seed(self._seed + GAUSS_SEED)

            def s(z: Tensor) -> float:
                return float(sigreg(z.to(self._dev), z[:0].to(self._dev), a))

            n, b = len(self._zx), len(self._heldin[0])
            gauss = [s(torch.randn(b, d, generator=gen)) for _ in self._heldin]
            self._refs[d] = {
                "sigreg_zurich_gauss": s(torch.randn(n, d, generator=gen)),
                "sigreg_zurich_const": s(torch.zeros(n, d)),
                "sigreg_heldin_gauss": sum(gauss) / len(gauss),
                "sigreg_heldin_const": s(torch.zeros(b, d)),
            }
        return self._refs[d]

    @torch.no_grad()
    def of(self, code: Code) -> dict[str, float]:
        """``sigreg_zurich``, ``sigreg_heldin``, ``z_var`` (with the
        references), ``participation`` and, of a code with ``tokens``,
        ``rho_in`` and ``token_r2``."""
        z = _encode(code.mu, self._zx, self._dev)  # (N, D)
        a = self._dirs(z.shape[1])
        held = [
            float(sigreg(zb, zb[:0], a))
            for zb in (_encode(code.mu, x, self._dev) for x in self._heldin)
        ]
        row = {
            "sigreg_zurich": float(sigreg(z, z[:0], a)),
            "sigreg_heldin": sum(held) / len(held),
            "z_var": float(z.var(0).mean()),
            "participation": float(participation(z)),
        }
        row |= self._references(z.shape[1])
        tokens = getattr(code, "tokens", None)  # of a code made of tokens
        if tokens is not None:
            held = _encode(tokens, self._zx, self._dev)  # (N, T, d)
            row["rho_in"] = float(rho_in(held))
            fit = _encode(tokens, self._fit, self._dev)
            row["token_r2"] = float(token_r2(fit, held))
        return row

    @torch.no_grad()
    def pred(self, mods: JEPAModules, k: int, expand: int = 1) -> dict[str, float]:
        """``pred``, ``pred_triv``, ``pred_ratio`` and those of the calib
        windows (module docstring) of the JEPA ``mods`` under masks of the
        coarse grid hiding ``k``, spread by ``expand`` (``jepa.expand_masks``)
        to the JEPA's grid."""
        z_bar = _encode(mods.tokens, self._fit, self._dev).mean(0)  # (T, d)
        z_cal = _encode(mods.tokens, self._calib, self._dev).mean(0)  # (T, d)
        t = len(z_bar)
        family = mask_family(k, COARSE_GRID).to(self._dev)
        gen = torch.Generator().manual_seed(self._seed + EVAL_MASK_SEED)
        masks = draw_masks(family, EVAL_MASKS * len(self._zx), gen)
        masks = expand_masks(masks, expand).reshape(EVAL_MASKS, len(self._zx), t)
        err = triv = cal = 0.0
        for i, x in enumerate(self._zx.split(BATCH)):
            images = early_fusion(x.to(self._dev))
            tokens = mods.proj(mods.encoder(images))  # (b, T, d)
            for mask in masks[:, i * BATCH : i * BATCH + len(x)]:
                z_hat, hide = predict(mods, images, mask)
                target = take(tokens, hide)
                err += float((z_hat - target).double().square().sum())
                triv += float((z_bar[hide] - target).double().square().sum())
                cal += float((z_cal[hide] - target).double().square().sum())
        count = EVAL_MASKS * len(self._zx) * expand**2 * k * z_bar.shape[1]
        pred, pred_triv, pred_cal = err / count, triv / count, cal / count
        return {
            "pred": pred,
            "pred_triv": pred_triv,
            "pred_ratio": pred / pred_triv,
            "pred_triv_calib": pred_cal,
            "pred_ratio_calib": pred / pred_cal,
        }


def _start(out: Path, settings: Mapping[str, Any]) -> None:
    """``out/config.json`` of ``settings``, empty ``jepa.jsonl`` and
    ``probes.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", dict(settings))
    for f in ("jepa.jsonl", "probes.jsonl"):
        (out / f).write_text("")


def evaluate_run(
    run_dir: Path,
    *,
    root: Path | None = None,
    span: float = SPAN,
    probe_steps: Sequence[int] = PROBE_STEPS,
    fit_steps: int = FIT_STEPS,
    spacing: Mapping[str, float] | None = None,
    heldin: tuple[int, int] = (HELDIN_BATCHES, HELDIN_BATCH),
    codes: Sequence[str] | None = None,
    zpca_dim: int = ZPCA_DIM,
) -> Path:
    """Evaluate every checkpoint of the JEPA run in ``run_dir`` (its
    ``config.yaml`` and ``jepa.json``) in the order of ``t``, under
    ``configure_determinism``, into ``run_dir / OUT`` (module docstring):
    the statistics of every checkpoint's ``codes`` (``arm_codes``; by default
    ``ARM_CODES`` of the run's SIGReg's data point), the probes of the steps
    of ``probe_steps`` and of the last one, on the span ``[0, span)``.
    ``spacing`` passes the spacings of ``StreamEvaluator`` by name and
    ``heldin`` the held-in batches and their stacks. Returns the directory.

    Raises:
        FileNotFoundError: The run has no checkpoint.
    """
    conf = from_dict(yaml.safe_load((run_dir / "config.yaml").read_text()))
    settings = json.loads((run_dir / "jepa.json").read_text())
    view = spanned(conf, span)
    determinism = configure_determinism(conf.device)
    paths = sorted(run_dir.glob("ckpt/*.pt"), key=lambda p: float(p.stem))
    if not paths:
        raise FileNotFoundError(f"no checkpoint under {run_dir / 'ckpt'}")
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = run_dir / OUT
    ev = StreamEvaluator(
        view, run_dir, root, out, fit_steps=fit_steps, c=1, **(spacing or {})
    )
    stats = CodeStats(view, root, ev, heldin_batches=heldin[0], heldin_batch=heldin[1])
    sigreg_on = settings.get("sigreg_on", "flat")  # V18's runs: flat
    expand = settings.get("expand", 1)
    names = list(ARM_CODES[sigreg_on] if codes is None else codes)
    _start(
        out,
        ev.settings()
        | {
            "span": view.measure.duration,
            "probe_steps": list(probe_steps),
            "ckpts": [p.name for p in paths],
            "codes": names,
            "token_dim": settings.get("token_dim", TOKEN_DIM),
            "proj": settings.get("proj", "learned"),
            "latent_dim": settings.get("latent_dim"),
            "sigreg_on": sigreg_on,
            "expand": expand,
            "zpca_dim": zpca_dim,
            "eval_masks": EVAL_MASKS,
            "eval_directions": EVAL_DIRECTIONS,
            "heldin": list(heldin),
            "determinism": determinism,
        },
    )
    mods = build_run_jepa(conf, settings)
    init = build_run_jepa(conf, settings)  # the weights of step 0
    for path in paths:
        head = ckpt.load_eval(path, mods.modules())
        step = int(head["frame"]) // settings["batch"]
        stamp = {"t": head["t"], "step": step}
        shared = stats.pred(mods, settings["k"], expand)
        shared |= drift(init.encoder, mods.encoder)
        probed = step in probe_steps or path == paths[-1]
        found = arm_codes(
            mods,
            names,
            ev.fit_stacks,
            seed=conf.seed,
            device=conf.device,
            zpca_dim=zpca_dim,
        )
        for name, code in found.items():
            row = stamp | {"code": name} | stats.of(code) | shared
            append_jsonl(out / "jepa.jsonl", [row])
            if probed:
                rows = ev.probes(code, f"{path.stem}-{name}", PICTURES)
                append_jsonl(
                    out / "probes.jsonl", [stamp | {"code": name} | r for r in rows]
                )
    return out


def vae_at(path: Path, device: str) -> Code:
    """The VAE of the checkpoint ``path`` (its run's ``config.yaml`` two
    levels up) on ``device``."""
    d = yaml.safe_load((path.parent.parent / "config.yaml").read_text())
    conf = replace(from_dict(d), device=device)
    mods = build_vae(conf, seed=conf.seed)
    ckpt.load_eval(path, mods.modules())
    return mods


def evaluate_yardsticks(
    conf: RunConf,
    out_dir: Path,
    *,
    root: Path | None = None,
    v16: Path | None = None,
    v17: Path | None = None,
    span: float = SPAN,
    fit_steps: int = FIT_STEPS,
    spacing: Mapping[str, float] | None = None,
    heldin: tuple[int, int] = (HELDIN_BATCHES, HELDIN_BATCH),
    pca_components: int = PCA_COMPONENTS,
    zpca_dim: int = ZPCA_DIM,
    token_dim: int = TOKEN_DIM,
    proj: str = "learned",
    sigreg_on: str = "flat",
) -> Path:
    """Evaluate the yardsticks (module docstring) of the JEPA of ``conf``
    into ``out_dir / OUT``: of ``sigreg_on`` ``"mix"`` (V18c's) ``init_c``
    (``build_jepa`` of ``token_dim``, ``proj`` and ``LATENT_DIM``),
    ``init_c_h_mean`` and ``color<patch>``; else of V18's grid ``init``,
    ``color``, ``pca``, of a finer grid those of the arms F and M at step 0
    and ``color<patch>``; and those of ``v16`` and ``v17`` given. Arguments
    as ``evaluate_run``'s. Returns the directory.

    Raises:
        ValueError: Arm F's $d$ is not a whole number of at least 1.
    """
    view = spanned(conf, span)
    determinism = configure_determinism(conf.device)
    root = tours_root(conf.data.tours.data_version) if root is None else root
    out = out_dir / OUT
    ev = StreamEvaluator(
        view, out_dir, root, out, fit_steps=fit_steps, c=1, **(spacing or {})
    )
    stats = CodeStats(view, root, ev, heldin_batches=heldin[0], heldin_batch=heldin[1])
    gen = torch.Generator().manual_seed(conf.seed + PCA_SEED)
    fit, latent = ev.fit_stacks, conf.model.latent
    grid = conf.data.tours.size // conf.model.patch
    extra: dict[str, Any] = {}
    if sigreg_on == "mix":  # V18c's
        init_c = build_jepa(
            conf, seed=conf.seed, token_dim=token_dim, proj=proj, latent_dim=LATENT_DIM
        )
        codes: dict[str, Code] = {
            "init_c": init_c,
            "init_c_h_mean": TokenCode(init_c, "h_mean"),
            f"color{conf.model.patch}": ColorCode(conf.model.patch),
        }
        extra = {"token_dim": token_dim, "proj": proj, "latent_dim": LATENT_DIM}
    elif mask_expansion(grid) == 1:  # V18's
        init = build_jepa(conf, seed=conf.seed)
        codes = {
            "init": init,
            # init's z in as many dimensions as the VAE's mu
            f"init{latent}": PCACode.fit(
                init.mu, fit, latent, gen=gen, device=conf.device
            ),
            "color": ColorCode(conf.model.patch),
            "pca": PCACode.fit(
                pixels, fit, pca_components, gen=gen, device=conf.device
            ),
        }
    else:  # V18b's arms: F of V18's D, M of V18's d
        d_f, rest = divmod(COARSE_GRID**2 * TOKEN_DIM, grid * grid)
        if d_f < 1 or rest:
            raise ValueError(f"arm F's d of {grid * grid} tokens is not whole")
        arm_m = arm_codes(
            build_jepa(conf, seed=conf.seed),
            ("zpca", "zbar", "zflat"),
            fit,
            seed=conf.seed,
            device=conf.device,
            zpca_dim=zpca_dim,
        )
        codes = {"init_f": build_jepa(conf, seed=conf.seed, token_dim=d_f)}
        codes |= {
            "init_m": arm_m["zpca"],
            "init_m_zbar": arm_m["zbar"],
            "init_m_zflat": arm_m["zflat"],
            f"color{conf.model.patch}": ColorCode(conf.model.patch),
        }
        extra = {"token_dims": {"init_f": d_f, "init_m": TOKEN_DIM}}
        extra["zpca_dim"] = zpca_dim
    for name, path in (("v16", v16), ("v17", v17)):
        if path is not None:
            codes[name] = vae_at(path, conf.device)
    _start(
        out,
        ev.settings()
        | {
            "span": view.measure.duration,
            "codes": list(codes),
            "v16": None if v16 is None else str(v16),
            "v17": None if v17 is None else str(v17),
            "pca_components": pca_components,
            "patch": conf.model.patch,
            "sigreg_on": sigreg_on,
            "eval_directions": EVAL_DIRECTIONS,
            "heldin": list(heldin),
            "determinism": determinism,
        }
        | extra,
    )
    for name, code in codes.items():
        append_jsonl(out / "jepa.jsonl", [{"code": name} | stats.of(code)])
        rows = ev.probes(code, name, PICTURES)
        append_jsonl(out / "probes.jsonl", [{"code": name} | r for r in rows])
    return out


def main(argv: Sequence[str] | None = None) -> Path:
    """``python -m exp.vision.jepa_eval``: ``evaluate_run`` of a run, or
    with ``--yardsticks`` ``evaluate_yardsticks``. Returns the directory."""
    ap = argparse.ArgumentParser(
        prog="python -m exp.vision.jepa_eval",
        description="Evaluate the checkpoints of a JEPA run, or the yardsticks.",
    )
    ap.add_argument("run_dir", type=Path, nargs="?", help="the run's directory")
    ap.add_argument("--yardsticks", action="store_true")
    ap.add_argument("--config", type=Path, help="yardsticks: (default v16_config)")
    ap.add_argument("--patch", type=int, help="yardsticks: instead of the config's")
    ap.add_argument("--token-dim", type=int, default=TOKEN_DIM, help="yardsticks: d")
    ap.add_argument(
        "--proj", choices=PROJ_KINDS, default="learned", help="yardsticks: P"
    )
    ap.add_argument(
        "--sigreg-on",
        choices=SIGREG_ON,
        default="flat",
        help="yardsticks: mix for V18c's",
    )
    ap.add_argument("--v16", type=Path, help="yardsticks: a VAE checkpoint")
    ap.add_argument("--v17", type=Path, help="yardsticks: a VAE checkpoint")
    ap.add_argument("--out", type=Path, help="yardsticks: the output directory")
    ap.add_argument("--device", default="cuda", help="yardsticks: the device")
    ap.add_argument("--root", type=Path, help="the data (default: tours_root)")
    ap.add_argument(
        "--span", type=float, default=SPAN, help="seconds of the span (default: SPAN)"
    )
    ap.add_argument("--fit-steps", type=int, default=FIT_STEPS, help="per decoder")
    ap.add_argument(
        "--probe-steps",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default=PROBE_STEPS,
        help="comma-separated steps (default: PROBE_STEPS; the last always)",
    )
    ap.add_argument(
        "--codes",
        type=lambda s: tuple(s.split(",")),
        help="a run: comma-separated codes (default: ARM_CODES of its run)",
    )
    args = ap.parse_args(argv)
    if not args.yardsticks:
        if args.run_dir is None:
            ap.error("give a run_dir, or --yardsticks")
        return evaluate_run(
            args.run_dir,
            root=args.root,
            span=args.span,
            probe_steps=args.probe_steps,
            fit_steps=args.fit_steps,
            codes=args.codes,
        )
    if args.out is None:
        ap.error("--yardsticks needs --out")
    d = v16_config() if args.config is None else yaml.safe_load(args.config.read_text())
    d["device"] = args.device
    if args.patch is not None:
        d["model"]["patch"] = args.patch
    return evaluate_yardsticks(
        from_dict(d),
        args.out,
        root=args.root,
        v16=args.v16,
        v17=args.v17,
        span=args.span,
        fit_steps=args.fit_steps,
        token_dim=args.token_dim,
        proj=args.proj,
        sigreg_on=args.sigreg_on,
    )


if __name__ == "__main__":
    print(main())
