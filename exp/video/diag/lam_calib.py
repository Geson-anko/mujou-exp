r"""DIAGNOSIS ONLY (M17): the calibration of lambda of ``m17/tok_n16`` without
training, on the untrained ViT (seed 100) and the training stream
(``StackedTourFrames``, Walking Tours 128 x 128, data ``3c3e6805fdaa``) of
token width $d$. It measures what the Gaussian model of ``lam_window`` needs;
``lam_window`` turns the result into the windows. One GPU (``--device``),
minutes; it prints progress lines and a last line ``RESULT {...}`` (JSON).
The results quoted in ``exp/video/results/m17.md`` are kept in
``exp/video/results/m17-calib/``.

Definitions (along a unit direction $a$, $u = a^\top s$ of a token $s$; $T$ =
512 tokens a stack):

- $\rho$, the correlation of two distinct tokens of one stack:
  $\rho_a = \mathrm{mean}_b \big[(\sum_t \delta u)^2 - \sum_t \delta u^2\big]
  / (T (T - 1) \mathrm{var}(u))$, $\delta u = u - \bar u$, the mean and var
  over all tokens of the set (``rho_of``), averaged over 256 random $a$
- the lag correlation of the stack sums $k$ frames apart in one stream:
  $\mathrm{mean}\, [\sum_t \delta u_f \sum_t \delta u_{f-k}] / (T^2
  \mathrm{var}(u))$, over the pairs of frames where both stacks exist
  (``stack_corr``); for tokens of a common component of correlation $\rho$
  carried over $k$ frames with correlation $\phi_k$ it is $\rho\,\phi_k$
- $S$, ``jepa.loss.weighted_sigreg`` of the tokens of the $B$ stacks of a frame,
  without and with the buffer of past tokens (``jepa.nodes.PastTokens``), along
  256 fresh directions each frame as in training

``pooled`` (backs 「$\lambda$ の格子の導出」 of 「トークンの SIGReg」; snapshot
``88c60e2``, std-01, seeds: model 100, directions 5, masks 6, buffer insert 7,
val 1)::

    python -m exp.video.diag.lam_calib pooled <d> 1536 512

the buffer of those runs, 8192 past tokens of all $B$ = 32 streams with 64
inserted a frame (``--buffer 256 --insert 2`` per stream: the same draws). It
measures $\rho$ (and the share of the variance of the position means) on the
held-out Zurich val stacks and on the training stream; $S$ of the raw tokens
(``h``), of them affinely whitened by the moments of a pre-pass (``hw``), and
of forced collapses within the stack (every token of a stack replaced by one:
``tok`` the token of a fixed position 203, ``white`` the whitened stack mean,
``gauss`` an iid $N(0, I)$ draw); each collapse fills its own buffer with its
own tokens (its steady state). Also the prediction loss of the untrained
predictor (``L_pred``), of the zero predictor and of the stack mean, and $S$
of the Gaussian model at the measured $\rho$.

``per-stream`` (backs 「回す前に決めたこと」 of 「ストリームごとのバッファ」;
snapshot ``81c2953``, mujou ``7a8b72e``, std-00 and std-02; seeds: model 100,
directions 5, insert 7, the $\rho$ directions 3, P's iid $g$ 12, I 13, the
optimal atoms 11)::

    python -m exp.video.diag.lam_calib per-stream 16 32 768 512
    python -m exp.video.diag.lam_calib per-stream 16 1 3072 2560

the per-stream buffer (2048 a stream, 512 inserted a frame). It measures $S$
without and with the buffer of ``h``, ``hw``, ``white`` (C), ``P`` (the
tokens of the position alone, $g(p)$ iid $N(0, I)$ fixed), ``Popt`` (P with
the 512 atoms that minimize the population Epps-Pulley distance, by Adam:
``optimal_atoms``) and ``I`` (iid $N(0, I)$ tokens, fresh each frame), and
$\rho$ and the lag correlations ($K$ = 48 lags) of ``h``, ``hw`` and
``white``, with the population (pre-pass) moments and the record's own.

The data comes from ``$MUJOU_EXP_DATA`` (default the fleet's
``/data/geson/mujou-exp/.fleet/data``); on the fleet: ``just fleet x std-0X --
bash -c 'CUDA_VISIBLE_DEVICES=0 python -m exp.video.diag.lam_calib ...'``.
"""

import argparse
import json
import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from exp.video.compose import compose
from exp.video.data.frames import StackedTourFrames
from exp.video.diag.sigreg_sims import gauss_stacks
from exp.video.jepa.loss import PastSums, jepa_loss_packed, weighted_sigreg
from exp.video.jepa.masks import multiblock_masks, pack_views, spatial_keep
from exp.video.jepa.nodes import PastTokens, draw_directions
from exp.video.jepa.vit import VisionModules, build_vision
from exp.video.programs.vision import PackedMeasures, tours_root
from exp.video.schema.run import RunConf

DATA = "/data/geson/mujou-exp/.fleet/data"
SEED = 100  # the untrained model
P_FIX = 203  # the fixed position of the 'tok' collapse


def rho_of(tok: torch.Tensor, a: torch.Tensor) -> dict[str, float]:
    """$\\rho$ (module docstring) of the tokens ``(n, T, d)`` along the unit
    directions ``a`` ``(A, d)``, and the share of the variance of the
    position means (the part common to the stacks at a position)."""
    T = tok.shape[1]
    u = tok @ a.T  # (n, T, A)
    du = u - u.mean((0, 1))
    var = du.pow(2).mean((0, 1))
    s, q = du.sum(1), du.pow(2).sum(1)  # (n, A)
    rho = ((s**2 - q).mean(0) / (T * (T - 1) * var)).mean()
    pos = du.mean(0).pow(2).mean(0) / var  # var of the position means / var
    return {
        "rho": float(rho),
        "pos_share": float(pos.mean()),
        "var_u": float(var.mean()),
    }


def stack_corr(
    S1: torch.Tensor,
    S2: torch.Tensor,
    V: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    T: int,
    K: int,
) -> tuple[float, list[float]]:
    """$\\rho$ and the lag correlations $1..K$ (module docstring) from the
    per-frame sums over each stack of $u$ and $u^2$, ``S1`` and ``S2`` ``(R +
    K, B, A)``, the stacks that exist ``V`` ``(R + K, B)``, and the moments
    ``mu`` and ``var`` ``(A,)`` of $u$: $\\rho$ over the last ``R`` frames,
    the lags between them and the ``K`` frames before (NaN where no pair
    exists)."""
    c1 = S1 - T * mu  # sum of du
    c2 = S2 - 2 * mu * S1 + T * mu**2  # sum of du^2
    m = V[K:]
    within = ((c1[K:] ** 2 - c2[K:]) / (T * (T - 1) * var))[m].mean()
    lags = []
    for lag in range(1, K + 1):
        both = V[K:] & V[K - lag : len(V) - lag]
        prod = c1[K:] * c1[K - lag : len(V) - lag] / (T * T * var)
        lags.append(float(prod[both].mean()) if bool(both.any()) else float("nan"))
    return float(within), lags


def whitening(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean and the inverse Cholesky factor of the covariance of the rows
    of ``x``: ``(x - mu) @ W.T`` has the identity covariance."""
    mu = x.mean(0)
    return mu, torch.linalg.inv(torch.linalg.cholesky(torch.cov(x.T)))


class Setup:
    """The config ``m17/tok_n16`` at width ``d`` and ``streams``, the untrained
    modules in eval mode, and the shapes of a stack."""

    def __init__(self, d: int, streams: int | None, dev: str) -> None:
        overrides = ["+experiment=m17/tok_n16", f"seed={SEED}"]
        overrides += [f"vision.token_dim={d}", f"device={dev}"]
        if streams is not None:
            overrides.append(f"data.streams={streams}")
        self.conf: RunConf = compose(overrides)
        v = self.conf.vision
        assert v is not None and v.token_dim == d
        self.v = v
        self.mods: VisionModules = build_vision(self.conf, seed=SEED)
        for m in self.mods.named().values():
            m.eval()
        self.d, self.dev = d, dev
        self.B = self.conf.data.streams
        self.T, self.M = self.mods.segments[0], v.sigreg_directions
        self.grid = self.mods.grid
        self.keeps = [spatial_keep(self.grid, m) for m in v.context_masks]

    def stream(self, n: int) -> StackedTourFrames:
        """``n`` streams of stacks of the training tours."""
        tours, v = self.conf.data.tours, self.v
        assert tours is not None
        return StackedTourFrames(
            tours_root(tours),
            n=n,
            held_out=tours.held_out,
            frames=v.frames,
            stride=v.stride,
            advance=tours.advance,
        )

    def target_tokens(self, x: torch.Tensor, mask_seed: int) -> torch.Tensor:
        """The target tokens ``(b, T, d)`` of stacks ``x``: the whole stack
        through the encoder (the mask of the context view, drawn from
        ``mask_seed``, only shapes the packing)."""
        v = self.v
        m0 = multiblock_masks(
            x.shape[0],
            self.grid,
            v.context_masks[0],
            keep=self.keeps[0],
            generator=torch.Generator().manual_seed(mask_seed),
        )
        h = pack_views(
            x.to(self.dev),
            [m0.to(self.dev)],
            keeps=self.keeps[:1],
            patch=v.patch,
            tubelet=v.tubelet,
        )[0]
        return self.mods.encoder(h[:, : self.T], (self.T,))

    def S(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        a: torch.Tensor,
        past: PastSums | None = None,
    ) -> float:
        """S of the tokens ``(b, T, d)`` of the streams ``w`` ``(b,)``."""
        rows = x.reshape(-1, self.d)
        return float(weighted_sigreg(rows, w.repeat_interleave(self.T), a, past))

    def past_tokens(self, capacity: int, insert: int) -> PastTokens:
        """Past tokens of ``capacity`` rows and ``insert`` a frame per stream."""
        return PastTokens(
            capacity=capacity,
            insert=insert,
            streams=self.B,
            tokens=self.T,
            dim=self.d,
            directions=self.M,
            device=self.dev,
        )


def gauss_model_S(
    su: Setup, rho: float, n: int, g: torch.Generator, reps: int = 3
) -> float:
    """S of ``n`` stacks of the Gaussian model at ``rho`` (no buffer)."""
    vals = []
    for _ in range(reps):
        s = gauss_stacks(n, su.T, su.d, rho, g, su.dev).reshape(-1, su.d)
        a = draw_directions(su.M, su.d, g)
        ones = torch.ones(len(s), device=su.dev)
        vals.append(float(weighted_sigreg(s, ones, a)))
    return sum(vals) / reps


def pooled(d: int, F: int, REC: int, buffer: int, insert: int, dev: str) -> None:
    """``pooled`` of the module docstring: ``F`` frames of the stream, the
    last ``REC`` recorded."""
    su = Setup(d, None, dev)
    v, mods = su.v, su.mods
    B, T, M = su.B, su.T, su.M
    out: dict[str, Any] = {
        "d": d,
        "B": B,
        "T": T,
        "M": M,
        "cap": B * buffer,
        "insert": B * insert,
        "lam_conf": v.sigreg,
    }
    t0 = time.time()

    def pred_scale(
        tokens: torch.Tensor, y: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, float]:
        t = jepa_loss_packed(
            tokens,
            y,
            valid,
            torch.zeros(M, v.z_width, device=dev),
            None,
            mods.predictor,
            keeps=list(mods.segments[1:]),
            proj=mods.proj,
            lam=0.0,
            sg=False,
            sigreg_on="tokens",
            pred_loss=v.pred_loss,
            token_sigreg="raw",
        )
        target = tokens[:, :T][valid]
        return {
            "L_pred": float(t.pred),
            "zero_pred": float(target.pow(2).mean()),
            "within_var": float((target - target.mean(1, keepdim=True)).pow(2).mean()),
        }

    grid, keeps = su.grid, su.keeps
    futures = [m.future for m in v.context_masks]

    def encode(
        chunks: torch.Tensor, masks: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, y = pack_views(
            chunks,
            masks,
            keeps=keeps,
            patch=v.patch,
            tubelet=v.tubelet,
            futures=futures,
        )
        return mods.encoder(h, mods.segments), y

    # whitening of the stack means: a pre-pass over every 4th of the first frames
    s0, zl, tl = su.stream(B), [], []
    for f in range(256):
        x, ready, _ = s0.next()
        if f % 4 == 0:
            t = su.target_tokens(x, f)
            zl.append(t.mean(1)[ready.to(dev)])
            tl.append(t[ready.to(dev)].reshape(-1, d)[::7])
    zs, ts = torch.cat(zl), torch.cat(tl)
    mu_k, W_k = whitening(ts)
    mu_t, W_t = whitening(zs)
    print(json.dumps({"prepass": len(zs), "s": time.time() - t0}), flush=True)

    # ---- held-out Zurich val: the first stack of each of its 256 windows ----
    pm = PackedMeasures(su.conf)
    win, masks = pm.windows["val"], pm.masks["val"]
    g = torch.Generator(device=dev).manual_seed(1)
    vl, rows = [], []
    for i in range(0, len(win), B):
        idx = pm.order["val"][i : i + B]
        tokens, y = encode(
            win[idx, : v.frames].to(dev), [m[idx].to(dev) for m in masks]
        )
        valid = torch.ones(len(idx), dtype=torch.bool, device=dev)
        tgt = tokens[:, :T]
        vl.append(tgt)
        a = draw_directions(M, d, g)
        c_tok = tgt[:, P_FIX : P_FIX + 1].expand(-1, T, -1)
        c_g = torch.randn(len(idx), 1, d, generator=g, device=dev).expand(-1, T, -1)
        rows.append(
            {
                "S_h": su.S(tgt, valid, a),
                "S_hw": su.S((tgt - mu_k) @ W_k.T, valid, a),
                "S_c_tok": su.S(c_tok, valid, a),
                "S_c_gauss": su.S(c_g, valid, a),
                **pred_scale(tokens, y, valid),
            }
        )
    vt = torch.cat(vl)
    val: dict[str, float] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    # whitened stack means of val (the 'white' collapse, no buffer)
    zm = vt.mean(1)
    mu, W = whitening(zm)
    cw = ((zm - mu) @ W.T)[:, None].expand(-1, T, -1)
    sw = []
    for i in range(0, len(vt), B):
        a = draw_directions(M, d, g)
        sw.append(su.S(cw[i : i + B], torch.ones(len(cw[i : i + B]), device=dev), a))
    val["S_c_white"] = sum(sw) / len(sw)
    val |= rho_of(vt, draw_directions(256, d, g))
    val["S_gauss_model_at_rho"] = gauss_model_S(su, val["rho"], B, g)
    val["S_iid_gauss"] = gauss_model_S(su, 0.0, B, g)
    out["val"] = val
    print(json.dumps({"val": val, "s": time.time() - t0}), flush=True)

    kinds = ("h", "hw", "tok", "white")
    past = {k: su.past_tokens(buffer, insert) for k in kinds}
    gens = {k: torch.Generator(device=dev).manual_seed(7) for k in kinds}
    gd = torch.Generator(device=dev).manual_seed(5)  # directions
    gm = torch.Generator(device=dev).manual_seed(6)  # masks
    s1 = su.stream(B)
    rec: list[dict[str, float]] = []
    keep_tok = []
    for f in range(F):
        x, ready, _ = s1.next()
        valid = ready.to(dev)
        a = draw_directions(M, d, gd)
        for k in kinds:
            past[k].next_frame(a, gens[k])
        mk = [
            multiblock_masks(B, grid, m, keep=kk, generator=gm)
            for m, kk in zip(v.context_masks, keeps, strict=True)
        ]
        tokens, y = encode(x.to(dev), mk)
        tgt = tokens[:, :T]
        col = {
            "h": tgt,
            "hw": (tgt - mu_k) @ W_k.T,
            "tok": tgt[:, P_FIX : P_FIX + 1].expand(-1, T, -1),
            "white": ((tgt.mean(1) - mu_t) @ W_t.T)[:, None].expand(-1, T, -1),
        }
        if f >= F - REC and bool(valid.any()):
            r = {"count": float(past["h"].count)}
            for k in kinds:
                r[f"S_{k}"] = su.S(col[k], valid, a)
                r[f"S_{k}_buf"] = su.S(col[k], valid, a, past[k].sums)
            if f % 8 == 0:
                r |= pred_scale(tokens, y, valid)
                keep_tok.append(tgt[valid])
            rec.append(r)
        for k in kinds:
            past[k].keep(col[k], valid)
        if f % 128 == 0:
            print(json.dumps({"frame": f, "s": time.time() - t0}), flush=True)
    tr: dict[str, float] = {}
    for k in rec[0]:
        vals = [r[k] for r in rec if k in r]
        tr[k] = sum(vals) / len(vals)
        if k.startswith("S_"):
            tr[k + "_sd"] = float(torch.tensor(vals).std())
    kt = torch.cat(keep_tok)
    tr |= rho_of(kt, draw_directions(256, d, g))
    tr["S_gauss_model_at_rho"] = gauss_model_S(su, tr["rho"], B, g)
    # val stacks with the training buffer of the healthy tokens
    a = draw_directions(M, d, g)
    past["h"].next_frame(a, gens["h"])
    val_buf = [
        su.S(vt[i : i + B], torch.ones(B, device=dev), a, past["h"].sums)
        for i in range(0, len(vt), B)
    ]
    out["val"]["S_h_trainbuf"] = sum(val_buf) / len(val_buf)
    out["train"] = tr
    out["seconds"] = time.time() - t0
    print("RESULT " + json.dumps(out), flush=True)


def optimal_atoms(su: Setup, steps: int = 3000) -> tuple[torch.Tensor, float]:
    """512 atoms g* (T, d) that minimize the population Epps-Pulley distance
    D(g) = E_a int |phi_g - phi|^2 w (= sigreg(g) / T) by Adam on fresh
    directions each step; returns g* and D(g*) * T on 64 x 256 fresh
    directions."""
    T, d, M, dev = su.T, su.d, su.M, su.dev
    g = torch.Generator(device=dev).manual_seed(11)
    atoms = torch.randn(T, d, generator=g, device=dev).requires_grad_(True)
    opt = torch.optim.Adam([atoms], lr=0.01)
    w = torch.ones(1, device=dev)
    for i in range(steps):
        a = draw_directions(M, d, g)
        loss = weighted_sigreg(atoms, w.expand(T), a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == steps // 2:
            for q in opt.param_groups:
                q["lr"] = 0.003
    with torch.no_grad():
        vals = [
            float(weighted_sigreg(atoms, w.expand(T), draw_directions(M, d, g)))
            for _ in range(64)
        ]
    return atoms.detach(), sum(vals) / len(vals)


def per_stream(d: int, B: int, F: int, REC: int, dev: str) -> None:
    """``per-stream`` of the module docstring: ``F`` frames of ``B`` streams,
    the last ``REC`` recorded."""
    CAP, INS = 2048, 512  # per stream
    K = 48  # lags of the stack-sum correlation
    A_RHO = 256
    su = Setup(d, B, dev)
    assert su.B == B
    T, M = su.T, su.M
    assert T == 512 and INS <= T
    out: dict[str, Any] = {
        "d": d,
        "B": B,
        "T": T,
        "M": M,
        "cap_per_stream": CAP,
        "insert_per_stream": INS,
        "F": F,
        "REC": REC,
    }
    t0 = time.time()

    with torch.no_grad():
        # population moments: a B = 32 pre-pass, every 4th of the first 256 frames
        s0, zl, tl = su.stream(32), [], []
        for f in range(256):
            x, ready, _ = s0.next()
            if f % 4 == 0 and bool(ready.any()):
                t = su.target_tokens(x[ready], 0)
                zl.append(t.mean(1))
                tl.append(t.reshape(-1, d)[::7])
        zs, ts = torch.cat(zl), torch.cat(tl)
        mu_k, W_k = whitening(ts)
        mu_t, W_t = whitening(zs)
        out["prepass"] = {"stacks": len(zs), "tokens": len(ts)}
        print(json.dumps({"prepass": len(zs), "s": time.time() - t0}), flush=True)

    g_opt, S_popt_T = optimal_atoms(su)
    out["Popt_S_at_T"] = S_popt_T  # S of the 512 atoms once each (N = T)
    print(json.dumps({"Popt_S_at_T": S_popt_T, "s": time.time() - t0}), flush=True)

    with torch.no_grad():
        gP = torch.Generator(device=dev).manual_seed(12)
        g_iid = torch.randn(T, d, generator=gP, device=dev)
        kinds = ("h", "hw", "white", "P", "Popt", "I")
        past = {k: su.past_tokens(CAP, INS) for k in kinds}
        gens = {k: torch.Generator(device=dev).manual_seed(7) for k in kinds}
        gd = torch.Generator(device=dev).manual_seed(5)  # SIGReg directions
        gI = torch.Generator(device=dev).manual_seed(13)
        a_rho = draw_directions(A_RHO, d, torch.Generator(device=dev).manual_seed(3))
        rho_kinds = ("h", "hw", "white")
        # population moments of u = a^T s along a_rho, from the pre-pass
        pop = {
            "h": ts @ a_rho.T,
            "hw": ((ts - mu_k) @ W_k.T) @ a_rho.T,
            "white": ((zs - mu_t) @ W_t.T) @ a_rho.T,
        }
        pop_mom = {k: (u.mean(0), u.var(0)) for k, u in pop.items()}
        s1: dict[str, list[torch.Tensor]] = {k: [] for k in rho_kinds}  # (B, A)
        s2: dict[str, list[torch.Tensor]] = {k: [] for k in rho_kinds}  # (B, A)
        vflag = []  # (B,) valid
        stx = su.stream(B)
        rec: list[dict[str, float]] = []
        for f in range(F):
            x, ready, _ = stx.next()
            valid = ready.to(dev)
            a = draw_directions(M, d, gd)
            for k in kinds:
                past[k].next_frame(a, gens[k])
            tgt = su.target_tokens(x, 0)
            col = {
                "h": tgt,
                "hw": (tgt - mu_k) @ W_k.T,
                "white": ((tgt.mean(1) - mu_t) @ W_t.T)[:, None].expand(-1, T, -1),
                "P": g_iid[None].expand(B, -1, -1),
                "Popt": g_opt[None].expand(B, -1, -1),
                "I": torch.randn(B, T, d, generator=gI, device=dev),
            }
            if f >= F - REC - K:
                for k in rho_kinds:
                    u = col[k] @ a_rho.T  # (B, T, A)
                    s1[k].append(u.sum(1))
                    s2[k].append(u.pow(2).sum(1))
                vflag.append(valid.clone())
            if f >= F - REC and bool(valid.any()):
                r = {"count": float(past["h"].count), "valid": float(valid.sum())}
                for k in kinds:
                    r[f"S_{k}"] = su.S(col[k], valid, a)
                    r[f"S_{k}_buf"] = su.S(col[k], valid, a, past[k].sums)
                rec.append(r)
            for k in kinds:
                past[k].keep(col[k], valid)
            if f % 128 == 0:
                print(
                    json.dumps(
                        {
                            "frame": f,
                            "valid": int(valid.sum()),
                            "count": past["h"].count,
                            "s": time.time() - t0,
                        }
                    ),
                    flush=True,
                )
        tr: dict[str, float] = {}
        for k in rec[0]:
            vals = torch.tensor([r[k] for r in rec], dtype=torch.float64)
            tr[k] = float(vals.mean())
            if k.startswith("S_"):
                tr[k + "_sd"] = float(vals.std())
        out["S"] = tr

        # rho (within-stack) and the lag correlation of the stack sums
        V = torch.stack(vflag)  # (R + K, B)
        corr = {}
        for k in rho_kinds:
            S1, S2 = torch.stack(s1[k]), torch.stack(s2[k])  # (R + K, B, A)
            for mom in ("pop", "rec"):
                if mom == "pop":
                    mu, var = pop_mom[k]
                else:
                    m = V[K:]
                    n = m.sum() * T
                    mu = S1[K:][m].sum(0) / n
                    var = S2[K:][m].sum(0) / n - mu**2
                within, lags = stack_corr(S1, S2, V, mu, var, T, K)
                corr[f"{k}_{mom}"] = {
                    "rho": within,
                    "lag": lags,
                    "var_u": float(var.mean()),
                }
        out["corr"] = corr
        out["seconds"] = time.time() - t0
        print("RESULT " + json.dumps(out), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """``pooled`` or ``per-stream`` (module docstring)."""
    p = argparse.ArgumentParser(description="The lambda calibration of M17.")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pooled")
    a.add_argument("d", type=int)
    a.add_argument("frames", type=int, nargs="?", default=768)
    a.add_argument("record", type=int, nargs="?", default=256)
    a.add_argument("--buffer", type=int, default=256, help="rows per stream")
    a.add_argument("--insert", type=int, default=2, help="per stream and frame")
    b = sub.add_parser("per-stream")
    b.add_argument("d", type=int)
    b.add_argument("streams", type=int)
    b.add_argument("frames", type=int, nargs="?", default=768)
    b.add_argument("record", type=int, nargs="?", default=512)
    for q in (a, b):
        q.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    os.environ.setdefault("MUJOU_EXP_DATA", DATA)
    if args.cmd == "pooled":
        with torch.no_grad():
            pooled(
                args.d, args.frames, args.record, args.buffer, args.insert, args.device
            )
    else:
        per_stream(args.d, args.streams, args.frames, args.record, args.device)


if __name__ == "__main__":
    main()
