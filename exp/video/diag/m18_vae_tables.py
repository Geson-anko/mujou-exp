r"""DIAGNOSIS ONLY (M18 stage 2): the tables of 「段 2：VAE の結果」 of
``exp/video/results/m18.md`` from the run directories pulled into ``./runs``
(``just fleet pull --batch ...``): the training runs of ``m18-vae-v1``
(``job.json``, ``status.json``, ``baselines.json``, ``evals.jsonl``,
``stream.jsonl``, ``timing.jsonl``), their posthoc runs of ``m18-vae-post-v1``
(``post/decode/metrics.jsonl``), the yardsticks of ``m18-vae-base-v1``
(``post/decode/baselines.jsonl`` and ``refs.jsonl``), the autoencoder of the
same setting (depth 4, RMSProp, $\eta$ = 1e-3: the posthoc ``AE_POST``) and
stage 1's untrained ViT (``AE_BASE``). Reads only; prints markdown, and with
``--json`` writes the values the tables are made of.

The criteria are those written before the runs (results/m18.md, 「段 2：VAE
（回す前に決めたこと）」), at t = 1800 s on val, with the roles of 「判定の役割
の変更」:

- U (the verdict): the posthoc ``recon_blur`` and ``dyn_1/copy_1`` of the
  code (the posterior mean $\mu$) both below those of the untrained $\mu$
  (``m18-vae-base-v1``'s ``code=init``);
- R (reference): the run's val ``bpd`` < ``R_BAR`` (the pPCA-1024);
- the overfit check (diagnosis): the pool arm's ``pool_bpd`` < ``POOL_BAR``
  (the pPCA-1024 of the same 256 stacks);
- a missing value is 不明, a diverged run fails its criteria
  (``m18_tables.below``).

The predictions (``predictions``): Q11 U holds; Q12 the stream's
``recon_blur`` is not below the autoencoder's; Q13 the stream's val ``bpd``
is above ``Q13_BAR`` (R fails; a diverged run holds it); Q14 it is below the mean stack's Gaussian
``COLLAPSE`` (no collapse); Q15 ``pool_bpd`` is not below ``POOL_BAR``. Q3
(the trap of the collapse) is a prediction of how stage 2 fails, judged only
if U fails; Q4 (the optimization of $\sigma$ is not slow) is not judged (the
pre-registration), and ``sigma_track`` gives its numbers.

``sigma_track``: the run's $\ln \sigma$ against the optimum of the stacks it
learns on, from ``stream.jsonl``. Its ``loss`` is the mean of $-\mathcal L$
over a second's frames and ``kl`` that of the KL, so that with the second's
last $\ln\sigma$ (``log_sigma``) the squared error a stack is
(``train_sigma_star_sq``, ``elbo.elbo_terms``)

$$\frac{\lVert y - g(z)\rVert^2}{P} = 2\sigma^2\Big(\frac{-\mathcal L -
\mathrm{KL}}{P} - \ln\sigma - \tfrac12\ln 2\pi\Big) = \sigma^{*2}_{\rm
train},$$

exact for a second whose $\sigma$ did not move. It does move within a
second: the largest change of $\ln\sigma$ from one second to the next in
the three runs is 0.13 (B = 1), 0.08 (B = 32) and 0.02 (the pool), not
bounded by $\eta$ a frame (a step of RMSProp is $\eta\, g/\sqrt{\hat v}$).

``ppca_kl``: the linear optimum's KL per latent (Lucas et al. 2019), the
yardstick of the KL quantiles. The ML pPCA of eigenvalues $\lambda_j$ and
$\sigma^2$ (``m18_stage0.ppca_bpd``'s, of the dequantized stacks) has, as
a VAE with the prior $N(0, I)$, the posterior $N(\mu_j, s_j^2)$ along its
$j$-th eigenvector with $s_j^2 = \sigma^2/\lambda_j$ and $\mu_j =
\sqrt{\lambda_j - \sigma^2}\, c_j/\lambda_j$ ($c_j$ the coordinate of
$y - m$), so that its KL averaged over a held-out split is

$$\tfrac12\big(E[\mu_j^2] + s_j^2 - 1 - \ln s_j^2\big), \quad E[\mu_j^2] =
\frac{\lambda_j - \sigma^2}{\lambda_j^2}\, E[c_j^2],$$

$E[c_j^2]$ the split's ``coord_ms`` plus the dequantization's $v$.

Run from the repository root: ``python -m exp.video.diag.m18_vae_tables
[--runs runs] [--json F]``.
"""

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from exp.lib.elbo import dequant_var, ppca_fit
from exp.video.diag.m17_tables import at, fmt, jl
from exp.video.diag.m18_tables import CKPTS, T_END, below, both, mark, train_rate
from exp.video.vae.measures import KL_QUANTILES

R_BAR = 6.887  # results/m18-stage0/ppca.json: val, k = 1024
POOL_BAR = 7.050  # the same, the pool's 256 stacks
COLLAPSE = 8.127  # the mean stack's Gaussian (k = 0), val
COLLAPSE_MARGIN = 0.002  # a collapsed model fitting sigma to val reaches 8.1252
Q13_BAR = 7.15  # 8 + log2(2 pi e 0.0180) / 2: the autoencoder's error alone
# the autoencoder of the same setting, val, 1800 s, as Q12 wrote it before the
# runs (rounded on purpose: AE_POST's 0.13221 is not the bar)
AE_RECON_BLUR = 0.132
T_SAME_STACKS = 60.0  # B = 32 at 60 s has seen 57600 stacks, B = 1 at 1800 s 54000
TIMES = (0.0, 60.0, *CKPTS)
TRAIN, POST, BASE = "m18-vae-v1", "m18-vae-post-v1", "m18-vae-base-v1"
AE_POST = "m18-dec-post-v1/m18-ae_t128-579a35ce-s100"  # depth 4, RMSProp, 1e-3
AE_BASE = "m18-ae-base-v1/m18-ae_t128-d7da0ae6-s100"
ARMS = (("stream", 1), ("stream", 32), ("pool", 1))
NAMES = {
    ("stream", 1): "ストリーム B = 1",
    ("stream", 32): "ストリーム B = 32",
    ("pool", 1): "過学習の確かめ",
}
LIKELIHOOD = ("bpd", "kl", "rate_bpd", "sigma", "sigma_star", "sigma_gap")
LIKELIHOOD += ("stack_mse", "pool_bpd", "pool_stack_mse")
KL_Q = ("kl_q0", "kl_q10", "kl_q25", "kl_q50", "kl_q75", "kl_q90", "kl_q100")
POSTHOC = ("recon", "recon_blur", "dyn_1/copy_1", "rank", "z_spread")
POSTHOC += ("own_recon", "own_recon_blur", "stack_mse")
SPLITS = ("val", "test")

Key = tuple[str, int]  # (sampling, B)


class Runs(NamedTuple):
    """The run directories of stage 2 by ``Key``, and the yardsticks'."""

    train: dict[Key, Path]
    post: dict[Key, Path]
    base: Path | None


def _jobs(root: Path, batch: str) -> list[tuple[Path, dict[str, Any]]]:
    d = root / batch
    runs = sorted(d.iterdir()) if d.exists() else []
    return [
        (r, json.loads((r / "job.json").read_text()))
        for r in runs
        if (r / "job.json").exists()
    ]


def load(root: Path) -> Runs:
    """The runs of the three batches under ``root``; a posthoc run is keyed by
    its parent's key (``job["after"]``)."""

    def key(job: dict[str, Any]) -> Key:
        c = job["config"]["data"]
        return c["tours"]["sampling"], c["streams"]

    train = {key(j): r for r, j in _jobs(root, TRAIN)}
    by_id = {r.name: k for k, r in train.items()}
    post = {
        by_id[j["after"][0]]: r
        for r, j in _jobs(root, POST)
        if j["after"] and j["after"][0] in by_id
    }
    base = [r for r, _ in _jobs(root, BASE)]
    return Runs(train, post, base[0] if base else None)


# --- the criteria and the predictions ------------------------------------------------


class Facts(NamedTuple):
    """A training run's values at 1800 s on val and its verdicts (module
    docstring); the posthoc's are of its code."""

    state: str
    bpd: float | None
    pool_bpd: float | None
    recon_blur: float | None
    dyn: float | None
    r: bool | None
    u: bool | None
    overfit: bool | None


def facts(runs: Runs, key: Key, init: dict[str, Any]) -> Facts:
    """The values and verdicts of the run ``key`` against the untrained $\\mu$'s
    line ``init``."""
    d = runs.train.get(key)
    if d is None:
        return Facts("無い", *(None,) * 7)
    state = json.loads((d / "status.json").read_text())["state"]
    dv = state == "diverged"
    ev = jl(d / "evals.jsonl")
    bpd, pool = at(ev, T_END, "val", "bpd"), at(ev, T_END, "val", "pool_bpd")
    p = runs.post.get(key)
    m = jl(p / "post/decode/metrics.jsonl") if p else []
    rb, dyn = at(m, T_END, "val", "recon_blur"), at(m, T_END, "val", "dyn_1/copy_1")
    u = (
        both(
            below(rb, init.get("recon_blur"), diverged=dv),
            below(dyn, init.get("dyn_1/copy_1"), diverged=dv),
        )
        if p
        else None
    )
    overfit = below(pool, POOL_BAR, diverged=dv) if key[0] == "pool" else None
    return Facts(state, bpd, pool, rb, dyn, below(bpd, R_BAR, diverged=dv), u, overfit)


def _not(v: bool | None) -> bool | None:
    return None if v is None else not v


def _above(x: float | None, bar: float) -> bool | None:
    return None if x is None else x > bar


def predictions(stream: Facts, pool: Facts) -> dict[str, str]:
    """The verdict of each prediction (module docstring) of the stream's and
    the pool's ``Facts``: ○ (held), × (did not), 判定不能 (a value is
    missing), or why it is not judged."""

    def judge(v: bool | None) -> str:
        return "判定不能" if v is None else ("○" if v else "×")

    dv = stream.state == "diverged"
    return {
        "Q11": judge(stream.u),
        "Q12": judge(_not(below(stream.recon_blur, AE_RECON_BLUR, diverged=False))),
        "Q13": judge(True if dv else _above(stream.bpd, Q13_BAR)),
        "Q14": judge(below(stream.bpd, COLLAPSE, diverged=dv)),
        "Q15": judge(_not(below(pool.pool_bpd, POOL_BAR, diverged=False))),
        "Q3": "判定しない（U を満たした）"
        if stream.u is True
        else judge(None)
        if stream.u is None
        else "読みで判定（U を外した）",
        "Q4": "判定しない（回す前に決めた）",
    }


# --- the KL of the linear optimum ---------------------------------------------------


def ppca_kl(pca: dict[str, Any], split: str) -> Tensor:
    """The KL per latent of the ML pPCA of ``pca`` (the output of
    ``m18_stage0 pca``, its ``k`` components) averaged over the held-out
    stacks of ``split`` (module docstring), ``(k,)`` float64 nats."""
    p, k = int(pca["P"]), int(pca["k"])
    v = dequant_var(int(pca["levels"]))
    evals = torch.tensor(pca["eigenvalues"], dtype=torch.float64)
    lam, sigma_sq = ppca_fit(evals, k, p, floor=v)
    c_sq = torch.tensor(pca["splits"][split]["coord_ms"][:k], dtype=torch.float64)
    s_sq = sigma_sq / lam
    mu_sq = (lam - sigma_sq) / lam.square() * (c_sq + v)
    return 0.5 * (mu_sq + s_sq - 1 - s_sq.log())


def stack_values(job: dict[str, Any]) -> int:
    """$P$, the values of a stack of the run of ``job.json`` ``job``."""
    c = job["config"]
    return int(c["vision"]["frames"]) * 3 * int(c["data"]["tours"]["size"]) ** 2


# --- sigma -----------------------------------------------------------------------------


def train_sigma_star_sq(loss: float, kl: float, log_sigma: float, p: int) -> float:
    """$\\sigma^{*2}_{\\rm train}$ (module docstring) of a second's mean
    $-\\mathcal L$ ``loss`` and KL ``kl`` (nats a stack) at $\\ln\\sigma$
    ``log_sigma``, of stacks of ``p`` values."""
    s2 = math.exp(2 * log_sigma)
    return 2 * s2 * ((loss - kl) / p - log_sigma - 0.5 * math.log(2 * math.pi))


def sigma_track(
    stream: Sequence[dict[str, Any]], p: int, *, after: float = 60.0
) -> dict[str, Any]:
    """Of the rows of ``stream.jsonl``: the first second whose $\\ln\\sigma$ is
    at or above $\\ln\\sigma^*_{\\rm train}$ (``first_t``: $\\sigma$ starts
    below it, at $\\sigma_0$ under the untrained decoder's error), and over the
    seconds after ``after`` the mean and the mean absolute value of
    $\\ln\\sigma - \\ln\\sigma^*_{\\rm train}$ and the fraction of them above
    0 (``above``); None without such seconds."""
    pts = []
    for r in stream:
        if r.get("loss") is None or r.get("log_sigma") is None:
            continue
        s2 = train_sigma_star_sq(r["loss"], r["kl"], r["log_sigma"], p)
        pts.append((r["t"], r["log_sigma"] - 0.5 * math.log(s2)))
    first = next((t for t, g in pts if g >= 0), None)
    late = [g for t, g in pts if t > after]
    if not late:
        return {"first_t": first, "mean": None, "mean_abs": None, "above": None}
    n = len(late)
    return {
        "first_t": first,
        "mean": sum(late) / n,
        "mean_abs": sum(abs(g) for g in late) / n,
        "above": sum(g > 0 for g in late) / n,
        "seconds": n,
    }


# --- the tables ------------------------------------------------------------------------


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return "\n".join([_row(head), _row(["--"] * len(head)), *map(_row, rows)])


def _line(path: Path, split: str, code: str) -> dict[str, Any]:
    for r in jl(path):
        if r.get("split") == split and r.get("code") == code:
            return r
    return {}


def _value(d: Path | None, t: float, split: str, key: str) -> Any:
    """``key`` of the run ``d`` at ``t`` (time 0: ``baselines.json``)."""
    if d is None:
        return None
    if t == 0.0:
        b = d / "baselines.json"
        return json.loads(b.read_text())[split].get(key) if b.exists() else None
    return at(jl(d / "evals.jsonl"), t, split, key)


def report(
    root: Path, last: Path | None = None, pca_json: Path | None = None
) -> tuple[str, dict[str, Any]]:
    """The markdown of the tables (module docstring), and their values;
    ``last`` the output of ``m18_stage0 last``, ``pca_json`` that of
    ``m18_stage0 pca`` (the KL quantiles' pPCA row)."""
    runs = load(root)
    base = runs.base / "post/decode" if runs.base else None
    data: dict[str, Any] = {}
    out = ["## 実行の確かめ", ""]
    rows = []
    for k in ARMS:
        for kind, d in (("学習", runs.train.get(k)), ("posthoc", runs.post.get(k))):
            if d is None:
                continue
            j = json.loads((d / "job.json").read_text())
            s = json.loads((d / "status.json").read_text())
            c = j["code"]
            rows.append(
                [NAMES[k], kind, d.name, s["state"], f"{s['t']:g} / {s['of']:g}"]
                + [c["snap"][:12], c["mujou_snap"][:7], str(c["dirty"])]
            )
    if runs.base:
        j = json.loads((runs.base / "job.json").read_text())
        s = json.loads((runs.base / "status.json").read_text())
        rows.append(
            [
                "物差し",
                "baselines",
                runs.base.name,
                s["state"],
                f"{s['t']:g} / {s['of']:g}",
            ]
            + [j["code"]["snap"][:12], j["code"]["mujou_snap"][:7]]
            + [str(j["code"]["dirty"])]
        )
    head = ["腕", "種類", "run", "状態", "t / of", "snapshot", "mujou", "dirty"]
    out.append(_table(head, rows))

    # U and its yardsticks
    lasts = json.loads(last.read_text())["splits"] if last and last.exists() else {}
    for split in SPLITS:
        init = _line(base / "baselines.jsonl", split, "init") if base else {}
        ae_init = _line(root / AE_BASE / "post/decode/baselines.jsonl", split, "init")
        pca = _line(base / "baselines.jsonl", split, "pca") if base else {}
        tube = _line(base / "refs.jsonl", split, "tube") if base else {}
        ae_m = jl(root / AE_POST / "post/decode/metrics.jsonl")
        rows, vals = [], {}
        for k in ARMS:
            p = runs.post.get(k)
            m = jl(p / "post/decode/metrics.jsonl") if p else []
            got = {key: at(m, T_END, split, key) for key in POSTHOC}
            dv = runs.train.get(k) is not None and (
                json.loads((runs.train[k] / "status.json").read_text())["state"]
                == "diverged"
            )
            u = (
                both(
                    below(got["recon_blur"], init.get("recon_blur"), diverged=dv),
                    below(got["dyn_1/copy_1"], init.get("dyn_1/copy_1"), diverged=dv),
                )
                if p
                else None
            )
            vals[NAMES[k]] = got | {"U": u}
            rows.append(
                [NAMES[k]]
                + [fmt(got[x]) for x in ("recon", "recon_blur", "dyn_1/copy_1", "rank")]
                + [mark(u) if p else "－（posthoc なし）"]
            )

        def ref(name: str, r: dict[str, Any], blur: str = "recon_blur") -> list[str]:
            keys = ("recon", blur, "dyn_1/copy_1", "rank")
            vals[name] = {x: r.get(x) for x in keys}
            return [name, *(fmt(r.get(x)) for x in keys), "－"]

        rows.append(ref("学習前の μ（m18-vae-base-v1）", init))
        rows.append(ref("学習前の ViT（段 1、m18-ae-base-v1）", ae_init))
        rows.append(ref("画素の PCA の符号（後から学ぶデコーダ）", pca))
        lin = {"recon": pca.get("recon_lin"), "recon_blur": pca.get("recon_lin_blur")}
        rows.append(ref("最後のフレームの PCA-1024（線形の再構成）", lin))
        sl = lasts.get(split, {})
        rows.append(
            ref("積みの PCA-1024 の最後のフレーム（線形、回した後に測った）", sl)
        )
        rows.append(ref("tube（平均の色）", tube))
        aer = {x: at(ae_m, T_END, split, x) for x in POSTHOC}
        rows.append(ref("自己符号化器（深さ 4、η = 1e-3）", aer))
        data[f"U_{split}"] = vals
        out += ["", f"## U と物差し（{split}、1800 秒、posthoc）", ""]
        out.append(
            _table(["符号", "recon", "recon_blur", "dyn_1/copy_1", "rank", "U"], rows)
        )

    # likelihood over time
    out += [
        "",
        "## 尤度の経過（val、run の evals.jsonl、時刻 0 は baselines.json）",
        "",
    ]
    rows = []
    for k in ARMS:
        d = runs.train.get(k)
        for t in TIMES:
            v = {x: _value(d, t, "val", x) for x in LIKELIHOOD}
            data.setdefault("likelihood", {}).setdefault(NAMES[k], {})[str(t)] = v
            ratio = (
                v["sigma"] / v["sigma_star"]
                if v["sigma"] is not None and v["sigma_star"]
                else None
            )
            rows.append(
                [NAMES[k], f"{t:.0f}", *(fmt(v[x]) for x in LIKELIHOOD), fmt(ratio)]
            )
    out.append(_table(["腕", "t", *LIKELIHOOD, "sigma/sigma_star"], rows))

    # KL over the latents
    out += ["", "## 次元ごとの KL の分位点（val、nats、1024 個の潜在）", ""]
    rows = []
    for k in ARMS:
        d = runs.train.get(k)
        for t in (0.0, 60.0, 300.0, T_END):
            v = [_value(d, t, "val", x) for x in KL_Q]
            data.setdefault("kl_q", {}).setdefault(NAMES[k], {})[str(t)] = v
            rows.append([NAMES[k], f"{t:.0f}", *map(fmt, v)])
    if pca_json and pca_json.exists():
        kl = ppca_kl(json.loads(pca_json.read_text()), "val")
        q = torch.tensor(KL_QUANTILES, dtype=torch.float64) / 100
        v = kl.quantile(q).tolist()
        data["kl_q_ppca"] = {"val": v, "total": float(kl.sum())}
        rows.append(["pPCA-1024（線形の最適解）", "－", *map(fmt, v)])
    out.append(_table(["腕", "t", *KL_Q], rows))

    # verdicts and predictions
    init = _line(base / "baselines.jsonl", "val", "init") if base else {}
    f = {k: facts(runs, k, init) for k in ARMS}
    out += ["", "## 判定（val、1800 秒）", ""]
    rows = []
    for k in ARMS:
        x = f[k]
        u = mark(x.u) if runs.post.get(k) else "－（posthoc なし）"
        rows.append(
            [NAMES[k], x.state, fmt(x.recon_blur), fmt(x.dyn), u]
            + [fmt(x.bpd), mark(x.r), fmt(x.pool_bpd)]
            + [mark(x.overfit) if k[0] == "pool" else "－"]
            + [mark(below(x.bpd, COLLAPSE, diverged=x.state == "diverged"))]
        )
    head = ["腕", "状態", "recon_blur", "dyn_1/copy_1", "U", "bpd", "R"]
    head += ["pool_bpd", "過学習", f"bpd < {COLLAPSE}"]
    out.append(_table(head, rows))
    pred = predictions(f[("stream", 1)], f[("pool", 1)])
    pred32 = predictions(f[("stream", 32)], f[("pool", 1)])
    data["facts"] = {NAMES[k]: f[k]._asdict() for k in ARMS}
    data["predictions"] = {"B=1": pred, "B=32": pred32}
    out += ["", "## 予測（ストリーム B = 1。括弧は同じ規則を B = 32 に当てた値）", ""]
    out.append(
        _table(["予測", "判定"], [[q, f"{v}（{pred32[q]}）"] for q, v in pred.items()])
    )

    # sigma against the stacks it learns on
    out += ["", "## σ と学ぶ積みの σ*（stream.jsonl、60 秒より後）", ""]
    rows = []
    for k in ARMS:
        d = runs.train.get(k)
        tr = (
            sigma_track(
                jl(d / "stream.jsonl"),
                stack_values(json.loads((d / "job.json").read_text())),
            )
            if d
            else {}
        )
        data.setdefault("sigma_track", {})[NAMES[k]] = tr
        rows.append(
            [NAMES[k], fmt(tr.get("first_t")), fmt(tr.get("mean"))]
            + [fmt(tr.get("mean_abs")), fmt(tr.get("above"))]
        )
    out.append(
        _table(
            [
                "腕",
                "最初に σ ≥ σ* の秒",
                "平均の ln σ/σ*",
                "ln σ/σ* の絶対値の平均",
                "σ > σ* の割合",
            ],
            rows,
        )
    )

    # B = 32 against B = 1
    b1, b32 = runs.train.get(("stream", 1)), runs.train.get(("stream", 32))
    same_t = (_value(b1, T_END, "val", "bpd"), _value(b32, T_END, "val", "bpd"))
    same_n = (_value(b1, T_END, "val", "bpd"), _value(b32, T_SAME_STACKS, "val", "bpd"))
    data["b32"] = {"same_steps": same_t, "same_stacks": same_n}
    out += ["", "## B = 32 と B = 1（val の bpd）", ""]
    out.append(
        _table(
            ["比べ方", "B = 1", "B = 32"],
            [
                ["同じ一歩の数（1800 秒）", fmt(same_t[0]), fmt(same_t[1])],
                [
                    "同じ積みの数（B = 1 の 1800 秒、B = 32 の 60 秒）",
                    *map(fmt, same_n),
                ],
            ],
        )
    )

    # speed
    out += ["", "## 速さ（timing.jsonl、評価を除く）", ""]
    rows = []
    for k in ARMS:
        d = runs.train.get(k)
        rate = train_rate(jl(d / "timing.jsonl")) if d else None
        target = 10 / k[1]
        data.setdefault("speed", {})[NAMES[k]] = {"rate": rate, "target": target}
        rows.append(
            [
                NAMES[k],
                fmt(rate),
                fmt(target),
                fmt(None if rate is None else rate / target),
            ]
        )
    out.append(_table(["腕", "単位/秒", "目標（10/B）", "目標との比"], rows))

    # the posthoc per checkpoint
    out += ["", "## checkpoint ごとの posthoc（val）", ""]
    rows = []
    for k in ARMS:
        p = runs.post.get(k)
        if p is None:
            continue
        m = jl(p / "post/decode/metrics.jsonl")
        for t in CKPTS:
            v = [at(m, t, "val", x) for x in POSTHOC]
            data.setdefault("posthoc", {}).setdefault(NAMES[k], {})[str(t)] = dict(
                zip(POSTHOC, v, strict=True)
            )
            rows.append([NAMES[k], f"{t:.0f}", *map(fmt, v)])
    ae_name = "自己符号化器（深さ 4、η = 1e-3）"
    ae_m = jl(root / AE_POST / "post/decode/metrics.jsonl")
    for t in CKPTS if ae_m else ():
        v = [at(ae_m, t, "val", x) for x in POSTHOC]
        data.setdefault("posthoc", {}).setdefault(ae_name, {})[str(t)] = dict(
            zip(POSTHOC, v, strict=True)
        )
        rows.append([ae_name, f"{t:.0f}", *map(fmt, v)])
    out.append(_table(["腕", "t", *POSTHOC], rows))
    return "\n".join(out), data


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument(
        "--last",
        type=Path,
        default=Path("exp/video/results/m18-stage2/last.json"),
        help="the output of m18_stage0 last",
    )
    p.add_argument(
        "--pca",
        type=Path,
        default=Path("exp/video/results/m18-stage0/pca.json"),
        help="the output of m18_stage0 pca",
    )
    p.add_argument("--json", type=Path, help="write the values of the tables here")
    a = p.parse_args(argv)
    text, data = report(a.runs, a.last, a.pca)
    print(text)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
