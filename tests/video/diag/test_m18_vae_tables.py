"""The tables of results/m18.md's stage 2 (diagnosis): the squared error of the
stacks the run learns on, recovered from its logged negative ELBO; the
verdicts of the criteria and of the predictions written before the runs."""

import json
import math
from pathlib import Path

import pytest
import torch

from exp.lib.elbo import gauss_nll
from exp.video.diag.m18_vae_tables import (
    AE_RECON_BLUR,
    COLLAPSE,
    POOL_BAR,
    Q13_BAR,
    R_BAR,
    Facts,
    facts,
    load,
    predictions,
    sigma_track,
    train_sigma_star_sq,
)


class TestSigma:
    def test_the_squared_error_from_the_negative_elbo(self):
        """-L of exp.lib.elbo's likelihood plus a KL gives back ||y - g||^2 / P."""
        g = torch.Generator().manual_seed(0)
        y = torch.rand(1, 3, 4, 5, generator=g, dtype=torch.float64)
        y_hat = y + 0.1 * torch.randn(y.shape, generator=g, dtype=torch.float64)
        log_sigma, kl = torch.tensor(-1.7, dtype=torch.float64), 37.5
        loss = float(gauss_nll(y, y_hat, log_sigma)[0]) + kl
        want = float((y - y_hat).square().sum()) / y.numel()
        got = train_sigma_star_sq(loss, kl, float(log_sigma), y.numel())
        assert got == pytest.approx(want, rel=1e-12)

    def test_sigma_track_of_a_sigma_that_catches_up(self):
        """sigma starts below the optimum, meets it at 3 s and then sits a
        known factor above and below it in turn."""
        p, kl = 1000, 5.0

        def row(t: float, log_sigma: float, star_sq: float) -> dict:
            s2 = math.exp(2 * log_sigma)
            loss = p * star_sq / (2 * s2) + p * (
                log_sigma + 0.5 * math.log(2 * math.pi)
            )
            return {"t": t, "loss": loss + kl, "kl": kl, "log_sigma": log_sigma}

        star = 0.01  # ln sigma* = ln 0.1
        rows = [row(t, math.log(0.1) - 0.5 + 0.2 * t, star) for t in (1.0, 2.0)]
        rows.append(row(3.0, math.log(0.1), star))
        rows += [
            row(60.0 + t, math.log(0.1) + (0.1 if t % 2 else -0.3), star)
            for t in range(1, 5)
        ]
        rows.append({"t": 70.0, "loss": None, "kl": None, "log_sigma": None})
        got = sigma_track(rows, p)
        assert got["first_t"] == 3.0
        assert got["seconds"] == 4
        assert got["mean"] == pytest.approx(-0.1)
        assert got["mean_abs"] == pytest.approx(0.2)
        assert got["above"] == pytest.approx(0.5)


def _run(root: Path, batch: str, rid: str, sampling: str, b: int, **kw) -> Path:
    d = root / batch / rid
    (d / "post/decode").mkdir(parents=True)
    config = {"data": {"streams": b, "tours": {"sampling": sampling}}}
    job = {"run_id": rid, "config": config, "after": kw.get("after", [])}
    (d / "job.json").write_text(json.dumps(job))
    (d / "status.json").write_text(json.dumps({"state": kw.get("state", "completed")}))
    return d


def _lines(path: Path, *rows: dict) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestVerdicts:
    def test_facts_of_the_runs_and_their_posthoc(self, tmp_path: Path):
        st = _run(tmp_path, "m18-vae-v1", "a-s100", "stream", 1)
        _lines(st / "evals.jsonl", {"t": 1800.0, "split": "val", "bpd": R_BAR - 0.1})
        po = _run(tmp_path, "m18-vae-post-v1", "b-s100", "stream", 1, after=[st.name])
        row = {"t": 1800.0, "split": "val", "recon_blur": 0.1, "dyn_1/copy_1": 0.5}
        _lines(po / "post/decode/metrics.jsonl", row)
        pl = _run(tmp_path, "m18-vae-v1", "c-s100", "pool", 1, state="diverged")
        row = {"t": 1800.0, "split": "val", "bpd": 1.0, "pool_bpd": 1.0}
        _lines(pl / "evals.jsonl", row)
        runs = load(tmp_path)
        assert runs.post[("stream", 1)] == po
        init = {"recon_blur": 0.3, "dyn_1/copy_1": 0.6}
        f = facts(runs, ("stream", 1), init)
        assert (f.u, f.r, f.overfit) == (True, True, None)
        assert facts(runs, ("stream", 1), {"dyn_1/copy_1": 0.6}).u is None
        g = facts(runs, ("pool", 1), init)
        # no posthoc: U unknown; diverged: R and the overfit check fail
        assert (g.u, g.r, g.overfit) == (None, False, False)
        assert facts(runs, ("stream", 32), init).state == "無い"

    def test_predictions_at_their_bars(self):
        def stream(
            rb: float | None,
            bpd: float | None,
            u: bool | None = True,
            state: str = "completed",
        ) -> Facts:
            return Facts(state, bpd, None, rb, None, None, u, None)

        def pool(pb: float | None) -> Facts:
            return Facts("completed", None, pb, None, None, None, None, None)

        # at the autoencoder's recon_blur exactly, the VAE is "not below" it
        got = predictions(stream(AE_RECON_BLUR, Q13_BAR + 0.01), pool(POOL_BAR))
        assert (got["Q11"], got["Q12"], got["Q13"], got["Q14"], got["Q15"]) == (
            "○",
            "○",
            "○",
            "○",
            "○",
        )
        assert got["Q3"].startswith("判定しない")
        got = predictions(stream(AE_RECON_BLUR - 0.01, Q13_BAR), pool(POOL_BAR - 0.01))
        assert (got["Q12"], got["Q13"], got["Q15"]) == ("×", "×", "×")
        got = predictions(stream(None, COLLAPSE, u=False), pool(None))
        assert (got["Q11"], got["Q12"], got["Q14"], got["Q15"]) == (
            "×",
            "判定不能",
            "×",
            "判定不能",
        )
        assert got["Q3"] == "読みで判定（U を外した）"
        got = predictions(stream(None, None, u=None, state="diverged"), pool(None))
        assert (got["Q13"], got["Q14"]) == ("○", "×")  # a diverged run fails R
