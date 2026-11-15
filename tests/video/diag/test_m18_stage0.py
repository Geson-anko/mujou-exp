import json
from dataclasses import replace

import pytest
import torch

from exp.video.compose import compose
from exp.video.data.frames import PooledStackedTourFrames
from exp.video.diag.m18_stage0 import (
    TOL,
    gram,
    lambda_max,
    pca_basis,
    projection_sums,
    run_hessian,
    run_pca,
)
from exp.video.measure.encoder_eval import stream_windows
from exp.video.programs.vision import DATA_ENV, tours_root
from exp.video.vae.measures import held_out_stacks
from exp.video.vae.program import ae_streams
from tests.video.packed import smooth_tours
from tests.video.vae.test_program import tiny_ae

CPU = torch.device("cpu")


def _rows(n: int = 50, p: int = 30, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.linspace(3.0, 0.1, p, dtype=torch.float64)
    return torch.randn(n, p, generator=g, dtype=torch.float64) * scale + 0.5


class TestPCA:
    def test_the_gram_matrix_by_blocks(self):
        x = _rows()
        m = x.mean(0)
        got = gram(lambda a, b: x[a:b], len(x), m, block=7, device=CPU)
        torch.testing.assert_close(got, (x - m) @ (x - m).T, rtol=0, atol=1e-12)

    def test_the_spectrum_and_basis_are_those_of_the_covariance(self):
        """The eigenvalues of (1/n) Y_c^T Y_c (descending, n - 1 nonzero) and
        the projector onto its top k eigenvectors."""
        x, k = _rows(), 5
        m = x.mean(0)
        evals, u = pca_basis(lambda a, b: x[a:b], len(x), m, k, block=8, device=CPU)
        cov = (x - m).T @ (x - m) / len(x)
        want, vecs = torch.linalg.eigh(cov)
        torch.testing.assert_close(evals[:30], want.flip(0), rtol=0, atol=1e-10)
        assert evals[30:].abs().max() < 1e-10
        top = vecs[:, -k:]
        torch.testing.assert_close(
            u.T @ u, torch.eye(k, dtype=u.dtype), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(u @ u.T, top @ top.T, atol=1e-10, rtol=0)

    def test_u8_rows_through_the_scale(self):
        g = torch.Generator().manual_seed(1)
        x = torch.randint(0, 256, (20, 12), generator=g, dtype=torch.uint8)
        y = x.double() / 256
        m = y.mean(0)
        a = gram(lambda i, j: x[i:j], 20, m, block=6, device=CPU, scale=1 / 256)
        torch.testing.assert_close(a, (y - m) @ (y - m).T, rtol=0, atol=1e-12)

    def test_the_projection_sums(self):
        x = _rows()
        m, k = x[:40].mean(0), 4
        _, u = pca_basis(lambda a, b: x[a:b], 40, m, k, block=40, device=CPU)
        held = x[40:]
        c = held - m
        r = c - c @ u @ u.T
        got = projection_sums(u, held, m, batch=3)
        assert got["stack_mse"] == pytest.approx(
            float((r**2).sum() / (c**2).sum()), rel=1e-12
        )
        torch.testing.assert_close(
            torch.tensor(got["coord_ms"], dtype=torch.float64), ((c @ u) ** 2).mean(0)
        )
        # each held-out c splits into its coordinates and the residual
        assert got["base_ms"] == pytest.approx(
            sum(got["coord_ms"]) + got["residual_ms"]
        )


def _quadratic(eigs: list[float], seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """The point t and the matrix A = Q diag(eigs) Q^T of L = 1/2 t^T A t (its
    Hessian A), Q a random rotation."""
    g = torch.Generator().manual_seed(seed)
    n = len(eigs)
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    a = q @ torch.diag(torch.tensor(eigs, dtype=torch.float64)) @ q.T
    t = torch.randn(n, generator=g, dtype=torch.float64, requires_grad=True)
    return t, a


class TestLambdaMax:
    def test_the_top_eigenvalue_of_a_quadratic(self):
        """L = 1/2 t^T A t has the Hessian A: eigenvalues 5, 2, 1, 0.5."""
        t, a = _quadratic([5.0, 2.0, 1.0, 0.5])
        g = torch.Generator().manual_seed(1)
        got = lambda_max(lambda: 0.5 * t @ a @ t, [t], tol=1e-12, iters=500, gen=g)
        assert got["converged"]
        assert got["rho"] == pytest.approx(5.0, rel=1e-9)
        assert got["lam"] == pytest.approx(5.0, rel=1e-9)

    def test_a_negative_dominant_eigenvalue_keeps_its_sign_in_lam(self):
        """Eigenvalues -5, 2, 1: rho = 5 and the Rayleigh quotient -5."""
        t, a = _quadratic([-5.0, 2.0, 1.0])
        g = torch.Generator().manual_seed(1)
        got = lambda_max(lambda: 0.5 * t @ a @ t, [t], tol=1e-12, iters=500, gen=g)
        assert got["rho"] == pytest.approx(5.0, rel=1e-9)
        assert got["lam"] == pytest.approx(-5.0, rel=1e-9)

    def test_rho_converges_when_plus_and_minus_rho_are_both_there(self):
        """Eigenvalues 5, -4.99, 1: v stays a mixture of the two top
        directions (the Rayleigh quotient need not converge), but ||Hv|| is
        the power iteration of H^2 (eigenvalues 25, 24.9, 1): within 4.99 ..
        5 once the third direction is gone (rate (1/5)^2), so the stop at
        TOL leaves rho in 5 (1 - TOL) .. 5."""
        t, a = _quadratic([5.0, -4.99, 1.0])
        g = torch.Generator().manual_seed(1)
        got = lambda_max(lambda: 0.5 * t @ a @ t, [t], tol=TOL, iters=500, gen=g)
        assert got["converged"]
        assert 5.0 * (1 - TOL) <= got["rho"] <= 5.0 * (1 + 1e-12)

    def test_the_norms_never_decrease_and_stay_below_rho(self):
        """||H v_{k+1}|| >= ||H v_k|| (Cauchy-Schwarz on v^T H^2 v) and
        ||H v|| <= rho for unit v."""
        t, a = _quadratic([3.0, -2.9, 2.0, -1.0, 0.1])
        g = torch.Generator().manual_seed(1)
        got = lambda_max(lambda: 0.5 * t @ a @ t, [t], tol=0.0, iters=60, gen=g)
        h = got["history"]
        assert len(h) == 60 and not got["converged"]
        assert all(b >= a_ * (1 - 1e-12) for a_, b in zip(h, h[1:], strict=False))
        assert max(h) <= 3.0 * (1 + 1e-12)

    def test_against_the_explicit_hessian_of_a_small_net(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(6, 3, generator=g, dtype=torch.float64)
        y = torch.randn(6, 2, generator=g, dtype=torch.float64)
        w1 = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
        w2 = torch.randn(2, 4, generator=g, dtype=torch.float64, requires_grad=True)

        def loss_of(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return 0.5 * ((torch.tanh(x @ a.T) @ b.T - y) ** 2).sum()

        h = torch.autograd.functional.hessian(loss_of, (w1, w2))
        flat = torch.cat(
            [
                torch.cat([h[i][j].reshape(w.numel(), -1) for j in range(2)], 1)
                for i, w in enumerate((w1, w2))
            ]
        )
        evals = torch.linalg.eigvalsh(flat)
        want = float(evals[evals.abs().argmax()])
        got = lambda_max(
            lambda: loss_of(w1, w2), [w1, w2], tol=1e-12, iters=2000, gen=g
        )
        assert got["rho"] == pytest.approx(abs(want), rel=1e-6)
        assert got["lam"] == pytest.approx(want, rel=1e-6)


class TestRuns:
    @pytest.fixture
    def conf(self, tmp_path, monkeypatch):
        root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        return compose([*tiny_ae(root.name), "data.tours.pool=6", "seed=3"])

    def test_pca_is_that_of_the_explicit_covariance_of_the_stacks(self, tmp_path, conf):
        """Against the PCA of the explicit P x P covariance of every training
        stack (stream_windows, stack k of a window its frames k .. k + N - 1,
        y = x / 256): sigma_0^2 its trace over P, and the stack_mse of the
        held-out stacks and the pool through its top-k projector."""
        out = tmp_path / "pca.json"
        k, windows = 8, 32
        r = run_pca(conf, out, k=k, block=64, windows=windows)
        assert json.loads(out.read_text())["n"] == r["n"]
        assert out.with_suffix(".png").is_file()
        w = stream_windows(conf, tours_root(conf.data.tours), windows=windows)
        n = conf.vision.frames
        steps = w.shape[1] - n + 1
        y = torch.cat([w[:, i : i + n] for i in range(steps)]).double() / 256
        m = y.mean(0)
        c = (y - m).flatten(1)
        cov = c.T @ c / len(c)
        evals, vecs = torch.linalg.eigh(cov)
        assert evals[-k] > 1.01 * evals[-k - 1]  # the top-k subspace is unique
        u = vecs[:, -k:]
        assert r["n"] == len(y)
        assert r["sigma0_sq"] == pytest.approx(float(evals.sum()) / c.shape[1])
        assert r["sigma0_sq"] == pytest.approx(r["trace_over_P"], rel=1e-9)
        sets = held_out_stacks(conf, tours_root(conf.data.tours))
        t = replace(conf.data.tours, sampling="pool")
        pooled = ae_streams(replace(conf, data=replace(conf.data, tours=t)))
        assert isinstance(pooled, PooledStackedTourFrames)
        sets["pool"] = pooled.stacks
        assert set(r["splits"]) == set(sets) and len(sets["pool"]) == 6
        for name, x in sets.items():
            d = (x.double() / 256 - m).flatten(1)
            res = d - d @ u @ u.T
            want = float((res**2).sum() / (d**2).sum())
            assert r["splits"][name]["stack_mse"] == pytest.approx(want, rel=1e-8)

    def test_hessian_rows_and_a_batch_below_its_members_mean_rho(self, tmp_path, conf):
        """The batch of every (windows / batch)-th stack: rho of the mean of
        their Hessians is at most the mean of their rho (the triangle
        inequality of the spectral norm), up to the tolerance left in each."""
        r = run_hessian(conf, tmp_path / "h.json", windows=4, batch=2, iters=500)
        rows = r["stacks"]
        assert [s["stack"] for s in rows] == [0, 1, 2, 3]
        assert all(s["converged"] for s in rows) and r["all_converged"]
        assert r["max_stack_rho"] == max(s["rho"] for s in rows)
        assert r["batch"]["members"] == [0, 2]
        mean = (rows[0]["rho"] + rows[2]["rho"]) / 2
        assert r["batch_members_mean_rho"] == pytest.approx(mean)
        assert r["batch"]["rho"] <= mean * (1 + 10 * TOL)
        with pytest.raises(ValueError, match="multiple"):
            run_hessian(conf, tmp_path / "h.json", windows=5, batch=2, iters=5)
