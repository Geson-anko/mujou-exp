import json

import pytest
import torch

from exp.video.compose import compose
from exp.video.diag.m18_stage0 import (
    gram,
    lambda_max,
    pca_basis,
    projection_sums,
    run_hessian,
    run_pca,
)
from exp.video.programs.vision import DATA_ENV
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


class TestLambdaMax:
    def test_the_top_eigenvalue_of_a_quadratic(self):
        """L = 1/2 t^T A t has the Hessian A: eigenvalues 5, 2, 1, 0.5."""
        g = torch.Generator().manual_seed(0)
        q, _ = torch.linalg.qr(torch.randn(4, 4, generator=g, dtype=torch.float64))
        a = (
            q
            @ torch.diag(torch.tensor([5.0, 2.0, 1.0, 0.5], dtype=torch.float64))
            @ q.T
        )
        t = torch.randn(4, generator=g, dtype=torch.float64, requires_grad=True)
        got = lambda_max(lambda: 0.5 * t @ a @ t, [t], tol=1e-12, iters=500, gen=g)
        assert got["converged"] and got["lam"] == pytest.approx(5.0, rel=1e-9)

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
        assert got["lam"] == pytest.approx(want, rel=1e-6)


class TestRuns:
    @pytest.fixture
    def conf(self, tmp_path, monkeypatch):
        root = smooth_tours(tmp_path / "data" / "d", {"A": 120, "B": 80, "H": 200})
        monkeypatch.setenv(DATA_ENV, str(root.parent))
        return compose([*tiny_ae(root.name), "data.tours.pool=6", "seed=3"])

    def test_pca_writes_the_yardsticks(self, tmp_path, conf):
        """sigma_0^2 is the trace of the covariance over P; the PCA of k
        components reconstructs held-out stacks no better than the mean
        stack can be beaten (stack_mse in [0, 1]) and the pool's draws."""
        out = tmp_path / "pca.json"
        r = run_pca(conf, out, k=8, block=64, windows=32)
        assert json.loads(out.read_text())["n"] == r["n"]
        assert r["sigma0_sq"] == pytest.approx(r["trace_over_P"], rel=1e-9)
        assert set(r["splits"]) == {"val", "test", "pool"}
        assert r["splits"]["pool"]["count"] == 6
        for s in r["splits"].values():
            assert 0 <= s["stack_mse"] < 1.5
        assert r["orthonormal_max_dev"] < 1e-8
        assert out.with_suffix(".png").is_file()

    def test_hessian_writes_lambda_max_of_each_stack_and_the_batch(
        self, tmp_path, conf
    ):
        r = run_hessian(conf, tmp_path / "h.json", stacks=2, iters=30)
        assert len(r["stacks"]) == 2 and r["max_stack_lam"] == max(
            s["lam"] for s in r["stacks"]
        )
        assert r["batch"]["iters"] >= 2
