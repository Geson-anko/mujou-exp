import torch

from exp.video.vae.loss import ae_loss, batch_loss, stack_mse_sums, target


class TestTarget:
    def test_is_x_over_256(self):
        x = torch.tensor([0, 1, 128, 255], dtype=torch.uint8)
        assert torch.equal(target(x), torch.tensor([0.0, 1 / 256, 0.5, 255 / 256]))


class TestAELoss:
    def test_the_hand_value(self):
        """y - y_hat of the first sample (1, -2, 0.5, 0): 1/2 (1 + 4 + 0.25) =
        2.625; of the second (3, 0, 0, -1): 1/2 (9 + 1) = 5."""
        y = torch.tensor([[[1.0, 0.0], [0.5, 2.0]], [[3.0, 1.0], [1.0, 0.0]]])
        y_hat = torch.tensor([[[0.0, 2.0], [0.0, 2.0]], [[0.0, 1.0], [1.0, 1.0]]])
        assert torch.equal(ae_loss(y, y_hat), torch.tensor([2.625, 5.0]))

    def test_the_batch_loss_is_the_mean_over_the_valid(self):
        losses = torch.tensor([1.0, 2.0, 6.0])
        valid = torch.tensor([True, False, True])
        assert float(batch_loss(losses, valid)) == 3.5
        assert float(batch_loss(losses, torch.zeros(3, dtype=torch.bool))) == 0.0

    def test_a_linear_autoencoder_converges_to_the_pca_subspace(self):
        """Baldi & Hornik (1989): the minima of the squared error of a linear
        autoencoder of k < p units span the top k principal directions. Full
        batch gradient descent on the mean of ae_loss of centred Gaussian data
        (covariance eigenvalues 9, 4, 1 | 0.25, 0.1, 0.05, rotated) reaches
        the projection onto the top 3 eigenvectors of the sample covariance;
        the slowest direction contracts by about lr (1 - 0.25) per step."""
        g = torch.Generator().manual_seed(0)
        p, k, n = 6, 3, 4000
        q, _ = torch.linalg.qr(torch.randn(p, p, generator=g, dtype=torch.float64))
        scale = torch.tensor([9.0, 4.0, 1.0, 0.25, 0.1, 0.05], dtype=torch.float64)
        x = (torch.randn(n, p, generator=g, dtype=torch.float64) * scale.sqrt()) @ q.T
        x = x - x.mean(0)
        enc = (
            0.1 * torch.randn(k, p, generator=g, dtype=torch.float64)
        ).requires_grad_()
        dec = (
            0.1 * torch.randn(p, k, generator=g, dtype=torch.float64)
        ).requires_grad_()
        lr = 0.05
        for _ in range(4000):
            loss = ae_loss(x, x @ enc.T @ dec.T).mean()
            ge, gd = torch.autograd.grad(loss, [enc, dec])
            with torch.no_grad():
                enc -= lr * ge
                dec -= lr * gd
        _, vecs = torch.linalg.eigh(x.T @ x / n)
        top = vecs[:, -k:]
        u, _ = torch.linalg.qr(dec.detach())
        assert torch.allclose(u @ u.T, top @ top.T, atol=1e-8)


class TestStackMseSums:
    def test_the_sums_of_the_errors_and_of_the_mean_stacks(self):
        y = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
        y_hat = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        mean = torch.tensor([2.0, 1.0])
        got = stack_mse_sums(y, y_hat, mean)
        # errors: 0 + 1 + 4 + 0 = 5; from the mean: 1 + 1 + 1 + 1 = 4
        assert got.dtype == torch.float64 and got.tolist() == [5.0, 4.0]
