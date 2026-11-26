r"""``exp.lib.fidelity`` (the observation's content a code keeps).

The oracles: the pooled observation is each cell's mean by hand; $E$ and
$E^d$ are 1 and 1 of the observation itself and 0 and 0 of a constant
readout at the anchors' mean, and by hand of a few numbers; proposition 1
in closed form: of an orthogonal design $y = x$, the least squares from the
code with directions dropped explains exactly the kept variances' share, so
$E$ falls with each dropped direction; the Gaussian shrinkage $E[f \mid f +
n] = f / (1 + \sigma^2)$ in the large-sample limit of the least squares
(the conditional mean is linear); the ratio of a code with itself is 0 with
no spread, and its interval $\pm 2\sigma_{\mathrm{seed}}$; a readout
explaining nothing is $-\infty$ and dropped; the seeds' deviation by hand;
the front's verdicts by hand; an MLP of one seed the same twice and fitting
a linear map.
"""

import math

import pytest
import torch

from exp.lib.fidelity import (
    ContentSums,
    MLPConf,
    comparable,
    content_sums,
    fit_linear,
    fit_mlp,
    front_verdict,
    pool_obs,
    predict,
    predict_linear,
    ratio_interval,
    seed_sigma,
)
from exp.lib.metrics import block_counts

CPU = torch.device("cpu")
F64 = torch.float64


def _counts(
    nb: int, seed: int = 101, n: int = 400
) -> tuple[torch.Tensor, torch.Tensor]:
    return (block_counts(nb, n, torch.Generator().manual_seed(seed)),
            block_counts(nb // 2, n, torch.Generator().manual_seed(seed)))  # fmt: skip


class TestObservation:
    def test_each_cell_is_its_mean(self):
        frames = torch.arange(2 * 3 * 4 * 4, dtype=torch.uint8).reshape(2, 3, 4, 4)
        got = pool_obs(frames, 2)
        f = frames.double() / 255
        want = f.reshape(2, 3, 2, 2, 2, 2).mean((3, 5)).flatten(1)
        assert got.shape == (2, 12)
        assert torch.allclose(got.double(), want, atol=1e-6)


def _series(n: int = 60, k: int = 3) -> torch.Tensor:
    return torch.randn(n, k, generator=torch.Generator().manual_seed(0), dtype=F64)


class TestQuantities:
    def test_the_observation_itself_and_a_constant(self):
        y = _series()
        t = torch.arange(1, 60)
        blocks = t // 20
        mean = y[t].mean(0)
        cs = content_sums(y, [y, mean.expand_as(y)], mean, t, blocks, 3)
        e, ed = cs.per_seed(), cs.per_seed(change=True)
        assert e.tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
        assert ed.tolist() == pytest.approx([1.0, 0.0], abs=1e-12)
        # the value is the mean over the seeds (one denominator)
        assert cs.value() == pytest.approx(0.5, abs=1e-12)

    def test_by_hand(self):
        y = torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=F64)
        p = torch.tensor([[0.0], [2.0], [2.0], [6.0]], dtype=F64)
        t = torch.tensor([1, 2, 3])
        cs = content_sums(
            y,
            [p],
            torch.tensor([2.0], dtype=F64),
            t,
            torch.zeros(3, dtype=torch.long),
            1,
        )
        # err: 1 + 1 + 0; tot: 1 + 1 + 16; d: 1, 2, 3; dhat: 2, 0, 4
        assert cs.err.tolist() == [[2.0]] and cs.tot.tolist() == [18.0]
        assert cs.derr.tolist() == [[1.0 + 4.0 + 1.0]] and cs.dtot.tolist() == [14.0]
        assert cs.value() == pytest.approx(1 - 2 / 18)
        assert cs.value(True) == pytest.approx(1 - 6 / 14)

    def test_proposition_1_closed_form(self):
        # an orthogonal design of variances v: y = x, the code x P (some
        # directions dropped); the least squares in sample explains the kept
        # variances exactly, so E falls with each dropped direction
        g = torch.Generator().manual_seed(1)
        n, d = 400, 5
        # columns orthogonal to each other and to the constant (centred)
        ones = torch.ones(n, 1, dtype=F64) / math.sqrt(n)
        q, _ = torch.linalg.qr(
            torch.cat([ones, torch.randn(n, d, generator=g, dtype=F64)], 1)
        )
        x = q[:, 1:] * torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], dtype=F64)
        v = x.square().sum(0)
        es = []
        for r in range(d):
            keep = torch.ones(d, dtype=F64)
            keep[:r] = 0  # the largest r directions dropped
            xc = x * keep
            b = fit_linear(xc, x)
            cs = content_sums(x, [predict_linear(b, xc).double()], x.mean(0),
                              torch.arange(n), torch.zeros(n, dtype=torch.long), 1)  # fmt: skip
            want = float(v[r:].sum() / v.sum())
            assert cs.value() == pytest.approx(want, abs=1e-9)
            es.append(cs.value())
        assert all(b < a for a, b in zip(es, es[1:], strict=False))

    def test_gaussian_shrinkage(self):
        # E[f | f + n] = f / (1 + s2): the share explained of a unit f is 1/(1+s2)
        g = torch.Generator().manual_seed(2)
        n, s2 = 200000, 0.5
        f = torch.randn(n, 1, generator=g, dtype=F64)
        c = f + math.sqrt(s2) * torch.randn(n, 1, generator=g, dtype=F64)
        b = fit_linear(c, f)
        cs = content_sums(f, [predict_linear(b, c).double()], f.mean(0), torch.arange(1, n),
                          torch.zeros(n - 1, dtype=torch.long), 1)  # fmt: skip
        assert cs.value() == pytest.approx(1 / (1 + s2), abs=5e-3)


def _sums(errs: list[list[float]], tot: list[float]) -> ContentSums:
    e = torch.tensor(errs, dtype=F64)
    t = torch.tensor(tot, dtype=F64)
    return ContentSums(e, t, e.clone(), t.clone())


class TestRatio:
    def test_a_code_with_itself(self):
        g = torch.Generator().manual_seed(3)
        tot = (1 + torch.rand(8, generator=g, dtype=F64)).tolist()
        errs = [[0.3 * x for x in tot], [0.31 * x for x in tot]]
        a = _sums(errs, tot)
        c, c2 = _counts(8)
        got = ratio_interval(a, a, c, c2, 0.0)
        assert got["point"] == 0.0 and got["se"] == pytest.approx(0.0, abs=1e-15)
        got = ratio_interval(a, a, c, c2, 0.05)
        assert got["lo"] == pytest.approx(-0.1) and got["hi"] == pytest.approx(0.1)

    def test_the_point_and_the_total_spread(self):
        g = torch.Generator().manual_seed(4)
        tot = 1 + torch.rand(8, generator=g, dtype=F64)
        ea = tot * (0.2 + 0.1 * torch.rand(8, generator=g, dtype=F64))
        eb = tot * 0.1
        a, b = _sums([ea.tolist()], tot.tolist()), _sums([eb.tolist()], tot.tolist())
        c, c2 = _counts(8)
        got = ratio_interval(a, b, c, c2, 0.03)
        want = math.log2((1 - float(ea.sum() / tot.sum())) / 0.9)
        assert got["point"] == pytest.approx(want)
        sds = []
        for sa, sb, cc in ((a, b, c), (a.paired(), b.paired(), c2)):
            r = torch.log2(sa.draws(cc) / sb.draws(cc))
            sds.append(float(r.std()))
        tot_se = math.sqrt(max(sds) ** 2 + 0.03**2)
        assert got["se"] == pytest.approx(max(sds))
        assert got["lo"] == pytest.approx(want - 2 * tot_se)
        assert got["hi"] == pytest.approx(want + 2 * tot_se)

    def test_nothing_explained_is_dropped(self):
        tot = [1.0] * 8
        a = _sums([[1.2] * 8], tot)  # E = -0.2
        b = _sums([[0.1] * 8], tot)
        c, c2 = _counts(8)
        got = ratio_interval(a, b, c, c2, 0.01)
        assert (
            got["point"] == -math.inf
            and got["lo"] == -math.inf
            and got["hi"] == -math.inf
        )
        undefined = ratio_interval(b, a, c, c2, 0.01)
        assert undefined["point"] is None and undefined["lo"] is None

    def test_the_seeds_deviation_by_hand(self):
        tot = [1.0, 1.0]
        a = _sums([[0.25, 0.25], [0.5, 0.5], [0.0, 0.0]], tot)  # E_s 0.75, 0.5, 1
        b = _sums([[0.5, 0.5]] * 3, tot)  # E_s 0.5
        r = [math.log2(1.5), 0.0, 1.0]
        m = sum(r) / 3
        want = math.sqrt(sum((x - m) ** 2 for x in r) / 2)
        assert seed_sigma(a, b) == pytest.approx(want)

    def test_comparable(self):
        c, c2 = _counts(8)
        tot = [1.0] * 8
        assert not comparable(_sums([[1.0] * 8], tot), c, c2)["ok"]  # E = 0
        assert comparable(_sums([[0.5] * 8, [0.5] * 8], tot), c, c2)["ok"]
        # the seeds' spread counts: E_s 0.9 and -0.1 average to 0.4, not sure > 0
        assert not comparable(_sums([[0.1] * 8, [1.1] * 8], tot), c, c2)["ok"]


class TestVerdict:
    @staticmethod
    def iv(lo: float | None, hi: float | None) -> dict[str, float | None]:
        return {"lo": lo, "hi": hi}

    def test_by_hand(self):
        ok, low, mid = self.iv(-0.05, 0.05), self.iv(-0.5, -0.2), self.iv(-0.2, 0.0)
        wide, narrow = (0.6, 1.0), (0.4, 1.0)
        assert front_verdict(narrow, ok, ok, True, 0.5, 0.1) == "width"
        assert front_verdict((None, 3.0), ok, ok, True, 0.5, 0.1) == "width"
        assert front_verdict(wide, ok, ok, False, 0.5, 0.1) == "incomparable"
        assert front_verdict(wide, ok, ok, True, 0.5, 0.1) == "kept"
        assert front_verdict(wide, ok, low, True, 0.5, 0.1) == "dropped"
        assert front_verdict(wide, low, ok, True, 0.5, 0.1) == "dropped"
        assert front_verdict(wide, mid, ok, True, 0.5, 0.1) == "unresolved"
        assert (
            front_verdict(wide, self.iv(None, None), ok, True, 0.5, 0.1) == "unresolved"
        )


class TestReadout:
    def test_one_seed_twice_and_a_linear_map(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randn(512, 3, generator=g)
        y = x @ torch.randn(3, 2, generator=g)
        conf = MLPConf(hidden=32, steps=600, batch=64, lr=3e-3)
        a, la = fit_mlp(x, y, conf, 7, CPU)
        b, lb = fit_mlp(x, y, conf, 7, CPU)
        assert la == lb
        for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(pa, pb)
        _, lc = fit_mlp(x, y, conf, 8, CPU)
        assert lc != la
        p = predict(a, x, CPU)
        cs = content_sums(y, [p], y.mean(0), torch.arange(1, 512),
                          torch.zeros(511, dtype=torch.long), 1)  # fmt: skip
        assert cs.value() > 0.98
        assert not a.training
