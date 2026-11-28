r"""The features of images by the InceptionV3 of FID (as pytorch-fid's), and
the Fréchet distance of two sets of them.

The net is torchvision's ``inception_v3(num_classes=1008, aux_logits=False,
init_weights=False, weights=None)`` with the blocks ``Mixed_5b`` ..
``Mixed_7c`` replaced by ``FIDInceptionA``, ``FIDInceptionC``,
``FIDInceptionE_1`` and ``FIDInceptionE_2``: the TensorFlow graph of FID
averages its pools without the padded zeros and takes the maximum in the
last block's pool. These four classes are moved from pytorch-fid
(https://github.com/mseitzer/pytorch-fid, ``src/pytorch_fid/inception.py``,
Apache License 2.0, Copyright Maximilian Seitzer). The weights are its
release's ``pt_inception-2015-12-05-6726825d.pth`` (``WEIGHTS``), whose
sha256 begins with ``6726825d``; they are not downloaded here.

The features are pool3, the output of ``avgpool`` (2048 numbers, after a
ReLU, so not negative), of the images made as pytorch-fid's
``InceptionV3(resize_input=True, normalize_input=True)``: u8 $x$ to $x /
255$, enlarged to 299 x 299 by ``F.interpolate(mode="bilinear",
align_corners=False)`` (a frame that is not square is stretched), then $2x -
1$ (``preprocess``).

$\mathrm{FID} = \lVert \mu_1 - \mu_2 \rVert^2 + \mathrm{tr}(\Sigma_1 +
\Sigma_2 - 2 (\Sigma_1 \Sigma_2)^{1/2})$ of the means and covariances of two
sets, in float64, $\mathrm{tr}(\Sigma_1 \Sigma_2)^{1/2}$ the sum of the
square roots of the eigenvalues of the symmetric $\Sigma_1^{1/2} \Sigma_2
\Sigma_1^{1/2}$ (negative ones from rounding set to 0; no scipy).
"""

import hashlib
import os
from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.inception import (
    InceptionA,
    InceptionC,
    InceptionE,
    inception_v3,
)

from exp.lib.tours.stacks import DATA_ENV

__all__ = [
    "BATCH",
    "FEATURES",
    "SHA256_PREFIX",
    "SIDE",
    "WEIGHTS",
    "FIDInception",
    "fid",
    "frechet_distance",
    "load_fid_inception",
    "preprocess",
    "sha256",
    "weights_path",
]

WEIGHTS = "pt_inception-2015-12-05-6726825d.pth"
SHA256_PREFIX = "6726825d"
SIDE, FEATURES, BATCH = 299, 2048, 200


class FIDInceptionA(InceptionA):
    """InceptionA of FID (pytorch-fid): the pool averages without the
    padding."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_2(self.branch5x5_1(x))
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 1)


class FIDInceptionC(InceptionC):
    """InceptionC of FID (pytorch-fid): the pool averages without the
    padding."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 1)


class FIDInceptionE_1(InceptionE):
    """The first InceptionE of FID (pytorch-fid): the pool averages without
    the padding."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1
        )
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1
        )
        branch_pool = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


class FIDInceptionE_2(InceptionE):
    """The second InceptionE of FID (pytorch-fid): the pool is the
    maximum."""

    @override
    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = torch.cat(
            [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)], 1
        )
        branch3x3dbl = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        branch3x3dbl = torch.cat(
            [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)], 1
        )
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 1)


def preprocess(images: Tensor) -> Tensor:
    """The u8 images ``(B, 3, H, W)`` as the net's input ``(B, 3, 299,
    299)`` float32 in $[-1, 1]$ (module docstring)."""
    x = images.float() / 255
    x = F.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
    return 2 * x - 1


class FIDInception(nn.Module):
    """The InceptionV3 of FID up to pool3: u8 images ``(B, 3, H, W)`` to
    their features ``(B, 2048)`` (module docstring), ``BATCH`` at a time,
    without gradients. Its weights are random until loaded
    (``load_fid_inception``)."""

    def __init__(self) -> None:
        super().__init__()
        net = inception_v3(
            num_classes=1008, aux_logits=False, init_weights=False, weights=None
        )
        net.Mixed_5b = FIDInceptionA(192, pool_features=32)
        net.Mixed_5c = FIDInceptionA(256, pool_features=64)
        net.Mixed_5d = FIDInceptionA(288, pool_features=64)
        net.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        net.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        net.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        net.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        net.Mixed_7b = FIDInceptionE_1(1280)
        net.Mixed_7c = FIDInceptionE_2(2048)
        self.net = net
        self.eval()

    def _maps(self, x: Tensor) -> Tensor:
        """The last block's maps ``(B, 2048, h, w)`` of preprocessed ``x``."""
        n = self.net
        x = n.Conv2d_2b_3x3(n.Conv2d_2a_3x3(n.Conv2d_1a_3x3(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = n.Conv2d_4a_3x3(n.Conv2d_3b_1x1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        for block in ("5b", "5c", "5d", "6a", "6b", "6c", "6d", "6e"):
            x = getattr(n, f"Mixed_{block}")(x)
        for block in ("7a", "7b", "7c"):
            x = getattr(n, f"Mixed_{block}")(x)
        return x

    def _pool3(self, x: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(self._maps(x), (1, 1)).flatten(1)

    @torch.no_grad()
    def local(self, images: Tensor, cells: int = 2) -> Tensor:
        """The local features ``(B, 2048 cells^2)`` of u8 images ``(B, 3, H,
        W)``: the last block's maps averaged over each of ``cells`` x
        ``cells`` regions (``adaptive_avg_pool2d``; of 8 x 8 maps and 2 x 2
        cells, their mean over the cells is pool3), channel first, then the
        cells in rows (the linear horizon's guard, ``plans/jepa.md``, C11)."""
        dev = next(self.parameters()).device
        return torch.cat(
            [
                F.adaptive_avg_pool2d(self._maps(preprocess(c.to(dev))), cells).flatten(
                    1
                )
                for c in images.split(BATCH)
            ]
        )

    @override
    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        dev = next(self.parameters()).device
        return torch.cat(
            [self._pool3(preprocess(c.to(dev))) for c in images.split(BATCH)]
        )


def weights_path() -> Path:
    """Where the weights are by default: ``$MUJOU_EXP_DATA/inception/``
    ``WEIGHTS``, or ``.fleet/data/inception/`` of this checkout."""
    repo = Path(__file__).resolve().parents[2]
    return (
        Path(os.environ.get(DATA_ENV, repo / ".fleet" / "data")) / "inception" / WEIGHTS
    )


def sha256(path: Path) -> str:
    """The sha256 of the file ``path`` in hexadecimal."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_fid_inception(path: Path, device: str | torch.device = "cpu") -> FIDInception:
    """``FIDInception`` of the weights at ``path`` (``WEIGHTS``; loaded
    strictly, as pytorch-fid) on ``device``, in eval mode.

    Raises:
        ValueError: The file's sha256 does not begin with ``SHA256_PREFIX``.
    """
    digest = sha256(path)
    if not digest.startswith(SHA256_PREFIX):
        raise ValueError(f"{path}: sha256 {digest} is not FID's ({SHA256_PREFIX}...)")
    net = FIDInception()
    net.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()


def _sqrtm_psd(a: Tensor) -> Tensor:
    """The symmetric square root of the symmetric ``a`` (negative eigenvalues
    set to 0)."""
    w, v = torch.linalg.eigh(a)
    return (v * w.clamp(min=0).sqrt()) @ v.T


def frechet_distance(mu1: Tensor, s1: Tensor, mu2: Tensor, s2: Tensor) -> float:
    r"""The Fréchet distance of $N(\mu_1, \Sigma_1)$ and $N(\mu_2, \Sigma_2)$
    (module docstring), in float64; finite and not negative for covariances
    of any rank."""
    mu1, s1, mu2, s2 = (x.double() for x in (mu1, s1, mu2, s2))
    r1 = _sqrtm_psd(s1)
    m = r1 @ s2 @ r1
    cross = torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0).sqrt().sum()
    d = (mu1 - mu2).square().sum() + s1.trace() + s2.trace() - 2 * cross
    return max(float(d), 0.0)


def fid(f1: Tensor, f2: Tensor) -> float:
    """``frechet_distance`` of the means and the covariances (of $N - 1$) of
    the features ``f1`` ``(N1, F)`` and ``f2`` ``(N2, F)``."""
    a, b = f1.double(), f2.double()
    return frechet_distance(a.mean(0), a.T.cov(), b.mean(0), b.T.cov())
