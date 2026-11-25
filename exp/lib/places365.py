r"""Places365's ResNet-50 and its scene hierarchy: the coarse scene labels of
V18c's main measure (``exp/vision/plans/jepa.md``, 「V18c: 主の指標（場面の
カテゴリ）」).

The network is Places365-CNN's ResNet-50 (Zhou, Lapedriza, Khosla, Oliva and
Torralba, "Places: A 10 million Image Database for Scene Recognition", TPAMI
2017; the weights ``resnet50_places365.pth.tar`` from
``http://places2.csail.mit.edu/models_places365/``, CC BY), loaded into
``torchvision.models.resnet50(num_classes=365)`` with the ``module.``
prefix taken off, its sha256 checked against ``RESNET50_SHA256``.

Vendored byte for byte (fetched 2026-11-25; their sha256 checked on
reading, so pre-commit leaves them alone):

- ``places365_categories.txt``: the 365 class names in order (the official
  repository ``CSAILVision/places365`` at commit ``8a953ed5``, MIT);
- ``places365_hierarchy.csv``: the official scene hierarchy (the Google
  spreadsheet ``1H7ADoEIGgbF_eXh9kcJjCs5j_r3VJwke4nebhkdzksg`` linked from
  that repository's README, exported as CSV): two header rows, then a row per
  class (its name quoted with ``'``) and 0/1 of level 1's 3 columns and level
  2's 16 groups. A class may be in several groups (63 of the 365 are).

The preprocessing is the official demo's (``run_placesCNN_basic.py`` at
``8a953ed5``): each image as a PIL image, ``Resize((256, 256))``,
``CenterCrop(224)``, ``ToTensor()`` and ImageNet's ``Normalize``. The
groups' probabilities: with the softmax $p_c$ of the 365 logits and the
membership $M_{cg} \in \{0, 1\}$ ``(365, 16)``, $q_g = \sum_c p_c M_{cg} /
\sum_{g'} M_{cg'}$ (a class in several groups split evenly, so $\sum_g q_g
= 1$); the label is $\arg\max_g q_g$.

``check`` is V-c0's of the network (README 「有効性の確かめ」): the files'
sha256, the class names against the hierarchy's rows (in order), every class
in a group, the weights loaded with nothing missing or left over, and the top
5 classes of a demo image and their probabilities against the official demo
script's (``--reference``: ``{"image_sha256", "top5": [[name, prob], ...]}``,
the script run with ``arch = 'resnet50'`` on the same weights)::

    python -m exp.lib.places365 check --weights <resnet50_places365.pth.tar> \
        --image <demo image> --reference <json> [--out <json>]
"""

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import models, transforms

from exp.lib.inception import sha256

__all__ = [
    "CATEGORIES",
    "CATEGORIES_SHA256",
    "HIERARCHY",
    "HIERARCHY_SHA256",
    "RESNET50_SHA256",
    "check",
    "class_names",
    "group_names",
    "group_probs",
    "load_places365",
    "logits",
    "membership",
    "penultimate",
    "preprocess",
]

CATEGORIES = Path(__file__).with_name("places365_categories.txt")
HIERARCHY = Path(__file__).with_name("places365_hierarchy.csv")
CATEGORIES_SHA256 = "2affba635eb657e7ca95f4e6cc69bd9fac29ef4c32aeb83cafdfcd06ec6a1ea6"
HIERARCHY_SHA256 = "83bb268885f98de62e8c426115d358094a6c0f01c4e3c07450a0f576adc65ee9"
RESNET50_SHA256 = "46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5"
CLASSES, GROUPS = 365, 16
_LEVEL2 = slice(4, 4 + GROUPS)  # the hierarchy's columns of level 2
PROB_TOL = 1e-4  # V-c0: the demo's probabilities against the official script's

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _input(image: Image.Image) -> Tensor:
    """The official transform of a PIL image ``(3, 224, 224)``."""
    out = _TRANSFORM(image)
    assert isinstance(out, Tensor)
    return out


def _checked(path: Path, digest: str) -> Path:
    """``path`` if its sha256 is ``digest``.

    Raises:
        ValueError: It is not.
    """
    got = sha256(path)
    if got != digest:
        raise ValueError(f"{path} has sha256 {got}, expected {digest}")
    return path


def _hierarchy() -> list[list[str]]:
    """The hierarchy's rows (module docstring), two headers first."""
    text = _checked(HIERARCHY, HIERARCHY_SHA256).read_text()
    return list(csv.reader(text.splitlines()))


def class_names() -> list[str]:
    """The 365 class names in order (``/a/airfield``, ...)."""
    text = _checked(CATEGORIES, CATEGORIES_SHA256).read_text()
    return [line.split(" ")[0] for line in text.splitlines()]


def group_names() -> list[str]:
    """The 16 groups of level 2, in the hierarchy's order."""
    return _hierarchy()[1][_LEVEL2]


def membership() -> Tensor:
    """$M_{cg}$ ``(365, 16)`` float64: 1 where class $c$ (``class_names``'
    order) is in group $g$ of level 2.

    Raises:
        ValueError: The hierarchy's classes are not ``class_names()`` in
            order, or a class is in no group.
    """
    rows = _hierarchy()[2:]
    names = [r[0].strip("'") for r in rows]
    if names != class_names():
        raise ValueError("the hierarchy's classes are not the class names in order")
    m = torch.tensor(
        [[float(v) for v in r[_LEVEL2]] for r in rows], dtype=torch.float64
    )
    empty = (m.sum(1) == 0).nonzero().flatten().tolist()
    if empty:
        raise ValueError(f"classes in no group: {[names[i] for i in empty]}")
    return m


def load_places365(path: Path, device: str | torch.device) -> nn.Module:
    """The ResNet-50 of the weights at ``path`` (module docstring) on
    ``device``, in eval mode.

    Raises:
        ValueError: The weights' sha256 is not ``RESNET50_SHA256``.
        RuntimeError: A key is missing or left over (``load_state_dict``).
    """
    _checked(path, RESNET50_SHA256)
    net = models.resnet50(num_classes=CLASSES)
    d = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.replace("module.", "", 1): v for k, v in d["state_dict"].items()}
    net.load_state_dict(state, strict=True)
    return net.to(device).eval()


def preprocess(frames: Tensor) -> Tensor:
    """The u8 frames ``(B, 3, H, W)`` through the official demo's transform
    (module docstring), each as a PIL image: ``(B, 3, 224, 224)`` float32 on
    the CPU."""
    hwc = frames.cpu().permute(0, 2, 3, 1).contiguous().numpy()
    return torch.stack([_input(Image.fromarray(x)) for x in hwc])


@torch.no_grad()
def logits(net: nn.Module, x: Tensor) -> Tensor:
    """The 365 logits ``(B, 365)`` float32 of the preprocessed ``x``."""
    dev = next(net.parameters()).device
    return net(x.to(dev)).float()


def group_probs(logit: Tensor, m: Tensor) -> Tensor:
    """$q$ ``(B, G)`` float64 (module docstring) of the logits ``(B, C)``
    and the membership ``m`` ``(C, G)``."""
    p = torch.softmax(logit.double(), 1)
    md = m.double().to(p.device)
    return p @ (md / md.sum(1, keepdim=True))


@torch.no_grad()
def penultimate(net: nn.Module, x: Tensor) -> Tensor:
    """The features ``(B, 2048)`` float32 before the last layer (after the
    global average pool) of the preprocessed ``x``."""
    dev = next(net.parameters()).device
    body = nn.Sequential(*list(net.children())[:-1])
    return body(x.to(dev)).flatten(1).float()


def check(weights: Path, image: Path, reference: dict[str, Any]) -> dict[str, Any]:
    """V-c0's checks of the network (module docstring) and their record.

    Raises:
        ValueError: A file's sha256, the hierarchy, the image or the top 5
            differ.
    """
    names = class_names()
    m = membership()  # the names against the hierarchy, every class in a group
    if m.shape != (CLASSES, GROUPS):
        raise ValueError(f"a membership of {tuple(m.shape)}")
    img_sha = sha256(image)
    if reference.get("image_sha256") not in (None, img_sha):
        raise ValueError(f"the reference is of another image than {image}")
    net = load_places365(weights, "cpu")
    with Image.open(image) as im:
        x = _input(im.convert("RGB"))[None]
    p = torch.softmax(logits(net, x).double(), 1)[0]
    probs, idx = p.sort(descending=True)
    top5 = [[names[int(i)][3:], float(q)] for q, i in zip(probs[:5], idx[:5])]
    want = reference["top5"]
    same = [a[0] for a in top5] == [b[0] for b in want]
    diff = max(abs(a[1] - b[1]) for a, b in zip(top5, want))
    if not same or diff > PROB_TOL:
        raise ValueError(f"the top 5 {top5} differ from the official script's {want}")
    return {
        "weights_sha256": RESNET50_SHA256,
        "categories_sha256": CATEGORIES_SHA256,
        "hierarchy_sha256": HIERARCHY_SHA256,
        "image": str(image),
        "image_sha256": img_sha,
        "classes": CLASSES,
        "groups": group_names(),
        "multi_group_classes": int((m.sum(1) > 1).sum()),
        "top5": top5,
        "reference_top5": want,
        "max_prob_diff": diff,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(
        prog="python -m exp.lib.places365", description="Places365's V-c0 checks."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="V-c0's checks of the network")
    c.add_argument("--weights", type=Path, required=True)
    c.add_argument("--image", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    got = check(args.weights, args.image, json.loads(args.reference.read_text()))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(got, indent=1))
    return got


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
