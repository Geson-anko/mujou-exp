r"""The visual check of V18c's remade main measure's target (exp/vision/
README.md, "V18c: 主の指標を作り直す", 目視): for each of the first
``N_COMP`` whitened components of ``sem``, the ``N_EACH`` grid stacks of sel
with the lowest and the highest values, their hires last frames side by side
(a row of lows, a row of highs per component).

Zurich's hires frames are the grid's alone, in the grid's order (the frames
of ``sem_eval frames`` are sorted and the grid's starts ascend), so the
``i``-th Zurich frame of the hires data is grid stack ``i``. Runs where the
hires data and the cache are::

    python -m exp.vision.results.v18c.sem_extremes --sem <cache> --hires <dir> --out <jpg>
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from exp.lib.video_io import U8Reader
from exp.vision.fill_eval import Targets

N_COMP = 4
N_EACH = 8
THUMB = 96


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sem", type=Path, required=True)
    ap.add_argument("--hires", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t = Targets.load(args.sem)
    y = t.y("sem", "grid")
    sel = torch.tensor(t.meta["sets_split"]["sel"]["idx"])
    frames = U8Reader(args.hires / "Zurich")
    rows = []
    for j in range(N_COMP):
        order = sel[y[sel, j].argsort()]
        for pick in (order[:N_EACH], order[-N_EACH:]):
            x = torch.stack([frames[int(i)] for i in pick]).float()
            x = F.interpolate(x, size=THUMB, mode="area")
            rows.append(torch.cat(list(x), 2))  # (3, THUMB, N_EACH THUMB)
    grid = torch.cat(rows, 1).round().clamp(0, 255).to(torch.uint8)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(args.out, quality=85)
    print(args.out)


if __name__ == "__main__":
    main()
