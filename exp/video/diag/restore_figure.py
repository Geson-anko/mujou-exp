r"""DIAGNOSIS ONLY (M17): the restore pictures of the posthoc
(``<ckpt>-restore.png`` of ``measure/encoder_eval.py``: rows the true frame,
the unmasked ceiling, then per context view the masked input and its
restoration) stacked into one PNG, each under a title line and with the
names of its rows on the left.

Made the figures of 「画像（val の 10 窓、1800 秒）」 of 「ストリームごとのバッファ」
in ``exp/video/results/m17.md`` (``m17-buf-32.png``, ``m17-buf-1.png``,
``m17-buf-all.png``; not in the repository). Needs Pillow and a font with
Japanese (default macOS's Hiragino)::

    uv run --with pillow python -m exp.video.diag.restore_figure OUT.png TITLE1 PNG1 [TITLE2 PNG2 ...]
"""

import argparse
from collections.abc import Sequence

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
ROWS = [
    "入力",
    "復元（マスクなし）",
    "マスク（short）",
    "マスク復元（予測器、short）",
    "マスク（long）",
    "マスク復元（予測器、long）",
]
LABEL_W, TITLE_H = 300, 44


def layout(
    sizes: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """The canvas ``(width, height)`` of pictures of ``sizes`` ``(w, h)``, and
    the top-left corner of each: a column right of the labels, each picture
    under a title line of ``TITLE_H``."""
    width = LABEL_W + max(w for w, _ in sizes)
    corners, y = [], 0
    for _, h in sizes:
        y += TITLE_H
        corners.append((LABEL_W, y))
        y += h
    return (width, y), corners


def compose_restore(out: str, pairs: Sequence[tuple[str, str]], font: str) -> None:
    """Write ``out`` from ``(title, png)`` pairs (module docstring)."""
    from PIL import (  # pyright: ignore[reportMissingImports]
        Image,
        ImageDraw,
        ImageFont,
    )

    lfont = ImageFont.truetype(font, 20)
    tfont = ImageFont.truetype(font, 22)
    ims = [Image.open(p).convert("RGB") for _, p in pairs]
    size, corners = layout([im.size for im in ims])
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    for (title, _), im, (x, y) in zip(pairs, ims, corners, strict=True):
        draw.text((8, y - TITLE_H + 10), title, fill="black", font=tfont)
        canvas.paste(im, (x, y))
        row_h = im.height / len(ROWS)
        for i, name in enumerate(ROWS):
            cy = y + (i + 0.5) * row_h
            draw.text((LABEL_W - 10, cy), name, fill="black", font=lfont, anchor="rm")
    canvas.save(out)
    print(out, canvas.size)


def main(argv: Sequence[str] | None = None) -> None:
    """``OUT TITLE1 PNG1 [TITLE2 PNG2 ...]``."""
    p = argparse.ArgumentParser(description="Stack restore pictures (M17).")
    p.add_argument("out")
    p.add_argument("pairs", nargs="+", help="TITLE PNG, repeated")
    p.add_argument("--font", default=FONT)
    args = p.parse_args(argv)
    if len(args.pairs) % 2:
        p.error("the pictures come in TITLE PNG pairs")
    pairs = list(zip(args.pairs[::2], args.pairs[1::2], strict=True))
    compose_restore(args.out, pairs, args.font)


if __name__ == "__main__":
    main()
