"""Build ``exp/vision/eval-guide.html``: a guide to V18c's evaluation metrics.

The prose lives in ``eval_guide.html.in``; each ``<!--SVG:name-->`` in it is
replaced by the schematic figure drawn by ``FIGURES[name]``. The only real
data drawn is the G bar chart, whose numbers are copied from the README
("V18c: 生成と Inception・未観測の部分の評価の結果", results 2 and 3).

Subcommands for the prose review (the yomiyasu pass):

- ``build``: write the HTML.
- ``extract <out.json>``: list the inner HTML of the template's text blocks.
- ``apply <in.json>``: write back the edited blocks (matched by index and
  checked against the original text).

Run with ``uv run python -m exp.vision.report.build_eval_guide build``.
"""

import json
import math
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "eval_guide.html.in"
OUT = HERE.parent / "eval-guide.html"

BLOCK = re.compile(
    r"<(h1|h2|h3|p|li|figcaption|dt|dd|td|th)(\s[^>]*)?>(.*?)</\1>", re.DOTALL
)


def _opacities(seed: int, n: int, lo: float = 0.2, hi: float = 0.9) -> list[float]:
    rng = random.Random(seed)
    return [round(lo + (hi - lo) * rng.random(), 2) for _ in range(n)]


def grid(
    x: float,
    y: float,
    n: int,
    cell: float,
    cls: list[str] | str,
    op: list[float] | float,
) -> str:
    """An ``n`` x ``n`` grid of cells, row-major, each with a class and an
    opacity."""
    cells = n * n
    classes = cls if isinstance(cls, list) else [cls] * cells
    ops = op if isinstance(op, list) else [op] * cells
    out = []
    for i in range(cells):
        r, c = divmod(i, n)
        out.append(
            f'<rect class="cell {classes[i]}" x="{x + c * cell}" y="{y + r * cell}" '
            f'width="{cell}" height="{cell}" fill-opacity="{ops[i]}"/>'
        )
    return "".join(out)


def text(
    x: float,
    y: float,
    s: str,
    anchor: str = "middle",
    cls: str = "",
    size: float | None = None,
) -> str:
    style = f' font-size="{size}"' if size else ""
    c = f' class="{cls}"' if cls else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}"{c}{style}>{s}</text>'


def arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    d = ' stroke-dasharray="4 4"' if dashed else ""
    return (
        f'<line class="ln" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"{d} '
        'marker-end="url(#ah)"/>'
    )


def box(x: float, y: float, w: float, h: float, lines: list[str]) -> str:
    out = [f'<rect class="box" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>']
    n = len(lines)
    for i, s in enumerate(lines):
        cy = y + h / 2 + (i - (n - 1) / 2) * 18 + 5
        cls = "strong" if i == 0 else "small"
        out.append(text(x + w / 2, cy, s, cls=cls))
    return "".join(out)


def svg(view: str, body: str, label: str) -> str:
    return (
        f'<div class="chart"><svg class="dg" viewBox="{view}" role="img" '
        f'aria-label="{label}">{body}</svg></div>'
    )


def fig_jepa() -> str:
    img = grid(20, 40, 8, 20, "cn", _opacities(1, 64, 0.15, 0.8))
    hid = '<rect class="hatch" x="100" y="120" width="80" height="80"/>'
    hid += '<rect class="qline" x="100" y="120" width="80" height="80"/>'
    pred = grid(515, 44, 4, 18, "c1", _opacities(2, 16, 0.3, 0.9))
    real = grid(515, 179, 4, 18, "c1", _opacities(3, 16, 0.3, 0.9))
    b = [
        text(12, 28, "入力の stack（64×64 画素）", anchor="start", cls="strong"),
        img,
        hid,
        text(100, 222, "8×8 のパッチに分ける", cls="small"),
        text(100, 240, "斜線：隠した四半分", cls="small"),
        arrow(182, 80, 218, 80),
        box(220, 55, 150, 50, ["エンコーダ", "見える部分だけ"]),
        arrow(370, 80, 398, 80),
        box(400, 55, 90, 50, ["予測器"]),
        arrow(490, 80, 512, 80),
        pred,
        text(551, 34, "予測", cls="strong"),
        '<polyline class="ln" points="182,160 200,160 200,215 218,215" '
        'marker-end="url(#ah)"/>',
        box(220, 190, 150, 50, ["エンコーダ", "全体を見る"]),
        arrow(370, 215, 393, 215),
        box(395, 190, 112, 50, ["R（固定）", "1024 → 16 次元"]),
        arrow(507, 215, 513, 215),
        real,
        text(551, 268, "本物 t（的）", cls="strong"),
        '<line class="ln" x1="551" y1="126" x2="551" y2="174" '
        'marker-start="url(#ah)" marker-end="url(#ah)"/>',
        text(540, 155, "L_pred：差を小さくする", anchor="end", cls="small"),
        '<line class="ln" x1="295" y1="107" x2="295" y2="188" stroke-dasharray="3 4"/>',
        text(303, 152, "同じ重み", anchor="start", cls="small"),
    ]
    return svg("0 0 640 285", "".join(b), "JEPA の学習の模式図")


def fig_collapse() -> str:
    b = []
    panels = [
        ("健全な表現", ("c1", _opacities(4, 16)), ("c1", _opacities(5, 16))),
        ("K1：定数に崩壊", ("cn", 0.55), ("cn", 0.55)),
        ("K3：stack の中で一定", ("c1", 0.7), ("c2", 0.7)),
    ]
    notes = [
        ["位置ごと・stack ごとに", "違う値になる"],
        ["どの stack でも同じ値。", "SIGReg が約 52 に跳ね上がる"],
        ["stack ごとには違うが、", "中の位置はみな同じ情報"],
    ]
    for i, (title, s1, s2) in enumerate(panels):
        x0 = 10 + i * 220
        b.append(text(x0 + 100, 22, title, cls="strong"))
        b.append(grid(x0 + 10, 36, 4, 20, s1[0], s1[1]))
        b.append(grid(x0 + 110, 36, 4, 20, s2[0], s2[1]))
        b.append(text(x0 + 50, 134, "stack 1", cls="small"))
        b.append(text(x0 + 150, 134, "stack 2", cls="small"))
        for k, s in enumerate(notes[i]):
            b.append(text(x0 + 100, 166 + k * 20, s, cls="small"))
    return svg("0 0 660 215", "".join(b), "崩壊した表現の模式図")


def fig_g_bars() -> str:
    """Real data: G's ``top1`` (README, V18c generation results 2 and 3)."""
    x0, w, top, step = 175.0, 420.0, 34.0, 34.0
    scale = w / 0.03

    def bx(v: float) -> float:
        return x0 + v * scale

    rows = [
        ("本物の 1 秒後（参照）", 0.840, "c3", "0.840（目盛りの外）"),
        ("v16（VAE）", 0.024, "cn", "0.024"),
        ("pca（画素の PCA）", 0.023, "cn", "0.023"),
        ("v17（VAE）", 0.020, "cn", "0.020"),
        ("color（色の平均）", 0.014, "cn", "0.014"),
        ("JEPA の 60000 step", None, "c1", "0.008〜0.017"),
        ("uncond（条件なし）", 0.002, "cn", "0.002"),
    ]
    b = []
    bottom = top + step * len(rows)
    for t in (0.0, 0.01, 0.02, 0.03):
        b.append(
            f'<line class="gridline" x1="{bx(t)}" y1="{top - 10}" '
            f'x2="{bx(t)}" y2="{bottom}"/>'
        )
        b.append(text(bx(t), bottom + 18, f"{t:.2f}", cls="mono small"))
    for i, (lab, v, cls, val) in enumerate(rows):
        y = top + i * step
        b.append(text(x0 - 8, y + 13, lab, anchor="end", cls="small"))
        if v is None:
            a, z = bx(0.008), bx(0.017)
            b.append(
                f'<rect class="{cls}" x="{a}" y="{y}" width="{z - a}" height="18" '
                'fill-opacity="0.85" rx="2"/>'
            )
            b.append(text(z + 6, y + 14, val, anchor="start", cls="mono small"))
        elif v > 0.03:
            b.append(
                f'<rect class="{cls}" x="{x0}" y="{y}" width="{w}" height="18" '
                'fill-opacity="0.85" rx="2"/>'
            )
            b.append(
                text(x0 + w - 6, y + 14, val, anchor="end", cls="mono small onbar")
            )
        else:
            b.append(
                f'<rect class="{cls}" x="{x0}" y="{y}" width="{bx(v) - x0}" '
                'height="18" fill-opacity="0.85" rx="2"/>'
            )
            b.append(text(bx(v) + 6, y + 14, val, anchor="start", cls="mono small"))
    c = bx(1 / 512)
    b.append(f'<line class="chance" x1="{c}" y1="{top - 10}" x2="{c}" y2="{bottom}"/>')
    b.append(text(c + 4, top - 14, "偶然 1/512", anchor="start", cls="small"))
    return svg(
        f"0 0 640 {bottom + 30}", "".join(b), "生成と Inception による同定の正解率"
    )


def fig_timeline() -> str:
    b = [
        '<line class="axis" x1="20" y1="60" x2="620" y2="60"/>',
        text(320, 24, "Zurich の動画（1 秒おきの格子 3899 個）", cls="strong"),
    ]
    for k in range(31):
        x = 20 + k * 20
        b.append(f'<line class="tick c2s" x1="{x}" y1="48" x2="{x}" y2="60"/>')
    b.append(
        text(320, 42, "易しい候補：全体から等間隔に 512 個（7〜8 秒おき）", cls="small")
    )
    b.append('<circle class="c1" cx="330" cy="60" r="6"/>')
    b.append(
        '<line class="ln" x1="318" y1="66" x2="138" y2="120" stroke-dasharray="3 4"/>'
    )
    b.append(
        '<line class="ln" x1="342" y1="66" x2="522" y2="120" stroke-dasharray="3 4"/>'
    )
    b.append('<line class="axis" x1="130" y1="125" x2="530" y2="125"/>')
    for s in range(-32, 33, 2):
        if s == 0:
            continue
        x = 330 + s * 6
        b.append(f'<line class="tick c4s" x1="{x}" y1="117" x2="{x}" y2="133"/>')
    b.append('<circle class="c1" cx="330" cy="125" r="6"/>')
    b.append(text(138, 150, "−32 秒", cls="mono small"))
    b.append(text(522, 150, "+32 秒", cls="mono small"))
    b.append(text(330, 150, "問い合わせ（自分）", cls="small"))
    b.append(
        text(
            330,
            176,
            "難しい候補：同じ動画の前後 ±2〜±32 秒、2 秒おきの 32 個と自分（計 33 個）",
            cls="small",
        )
    )
    return svg("0 0 640 190", "".join(b), "易しい候補と難しい候補の模式図")


def fig_pairwin() -> str:
    b = [
        text(80, 60, "問い合わせ", cls="strong"),
        text(80, 80, "フレーム n の予測", cls="small"),
        grid(44, 96, 4, 18, "c1", _opacities(6, 16, 0.3, 0.9)),
        arrow(126, 132, 168, 132),
        text(620, 22, "値は説明のための例", anchor="end", cls="small warn"),
    ]
    rows = [
        ("本物：フレーム n（正解）", 0.62, "c3", "—"),
        ("本物：フレーム a", 0.71, "cn", "負け"),
        ("本物：フレーム b", 0.40, "cn", "勝ち"),
        ("本物：フレーム c", 0.15, "cn", "勝ち"),
        ("本物：フレーム d", 0.33, "cn", "勝ち"),
    ]
    for i, (lab, v, cls, res) in enumerate(rows):
        y = 40 + i * 40
        b.append(text(178, y + 14, lab, anchor="start", cls="small"))
        b.append(
            f'<rect class="{cls}" x="370" y="{y}" width="{v * 220}" height="18" '
            'fill-opacity="0.85" rx="2"/>'
        )
        b.append(
            text(376 + v * 220, y + 14, f"{v:.2f}", anchor="start", cls="mono small")
        )
        b.append(text(620, y + 14, res, anchor="end", cls="strong"))
    b.append('<line class="chance" x1="506.4" y1="34" x2="506.4" y2="236"/>')
    b.append(
        text(
            320,
            262,
            "対の勝率 Aₙ = 3 / 4 = 0.75　　top1 = 外れ（a が上にいる）",
            cls="strong",
        )
    )
    return svg("0 0 640 275", "".join(b), "対の勝率の模式図")


def fig_center() -> str:
    pts = [
        (-25, -10),
        (18, -22),
        (5, 20),
        (-12, 25),
        (28, 8),
        (-30, 12),
        (10, -5),
        (22, 26),
    ]
    b = [
        text(160, 22, "中心化の前", cls="strong"),
        text(480, 22, "参照の平均を引いた後", cls="strong"),
    ]
    ox, oy, cx, cy = 30, 220, 200, 90
    b.append(f'<line class="axis" x1="{ox}" y1="{oy}" x2="300" y2="{oy}"/>')
    b.append(f'<line class="axis" x1="{ox}" y1="{oy}" x2="{ox}" y2="40"/>')
    b.append(text(ox - 4, oy + 4, "原点", anchor="end", cls="small"))
    for i in (0, 4):
        px, py = cx + pts[i][0], cy + pts[i][1]
        b.append(f'<line class="ray" x1="{ox}" y1="{oy}" x2="{px}" y2="{py}"/>')
    for dx, dy in pts:
        b.append(
            f'<circle class="c1" cx="{cx + dx}" cy="{cy + dy}" r="5" fill-opacity="0.85"/>'
        )
    b.append(f'<path class="cross" d="M{cx - 7} {cy} h14 M{cx} {cy - 7} v14"/>')
    b.append(text(cx + 40, cy + 50, "参照の組の平均", anchor="start", cls="small"))
    b.append(text(160, 256, "どの点も同じ向きにずれ、cos はほぼ 1", cls="small"))
    ox2, oy2 = 480, 130
    b.append(f'<line class="axis" x1="350" y1="{oy2}" x2="610" y2="{oy2}"/>')
    b.append(f'<line class="axis" x1="{ox2}" y1="40" x2="{ox2}" y2="220"/>')
    for i in (0, 4):
        px, py = ox2 + 2.4 * pts[i][0], oy2 + 2.4 * pts[i][1]
        b.append(f'<line class="ray" x1="{ox2}" y1="{oy2}" x2="{px}" y2="{py}"/>')
    for dx, dy in pts:
        b.append(
            f'<circle class="c1" cx="{ox2 + 2.4 * dx}" cy="{oy2 + 2.4 * dy}" r="5" '
            'fill-opacity="0.85"/>'
        )
    b.append(text(480, 256, "違いが向きに出る（2.4 倍に拡大して描いた）", cls="small"))
    return svg("0 0 640 270", "".join(b), "中心化の模式図")


def fig_controls() -> str:
    flat_op = round(sum(_opacities(7, 16, 0.3, 0.9)) / 16, 2)
    grad = [round(0.25 + 0.06 * (r + c), 2) for r in range(4) for c in range(4)]
    cols = [
        ("pred", "c1", _opacities(7, 16, 0.3, 0.9), ["予測器の出力"], ["測りたいもの"]),
        (
            "flat",
            "c1",
            flat_op,
            ["予測の四半分の", "平均を全位置に"],
            ["四半分の平均だけ", "で当たる分"],
        ),
        (
            "pool",
            "c3",
            0.6,
            ["見える部分の", "平均を全位置に"],
            ["周りを写すだけ", "で当たる分"],
        ),
        (
            "interp",
            "c4",
            grad,
            ["画素を補間して", "エンコーダへ"],
            ["滑らかな延長", "で当たる分"],
        ),
        (
            "oracle",
            "c1",
            _opacities(8, 16, 0.3, 0.9),
            ["全体を見た本物"],
            ["天井（A = 1）"],
        ),
    ]
    b = [
        text(
            10,
            18,
            "隠した四半分 Q の 16 トークンに何を置くか",
            anchor="start",
            cls="strong",
        )
    ]
    for i, (name, cls, op, what, removes) in enumerate(cols):
        cx = 70 + i * 140
        b.append(grid(cx - 32, 32, 4, 16, cls, op))
        b.append(text(cx, 120, name, cls="mono strong"))
        for k, s in enumerate(what):
            b.append(text(cx, 142 + k * 17, s, cls="small"))
        for k, s in enumerate(removes):
            b.append(text(cx, 190 + k * 17, s, cls="small accent"))
    return svg("0 0 700 230", "".join(b), "対照の段の模式図")


def fig_swap() -> str:
    img = grid(20, 40, 4, 30, "cn", _opacities(9, 16, 0.15, 0.7))
    b = [
        img,
        '<rect class="hatch" x="20" y="40" width="60" height="60"/>',
        '<rect class="hatch" x="80" y="100" width="60" height="60"/>',
        '<rect class="qline" x="20" y="40" width="60" height="60"/>',
        '<rect class="qline" x="80" y="100" width="60" height="60"/>',
        text(50, 76, "A", cls="big"),
        text(110, 136, "B", cls="big"),
        text(80, 186, "対角の 2 つの四半分を", cls="small"),
        text(80, 204, "同時に隠す（k = 8）", cls="small"),
        arrow(150, 100, 188, 100),
        text(300, 70, "d = p_A − p_B", cls="mono strong"),
        text(300, 94, "（参照の平均を引く）", cls="small"),
        text(300, 132, "d と A の本物 r_A の内積が", cls="small"),
        text(300, 152, "正なら 1、0 なら ½、負なら 0", cls="small"),
        arrow(410, 100, 438, 70),
        arrow(410, 100, 438, 150),
        box(440, 30, 210, 70, ["位置ごとに違う予測", "A の本物に向けば", "1 に近づく"]),
        box(440, 120, 210, 70, ["複製の K3", "d = 0 で必ず引き分け", "→ ちょうど 0.5"]),
    ]
    return svg("0 0 660 220", "".join(b), "swap の模式図")


def fig_transplant() -> str:
    j_ops = _opacities(10, 16, 0.25, 0.85)
    m_ops = _opacities(11, 16, 0.25, 0.85)
    q = {0, 1, 4, 5}

    def mixed(base_cls: str, base: list[float], q_cls: str, qv: list[float]) -> str:
        cls = [q_cls if i in q else base_cls for i in range(16)]
        ops = [qv[i] if i in q else base[i] for i in range(16)]
        return grid(0, 0, 4, 22, cls, ops)

    def at(x: float, y: float, inner: str) -> str:
        return f'<g transform="translate({x},{y})">{inner}</g>'

    qbox = '<rect class="qline" x="0" y="0" width="44" height="44"/>'
    b = [
        text(74, 30, "文脈 j（別の動画）", cls="strong"),
        at(30, 40, grid(0, 0, 4, 22, "c3", j_ops) + qbox),
        text(74, 150, "候補 m（Zurich）", cls="strong"),
        at(30, 160, grid(0, 0, 4, 22, "c2", m_ops) + qbox),
        text(354, 30, "順向き（j ← m）", cls="strong"),
        at(310, 40, mixed("c3", j_ops, "c2", m_ops) + qbox),
        text(354, 150, "逆向き（m ← j）", cls="strong"),
        at(310, 160, mixed("c2", m_ops, "c3", j_ops) + qbox),
        arrow(124, 95, 304, 95),
        arrow(124, 215, 304, 215),
        '<line class="ln" x1="76" y1="182" x2="304" y2="62" stroke-dasharray="4 4" '
        'marker-end="url(#ah)"/>',
        '<line class="ln" x1="76" y1="62" x2="304" y2="182" stroke-dasharray="4 4" '
        'marker-end="url(#ah)"/>',
        text(200, 170, "Q だけ入れ替え", cls="small"),
        text(416, 70, "Q だけが m の画素", anchor="start", cls="small"),
        text(416, 90, "→ 候補どうしの違いは", anchor="start", cls="small"),
        text(416, 110, "　Q の画素から来る", anchor="start", cls="small"),
        text(416, 190, "Q 以外が m の画素", anchor="start", cls="small"),
        text(416, 210, "→ 候補どうしの違いは", anchor="start", cls="small"),
        text(416, 230, "　文脈から来る", anchor="start", cls="small"),
        text(
            20,
            270,
            "太線の枠が四半分 Q。入れ替えた画像を全体を見るエンコーダに通し、",
            anchor="start",
            cls="small",
        ),
        text(20, 288, "Q の位置のトークンを候補にする", anchor="start", cls="small"),
    ]
    return svg("0 0 680 300", "".join(b), "移植の模式図")


# ---- time linearity and the width of information (V18c, 2026-11-26) ----
# The pixel PCA's ``in`` curve (README, "V18c: 時間の線形性と情報の幅の有効性",
# V-d3 (a)): lags in seconds, G_mid and s.
PCA_CURVE = [
    (0.8, 0.589, 0.319),
    (1.6, 0.662, 0.521),
    (3.2, 0.704, 0.701),
    (6.4, 0.724, 0.831),
    (12.8, 0.741, 0.911),
    (25.6, 0.743, 0.936),
]
# The largest delta V-d2 could inject at each lag (same README section).
VD2_MAX_DELTA = [0.08, 0.08, 0.04, 0.02, 0.005, 0.005]
# The failure example (``ou_mix_30`` whitened by another draw, k = 64) and
# the provisional bounds of the spread diagnosis (README, the user's decision).
OU_MIX_30 = {"V_over_k": 9.14, "var_max": 195.0, "top_share": 0.81}
BOUNDS = {"V_over_k": (0.5, 2.0), "var_max": (None, 10.0), "top_share": (None, 0.4)}
GEOM_JSON = HERE.parent / "results" / "v18c" / "judge_geom.json"
YARD_CODES = {
    "pca",
    "color",
    "mean8",
    "init",
    "init_f",
    "init_m",
    "init_c",
    "init_c1",
    "init_v16",
}


def geom_name(code: str) -> str:
    """A short label of a code of ``judge_geom.json``."""
    base, _, step = code.partition("@")
    tail = "" if step in ("", "60000") else f"（{step}）"
    if base.startswith("v18c-"):
        lam, seed = base.split("-lam")[1].split("-")[0], base[-2:]
        lr = "1e-4" if "-lr1e-4-" in base else "1e-5"
        return f"V18c {lr}・λ{lam}・{seed}{tail}"
    if base.startswith("v18b-long-"):
        arm = base.split("-")[2].upper()
        return f"V18b {arm} {base[-2:]}{tail}"
    if base.startswith("v18-long-"):
        return f"V18 r{base.split('-r')[1].split('-')[0]}・{base[-2:]}{tail}"
    return base


def _spread_rows() -> list[tuple[str, str, dict[str, float]]]:
    """``(place, code, diagnosis)`` of the yardsticks, the VAEs and the
    step-60000 checkpoints (``judge_geom.json``'s ``spread``)."""
    spread = json.loads(GEOM_JSON.read_text(encoding="utf-8"))["spread"]
    return [
        (place, code, v)
        for place in ("own", "h_mean")
        for code, v in spread[place].items()
        if not code.endswith("@6000")
    ]


def _flagged(v: dict[str, float], key: str, lo: float | None, hi: float | None) -> bool:
    x = v[key]
    return (lo is not None and x < lo) or (hi is not None and x > hi)


def fig_tl_double() -> str:
    def frame(x: float, dots: list[tuple[float, float]], label: str) -> str:
        out = [f'<rect class="box" x="{x}" y="40" width="80" height="80" rx="4"/>']
        for dx, op in dots:
            out.append(
                f'<circle class="c2" cx="{x + dx}" cy="80" r="11" fill-opacity="{op}"/>'
            )
        out.append(text(x + 40, 140, label, cls="small"))
        return "".join(out)

    b = [
        text(12, 24, "画素の空間", anchor="start", cls="strong"),
        frame(20, [(18, 1)], "t − Δ"),
        frame(120, [(62, 1)], "t + Δ"),
        arrow(206, 80, 244, 80),
        frame(250, [(18, 0.45), (62, 0.45)], "2 枚の平均"),
        text(362, 88, "≠", cls="big"),
        frame(390, [(40, 1)], "本物の中間 t"),
        text(500, 70, "平均は二重写しになり、", anchor="start", cls="small"),
        text(500, 90, "中間のフレームにならない", anchor="start", cls="small"),
        text(12, 184, "符号の空間", anchor="start", cls="strong"),
        # a curved path: the chord's midpoint m is far from z_t
        '<path class="ln" d="M40 300 Q150 150 260 300" stroke-dasharray="3 4"/>',
        '<line class="ray" x1="40" y1="300" x2="260" y2="300"/>',
        '<line class="ln" x1="150" y1="226" x2="150" y2="300" stroke-width="2.5"/>',
        '<circle class="c1" cx="40" cy="300" r="6"/>',
        '<circle class="c1" cx="260" cy="300" r="6"/>',
        '<circle class="c2" cx="150" cy="226" r="7"/>',
        '<path class="cross" d="M143 300 h14 M150 293 v14"/>',
        text(40, 322, "z(t−Δ)", cls="mono small"),
        text(260, 322, "z(t+Δ)", cls="mono small"),
        text(150, 214, "z(t)", cls="mono small"),
        text(160, 296, "m", anchor="start", cls="mono small"),
        text(142, 262, "‖z(t) − m‖ が大きい", anchor="end", cls="small"),
        text(150, 346, "曲がった軌跡", cls="strong"),
        # a straight path: m falls on z_t
        '<line class="ln" x1="400" y1="300" x2="620" y2="220" stroke-dasharray="3 4"/>',
        '<circle class="c1" cx="400" cy="300" r="6"/>',
        '<circle class="c1" cx="620" cy="220" r="6"/>',
        '<circle class="c2" cx="512" cy="258" r="7"/>',
        '<path class="cross" d="M503 260 h14 M510 253 v14"/>',
        text(400, 322, "z(t−Δ)", cls="mono small"),
        text(620, 242, "z(t+Δ)", cls="mono small"),
        text(500, 236, "z(t) ≈ m", anchor="end", cls="mono small"),
        text(510, 346, "真っすぐな軌跡", cls="strong"),
    ]
    return svg("0 0 660 360", "".join(b), "画素の平均と、符号の空間の中点の模式図")


def fig_tl_gs() -> str:
    """Real data: the pixel PCA's curve (README, V-d3 (a)) on the G-s plane,
    with the closed-form lines."""
    x0, w, y0, h = 70.0, 520.0, 20.0, 280.0
    gmax = 0.8

    def X(s: float) -> float:
        return x0 + s * w

    def Y(g: float) -> float:
        return y0 + h - g / gmax * h

    b = [
        f'<polygon class="shade" points="{X(0)},{Y(0)} {X(1)},{Y(0.5)} {X(1)},{Y(0)}"/>',
        text(X(0.78), Y(0.13), "定常なら届かない", cls="small"),
    ]
    for t in (0, 0.25, 0.5, 0.75):
        b.append(
            f'<line class="gridline" x1="{x0}" y1="{Y(t)}" x2="{x0 + w}" y2="{Y(t)}"/>'
        )
        b.append(text(x0 - 8, Y(t) + 4, f"{t:g}", anchor="end", cls="mono small"))
    for t in (0, 0.25, 0.5, 0.75, 1):
        b.append(text(X(t), y0 + h + 20, f"{t:g}", cls="mono small"))
    b.append(
        f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>'
    )
    b.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}"/>')
    b.append(
        text(
            x0 + w,
            y0 + h + 40,
            "s（その刻みで動く割合）→ 遅いほど左",
            anchor="end",
            cls="small",
        )
    )
    b.append(text(x0 - 56, y0 - 4, "G_mid", anchor="start", cls="mono small"))
    b.append(
        f'<line class="refline" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(0.5)}"/>'
    )
    b.append(
        text(X(0.62), Y(0.31) + 18, "下限の線 G = s/2", anchor="start", cls="small")
    )
    b.append(
        f'<line class="chance" x1="{x0}" y1="{Y(0.75)}" x2="{x0 + w}" y2="{Y(0.75)}"/>'
    )
    b.append(
        text(
            X(0.02), Y(0.75) - 6, "3/4：白い雑音・並べ替え", anchor="start", cls="small"
        )
    )
    b.append(
        f'<line class="chance" x1="{x0}" y1="{Y(0.25)}" x2="{x0 + w}" y2="{Y(0.25)}"/>'
    )
    b.append(
        text(
            X(0.02),
            Y(0.25) - 6,
            "1/4：等速の外挿が copy に勝つ目安",
            anchor="start",
            cls="small",
        )
    )
    ou = " ".join(f"{X(s / 50):.1f},{Y((2 + s / 50) / 4):.1f}" for s in range(51))
    b.append(f'<polyline class="ln c4s" points="{ou}" stroke-dasharray="6 3"/>')
    b.append(
        text(X(0.03), Y(0.5) + 16, "OU：G = (2 + s)/4", anchor="start", cls="small")
    )
    pts = " ".join(f"{X(s):.1f},{Y(g):.1f}" for _, g, s in PCA_CURVE)
    b.append(f'<polyline class="ln c1s" points="{pts}" stroke-width="2.5"/>')
    for lag, g, s in PCA_CURVE:
        b.append(f'<circle class="c1" cx="{X(s):.1f}" cy="{Y(g):.1f}" r="4.5"/>')
    b.append(text(X(0.319), Y(0.589) + 22, "pca 0.8 秒", cls="small accent"))
    b.append(
        text(X(0.936) - 4, Y(0.743) + 22, "25.6 秒", anchor="end", cls="small accent")
    )
    b.append(f'<circle class="c2" cx="{X(0)}" cy="{Y(0)}" r="6"/>')
    b.append(text(X(0) + 12, Y(0) - 22, "0：等速の直線", anchor="start", cls="small"))
    return svg(f"0 0 640 {y0 + h + 50}", "".join(b), "G_mid と s の平面の目盛り")


def fig_tl_whiten() -> str:
    rng = random.Random(21)
    path = []
    u, v = 0.0, 0.0
    for _ in range(26):
        u = 0.85 * u + rng.gauss(0, 0.45)
        v = 0.85 * v + rng.gauss(0, 0.45)
        path.append((u, v))

    def ell(p: tuple[float, float]) -> tuple[float, float]:
        # stretch and rotate: one code's own scale and axes
        a, c = 38 * p[0], 13 * p[1]
        th = math.radians(-25)
        return (
            150 + a * math.cos(th) - c * math.sin(th),
            120 + a * math.sin(th) + c * math.cos(th),
        )

    def circ(p: tuple[float, float]) -> tuple[float, float]:
        return 500 + 32 * p[0], 120 + 32 * p[1]

    def poly(pts: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    b = [
        text(150, 24, "白色化の前（符号ごとに尺度も軸も違う）", cls="strong"),
        '<ellipse class="cloud" cx="150" cy="120" rx="120" ry="40" transform="rotate(-25 150 120)"/>',
        f'<polyline class="ln c1s" points="{poly([ell(p) for p in path])}"/>',
        arrow(290, 120, 400, 120),
        text(345, 104, "白色化", cls="strong"),
        text(345, 148, "参照の平均を引き、", cls="small"),
        text(345, 166, "主成分に回して、", cls="small"),
        text(345, 184, "分散の平方根で割る", cls="small"),
        text(500, 24, "白色化の後", cls="strong"),
        '<circle class="cloud" cx="500" cy="120" r="80"/>',
        f'<polyline class="ln c1s" points="{poly([circ(p) for p in path])}"/>',
        text(500, 228, "どの方向も分散 1。残るのは時間の動きだけ", cls="small"),
    ]
    return svg("0 0 660 245", "".join(b), "白色化の模式図")


def fig_tl_twonn() -> str:
    rng = random.Random(5)
    b = []
    # left: points on a curve (one dimension)
    pts1 = [(30 + 9 * i, 110 + 40 * math.sin(i / 5)) for i in range(28)]
    for x, y in pts1:
        b.append(f'<circle class="cn" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>')
    cx, cy = pts1[14]
    r1 = 9.8
    r2 = 2 * r1
    b.append(f'<circle class="c2" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>')
    b.append(f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="{r1}"/>')
    b.append(
        f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="{r2}" stroke-dasharray="3 3"/>'
    )
    b.append(text(150, 24, "1 次元に並んだ点", cls="strong"))
    b.append(text(150, 200, "2 番目の近傍は 1 番目の約 2 倍の距離", cls="small"))
    b.append(text(150, 220, "μ = r₂ / r₁ が大きい → d̂ は小さい", cls="small"))
    # right: points spread in two dimensions
    pts2 = [(500 + rng.uniform(-95, 95), 110 + rng.uniform(-60, 60)) for _ in range(90)]
    for x, y in pts2:
        b.append(f'<circle class="cn" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>')
    cx2, cy2 = 500.0, 110.0
    d = sorted(((x - cx2) ** 2 + (y - cy2) ** 2) ** 0.5 for x, y in pts2)
    b.append(f'<circle class="c2" cx="{cx2}" cy="{cy2}" r="5"/>')
    b.append(f'<circle class="ring" cx="{cx2}" cy="{cy2}" r="{d[0]:.1f}"/>')
    b.append(
        f'<circle class="ring" cx="{cx2}" cy="{cy2}" r="{d[1]:.1f}" stroke-dasharray="3 3"/>'
    )
    b.append(text(500, 24, "広がった点（高い次元ほど同じ傾向が強まる）", cls="strong"))
    b.append(text(500, 200, "2 番目の近傍も 1 番目とほぼ同じ距離", cls="small"))
    b.append(text(500, 220, "μ が 1 に近い → d̂ は大きい", cls="small"))
    b.append(text(330, 110, "実線 r₁　破線 r₂", cls="small"))
    return svg("0 0 660 235", "".join(b), "TwoNN の模式図")


def fig_tl_classify() -> str:
    x0, w = 360.0, 270.0
    lo, hi, ds = -0.08, 0.08, 0.03

    def X(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * w

    rows = [
        (0.052, 0.009, "c2", "区別できた（大きい側）"),
        (0.006, 0.010, "c3", "小さい（区間が ±Δ* の中）"),
        (0.030, 0.010, "cn", "解像度が足りない（境をまたぐ）"),
        (-0.041, 0.008, "c1", "区別できた（小さい側）"),
    ]
    top, step = 40.0, 40.0
    bottom = top + step * len(rows)
    b = [
        f'<rect class="shade" x="{X(-ds)}" y="{top - 16}" width="{X(ds) - X(-ds)}" height="{bottom - top + 10}"/>',
        text(X(0), top - 22, "±Δ* = ±0.03 の帯", cls="small"),
    ]
    for t in (-0.06, -0.03, 0, 0.03, 0.06):
        b.append(
            f'<line class="gridline" x1="{X(t)}" y1="{top - 16}" x2="{X(t)}" y2="{bottom - 6}"/>'
        )
        b.append(text(X(t), bottom + 12, f"{t:+.2f}" if t else "0", cls="mono small"))
    for i, (v, half, cls, lab) in enumerate(rows):
        y = top + i * step
        b.append(
            f'<line class="whisk {cls}s" x1="{X(v - half)}" y1="{y}" x2="{X(v + half)}" y2="{y}"/>'
        )
        b.append(f'<circle class="{cls}" cx="{X(v)}" cy="{y}" r="5"/>')
        b.append(text(12, y + 5, lab, anchor="start", cls="small"))
    b.append(
        text(
            x0 + w,
            bottom + 30,
            "対の差（符号 − 物差し）。線は ±2SE* の区間",
            anchor="end",
            cls="small",
        )
    )
    return svg(f"0 0 650 {bottom + 40}", "".join(b), "言葉づかいの 3 つの分類の模式図")


def fig_tl_headroom() -> str:
    """Real data: 3/4 minus the pixel PCA's G_mid at each lag, and the largest
    delta V-d2 could inject (README)."""
    x0, w, top, step = 80.0, 330.0, 30.0, 34.0
    vmax = 0.18

    def X(v: float) -> float:
        return x0 + v / vmax * w

    b = []
    bottom = top + step * len(PCA_CURVE)
    for t in (0, 0.03, 0.06, 0.09, 0.12, 0.15):
        b.append(
            f'<line class="gridline" x1="{X(t)}" y1="{top - 10}" x2="{X(t)}" y2="{bottom}"/>'
        )
        b.append(text(X(t), bottom + 16, f"{t:.2f}", cls="mono small"))
    for i, ((lag, g, _), dmax) in enumerate(zip(PCA_CURVE, VD2_MAX_DELTA, strict=True)):
        y = top + i * step
        head = 0.75 - g
        cls = "c1" if lag <= 3.2 else "cn"
        b.append(text(x0 - 8, y + 14, f"{lag:g} 秒", anchor="end", cls="mono small"))
        b.append(
            f'<rect class="{cls}" x="{x0}" y="{y}" width="{X(head) - x0:.1f}" height="18" fill-opacity="0.85" rx="2"/>'
        )
        b.append(f'<path class="cross" d="M{X(dmax):.1f} {y - 3} v24"/>')
        b.append(
            text(
                max(X(head), X(dmax)) + 8,
                y + 14,
                f"{head:.3f}（作れた δ ≤ {dmax:g}）",
                anchor="start",
                cls="mono small",
            )
        )
    b.append(
        f'<line class="refline bad" x1="{X(0.03)}" y1="{top - 14}" x2="{X(0.03)}" y2="{bottom}"/>'
    )
    b.append(text(X(0.03) + 4, top - 16, "Δ* = 0.03", anchor="start", cls="small"))
    b.append(
        text(
            12,
            bottom + 34,
            "棒：3/4 − pca の G_mid（上に動ける余地）。縦線：V-d2 で作れた最大の δ",
            anchor="start",
            cls="small",
        )
    )
    return svg(f"0 0 640 {bottom + 44}", "".join(b), "長い刻みで残る余地")


def fig_tl_spread_schema() -> str:
    b = []
    same = [1.08, 0.94, 1.03, 0.97, 1.0, 0.92, 1.05, 0.99]
    off = [9.0, 0.5, 0.45, 0.4, 0.4, 0.35, 0.3, 0.3]
    for p, (title, vals, note) in enumerate(
        [
            (
                "参照と評価が同じ分布",
                same,
                ["どの方向も分散はほぼ 1", "V/k ≈ 1、最大 ≈ 1、上位 5 方向 ≈ 5/k"],
            ),
            (
                "参照と評価がずれた分布",
                off,
                ["参照で小さく評価で大きい方向が", "引き伸ばされ、V の大半を占める"],
            ),
        ]
    ):
        x0 = 30 + p * 330
        base, unit = 200.0, 40.0
        b.append(text(x0 + 140, 24, title, cls="strong"))
        b.append(
            f'<line class="axis" x1="{x0}" y1="{base}" x2="{x0 + 290}" y2="{base}"/>'
        )
        b.append(
            f'<line class="chance" x1="{x0}" y1="{base - unit}" x2="{x0 + 290}" y2="{base - unit}"/>'
        )
        b.append(text(x0 + 292, base - unit + 4, "1", anchor="start", cls="mono small"))
        for i, v in enumerate(vals):
            hgt = min(v * unit, 150)
            cls = "c2" if v > 2 else "c1"
            b.append(
                f'<rect class="{cls}" x="{x0 + 10 + i * 34}" y="{base - hgt}" width="24" height="{hgt}" fill-opacity="0.85" rx="2"/>'
            )
        if max(vals) * unit > 150:
            b.append(text(x0 + 22, base - 156, "（省略）", cls="small"))
        for k, s in enumerate(note):
            b.append(text(x0 + 140, base + 24 + k * 18, s, cls="small"))
    b.append(
        text(
            330,
            262,
            "棒は、白色化した評価の分散を、大きい方向から順に並べたもの（説明のための例）",
            cls="small warn",
        )
    )
    return svg("0 0 660 275", "".join(b), "白色化した評価の分散の模式図")


def fig_tl_spread_data() -> str:
    """Real data: the spread diagnosis of each code (``judge_geom.json``) and
    the failure example (README), on log axes with the provisional bounds."""
    rows = _spread_rows()
    axes = [
        ("V_over_k", "V/k", 0.3, 12.0, [0.5, 1, 2, 5, 10], 1.0),
        ("var_max", "方向の分散の最大", 0.8, 300.0, [1, 3, 10, 30, 100, 300], 1.0),
        (
            "top_share",
            "上位 5 方向の割合",
            0.05,
            1.0,
            [0.05, 0.1, 0.2, 0.4, 1.0],
            5 / 64,
        ),
    ]
    x0, w = 150.0, 460.0
    b = []
    y = 30.0
    for key, label, lo, hi, ticks, ideal in axes:

        def X(v: float, lo: float = lo, hi: float = hi) -> float:
            return (
                x0
                + (math.log10(v) - math.log10(lo))
                / (math.log10(hi) - math.log10(lo))
                * w
            )

        b.append(text(12, y + 4, label, anchor="start", cls="strong"))
        b.append(text(x0 - 8, y + 30, "自分の潜在", anchor="end", cls="small"))
        b.append(text(x0 - 8, y + 58, "h_mean", anchor="end", cls="small mono"))
        for t in ticks:
            b.append(
                f'<line class="gridline" x1="{X(t):.1f}" y1="{y + 12}" x2="{X(t):.1f}" y2="{y + 72}"/>'
            )
            b.append(text(X(t), y + 88, f"{t:g}", cls="mono small"))
        b.append(
            f'<line class="chance" x1="{X(ideal):.1f}" y1="{y + 10}" x2="{X(ideal):.1f}" y2="{y + 74}"/>'
        )
        blo, bhi = BOUNDS[key]
        for bv in (blo, bhi):
            if bv is not None:
                b.append(
                    f'<line class="refline bad" x1="{X(bv):.1f}" y1="{y + 8}" x2="{X(bv):.1f}" y2="{y + 76}"/>'
                )
        b.append(text(X(ideal), y + 6, "同じ分布", cls="small"))
        for place, code, v in rows:
            yy = y + (26 if place == "own" else 54)
            flag = _flagged(v, key, *BOUNDS[key])
            cls = "c2" if flag else ("cn" if code in YARD_CODES else "c1")
            b.append(
                f'<circle class="{cls}" cx="{X(v[key]):.1f}" cy="{yy}" r="4" fill-opacity="0.75"><title>{geom_name(code)}: {v[key]:.3g}</title></circle>'
            )
        xo = X(OU_MIX_30[key])
        b.append(
            f'<path class="cross c2s" d="M{xo - 5:.1f} {y + 35} l10 10 M{xo + 5:.1f} {y + 35} l-10 10"/>'
        )
        y += 112
    b.append(
        text(
            x0,
            y - 4,
            "灰：物差し　青：学習した符号　橙：仮の境の外　×：失敗の形の例（ou_mix_30）　赤い破線：仮の境",
            anchor="start",
            cls="small",
        )
    )
    return svg(f"0 0 640 {y + 6}", "".join(b), "白色化した評価の分散の診断の値")


def tl_whatif() -> str:
    """A table: which codes each alternative bound would flag
    (``judge_geom.json``'s ``spread``, the yardsticks, the VAEs and step
    60000)."""
    rows = _spread_rows()
    alts = [
        ("V/k が [0.8, 1.25] の外", "V_over_k", 0.8, 1.25, False),
        ("V/k が [0.5, 2] の外", "V_over_k", 0.5, 2.0, True),
        ("最大が 5 を超える", "var_max", None, 5.0, False),
        ("最大が 10 を超える", "var_max", None, 10.0, True),
        ("最大が 20 を超える", "var_max", None, 20.0, False),
        ("上位 5 方向が 0.25 を超える", "top_share", None, 0.25, False),
        ("上位 5 方向が 0.3 を超える", "top_share", None, 0.3, False),
        ("上位 5 方向が 0.4 を超える", "top_share", None, 0.4, True),
    ]
    body = []
    for label, key, lo, hi, now in alts:
        hits = [(p, c, v[key]) for p, c, v in rows if _flagged(v, key, lo, hi)]
        if hits:
            names = "、".join(
                f"{geom_name(c)}（{'h_mean' if p == 'h_mean' else '自分'}、"
                f"{x:.{2 if key == 'var_max' else 3}f}）"
                for p, c, x in sorted(hits, key=lambda h: -h[2])
            )
        else:
            names = "なし"
        mark = ' <span class="badge wait">仮の境</span>' if now else ""
        body.append(
            f'<tr><td>{label}{mark}</td><td class="n">{len(hits)}</td><td>{names}</td></tr>'
        )
    return (
        '<div class="tbl"><table><thead><tr><th>境</th><th>数</th>'
        "<th>注が付く符号（読む場所、値）</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


FIGURES: dict[str, Callable[[], str]] = {
    "tl_double": fig_tl_double,
    "tl_gs": fig_tl_gs,
    "tl_whiten": fig_tl_whiten,
    "tl_twonn": fig_tl_twonn,
    "tl_classify": fig_tl_classify,
    "tl_headroom": fig_tl_headroom,
    "tl_spread_schema": fig_tl_spread_schema,
    "tl_spread_data": fig_tl_spread_data,
    "tl_whatif": tl_whatif,
    "jepa": fig_jepa,
    "collapse": fig_collapse,
    "g_bars": fig_g_bars,
    "timeline": fig_timeline,
    "pairwin": fig_pairwin,
    "center": fig_center,
    "controls": fig_controls,
    "swap": fig_swap,
    "transplant": fig_transplant,
}


def build() -> None:
    src = TEMPLATE.read_text(encoding="utf-8")

    def sub(m: re.Match[str]) -> str:
        return FIGURES[m.group(1)]()

    html = re.sub(r"<!--SVG:(\w+)-->", sub, src)
    missing = re.findall(r"<!--SVG:(\w+)-->", html)
    assert not missing, missing
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html.encode('utf-8'))} bytes)")


def extract(path: Path) -> None:
    src = TEMPLATE.read_text(encoding="utf-8")
    blocks = [
        {"i": i, "tag": m.group(1), "html": m.group(3)}
        for i, m in enumerate(BLOCK.finditer(src))
    ]
    path.write_text(json.dumps(blocks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(blocks)} blocks -> {path}")


def apply(path: Path) -> None:
    edited = {b["i"]: b["html"] for b in json.loads(path.read_text(encoding="utf-8"))}
    src = TEMPLATE.read_text(encoding="utf-8")
    count = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal count
        i = sub.i  # type: ignore[attr-defined]
        sub.i += 1  # type: ignore[attr-defined]
        if i in edited and edited[i] != m.group(3):
            count += 1
            return f"<{m.group(1)}{m.group(2) or ''}>{edited[i]}</{m.group(1)}>"
        return m.group(0)

    sub.i = 0  # type: ignore[attr-defined]
    TEMPLATE.write_text(BLOCK.sub(sub, src), encoding="utf-8")
    print(f"applied {count} edited blocks")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "extract":
        extract(Path(sys.argv[2]))
    elif cmd == "apply":
        apply(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown command {cmd}")
