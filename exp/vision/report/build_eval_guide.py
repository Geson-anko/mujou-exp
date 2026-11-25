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


FIGURES: dict[str, Callable[[], str]] = {
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
