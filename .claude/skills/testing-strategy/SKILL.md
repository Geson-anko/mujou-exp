---
name: testing-strategy
description: mujou-exp の実験コードのテスト戦略。理論から導いた数値的オラクル（有限差分・BP 勾配との一致・退化・手計算できる小さな系）を最優先し、乱数と実行のスケジュールを固定して決定的に保つ。tests/<name>/ ミラーレイアウト、書く/書かないリスト、slow / cuda マーカー、モック方針。テストコードを書く・壊れたテストを調査する・pytest 周りを設定する前に読む。
---

# mujou-exp テスト方針

## 哲学

実験の結論は、実験コードが意図どおりに動いていることを前提にする。テストは「動いた」ではなく「**理論どおりに動いている**」ことを保証する。学習則・損失・optimizer・評価指標は、符号や係数を 1 つ間違えても、それなりに損失が下がって見えることがある。損失が下がることだけを見るテストは、こうした誤りを見逃し、間違った結論を README に残す。

mujou 自身の振る舞いは mujou のテスト（`cd mujou && just test`）が保証する。ここでは mujou を再テストせず、実験が足したものをテストする。

## オラクルの優先順位

1. **理論から導いた厳密な期待値**（最優先）
    - **有限差分**: 手書きの勾配・更新量を、目的関数の数値微分と比べる（float64 で 1e-6 程度）
    - **BP との一致**: PC-ALM の平衡点での重み更新が BP の負勾配と一致する、など（`mujou/.claude/skills/pc-alm/SKILL.md` 4 節）。BP の基準（`bp.py` など）は同じモジュール・同じ初期重みで比べる
    - **退化**: パラメータを特別な値にすると既知の手法に一致する（例: optimizer の係数 0 で SGD に一致する、$\alpha = 0$ で PC に一致する）
    - **手計算できる小さな系**: スカラーや 2 層の線形ネットで、期待値を手で出す
2. **ダイナミクスと指標の性質**: 不動点への収束、不変量（直交性・正規化）の保存、同じ seed で同じ結果、データの生成が指定した統計に従う
3. **学習の結果**: 小さな合成タスクで損失が下がる・精度が出る。最後の砦であって、これだけで正しさを主張しない。時間がかかるものは `@pytest.mark.slow`

## 基本原則

- 公開インターフェースと観測可能な振る舞いをテストする。内部実装の詳細はできるだけ見ない
- 必要十分なテストだけを書く。網羅率を稼ぐためのテストは書かない
- **決定性**: 乱数は `torch.Generator().manual_seed(...)` で固定する。mujou の実行は `mujou.testing` の仮想タイマーで手で回し、スケジュールをテストで明示する。たまに落ちるテストを作らない
- **数値精度**: 理論値との比較は float64 で行い、許容誤差は根拠のある値にする。符号ミスを見逃すほど緩い `atol` にしない
- テストは `class TestXxx:` に集約する。テスト関数に戻り値の型アノテーションは不要
- 複数パラメータは `@pytest.mark.parametrize`

## テストレイアウト

`tests/<name>/` は `experiments/<name>/` を 1 対 1 でミラーリングする（1 source 1 test ファイル）。`tests/<name>/__init__.py` を置く。

```
experiments/<name>/<module>.py  ↔ tests/<name>/test_<module>.py
```

ミラーの外に置くのは次のものだけ。

- `tests/test_claude_hooks.py`: `.claude/hooks/` を検証する
- `tests/test_mujou_install.py`: mujou が editable で入り、import できることを確かめる

本番の学習を回すだけのスクリプト（`run.py` の main など）は、小さな設定で最後まで走ることだけを確かめれば足りる。

## マーカー

- `--strict-markers` が有効。登録済みは `slow` と `cuda`。新しいマーカーは `pyproject.toml` の `[tool.pytest.ini_options].markers` に登録してから使う
- `slow`: 収束・学習を伴う長いテスト。`just test-fast` で除外、`just test` / `just run` では実行する
- `cuda(n=1)`: CUDA デバイスが `n` 個以上ないと skip する

## 何をテストするか / しないか

### 書く

- 正常系、異常系（設定や shape の不一致など境界での検証）、エッジケース（深さ 1、幅 1、バッチ 1）
- 上記オラクルによる数値検証
- 設定の読み書き（JSON との往復など、結果の再現に使うもの）
- 例外メッセージは **substring** で検証（`assert "expected" in str(exc.value)`）

### 書かない

- mujou の振る舞いの追試（mujou のテストの担当）
- 継承・import 可能性・定数 literal の追試
- getter / setter ラウンドトリップ
- framework / stdlib（テンソルライブラリ自身の演算）の動作追試
- 例外メッセージの完全一致
- モック戻り値をそのまま検証するテスト

## モック

- 原則使わない。数値計算は実データ・実テンソルで検証する
- 内部の private 関数はモックしない（リファクタのたびに壊れ、振る舞いは何も保証しない）
- 3rd-party のテンソルライブラリと mujou はモックしない
- 外部データ（実動画のデータセットなど）が要るテストは、同じ形の小さな合成データで置き換える。ダウンロードを伴うテストは書かない

## 関連

- 理論: `mujou/.claude/skills/pc-alm/SKILL.md` / 実行モデル: `mujou/.claude/skills/locality-and-async/SKILL.md`
- 規約: skill `refactor-conventions`
