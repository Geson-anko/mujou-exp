---
name: refactor-conventions
description: mujou-exp の実験コードのコーディング・リファクタリング規約集。数式と 1 対 1 に読めるコード、テンソル shape の明示、カプセル化（_ prefix）、実験のパッケージ構成、設定と再現性、型の書き方、エラーハンドリング、リファクタ技法の適用/非適用。コードを書き換える・テストを追加するときに参照する。
---

# コーディング・リファクタリング規約

CLAUDE.md の開発原則を補完し、判断基準と具体例を示す。テスト方針は skill `testing-strategy` が正典。`mujou/` の中のコードには mujou の規約（`mujou/.claude/skills/refactor-conventions/SKILL.md`）が適用される。

## 数式との対応

- 学習則のコードは出典の記号に揃える（PC-ALM なら `mujou/.claude/skills/pc-alm/SKILL.md` の `a`, `r`, `lam`, `eps`, `e_out` など）。docstring か直前のコメントに対応する式（`PLAN.md` の式番号でもよい）を書く
- 1 つの式を 1 つの関数またはまとまった数行に対応させる。複数の式を 1 行に畳んで速くする最適化は、効果を測ってから行い、そのときも式との対応をコメントで残す
- 誤差の向き（$r_i = h_i - \sigma(a_i)$、$e_L = y - \hat y$）を変数名や docstring で明示する

## テンソル

- 公開関数の docstring にテンソル引数・戻り値の shape を書く（例: `h: (batch, n_in)`）
- バッチ次元は先頭に置く
- ノードの状態の in-place 更新はそのノードだけが行う（`mujou/.claude/skills/locality-and-async/SKILL.md`）

## 実験のパッケージ構成

- 1 実験 1 パッケージ（`experiments/<name>/`）。実行は `uv run python -m experiments.<name>.<script>`
- 実験をまたいで使うコード・特定の実験に閉じないコードは、`experiments/` に共通モジュールを作らず mujou に足す（skill `mujou-upstream`）
- 実験の都合だけの暫定物は、その実験のパッケージの中に置く
- 他の実験のパッケージを import しない

## 設定と再現性

- ハイパーパラメータ・seed・パスは設定（`config.py` などの dataclass）に集め、コードの奥に定数を埋めない
- 乱数は seed から作った `torch.Generator` を明示的に渡す。グローバルな乱数状態（`torch.manual_seed`）に頼るのは、モジュールの初期化など Generator を渡せない箇所だけにする
- 結果の記録（実験の README）に seed・設定・mujou の submodule の SHA を残す。checkpoint・ログ・プロットは git の外（`runs/`、`outputs/`、`logs/`）に書く
- 既定の設定値を変えると README の数字が再現しなくなる。変えたら README も直す

## カプセル化

- クラスの内部実装と `__init__` で設定する属性は原則 `_` prefix
- 外部から参照する必要がある属性のみ public にする
- 「テストから参照したい」は public 化の理由にならない。公開 API 経由でテストする
- 1 つのモジュールの中だけで使う関数には `_` を付ける
- mujou の `_` 属性・`_` 関数に実験から触らない。要るなら mujou に公開 API を足す

## 型

- pyright（standard）を通す。`Any` と `# type: ignore` は理由をコメントで添えたときだけ使う
- 型エイリアスは `type X = ...` 構文を使う
- 変換・検証付きの生成は `@classmethod` ファクトリに置く

## エラーハンドリング

- 検証は境界（設定・外部データ・ストリーム）でだけ行う。内部で起こり得ない状態の防御コードは書かない
- 設定や shape の誤りは、何が期待され何が来たかを含むメッセージで早く失敗させる
- NaN / Inf は黙って握りつぶさない。学習が発散したら止めて、その事実を記録する

## リファクタリング技法

### 適用する

- 不要なネストの平坦化（early return / guard clause）
- 1 箇所でしか使わない関数・クラスの解消
- 意図が読めない命名（`data`, `tmp`, `helper`）の改善
- wrapper / pass-through メソッドの除去
- 重複の抽出（3 度現れたら、が目安）

### 適用しない

- 仮想的な将来要件のための抽象化（「次の実験でも使うかもしれないから」など）
- 「綺麗に見える」だけの分割
- 公開 IF を変えるリファクタ（別タスクで合意してから）
- 数式との対応が読めなくなる「賢い」書き換え
- 乱数の消費順や既定値を変えて、記録済みの結果を再現できなくする書き換え

## 検証

```bash
just run   # format → test → type
```

すべてパスして完了。lint warning・型エラーを放置しない。
