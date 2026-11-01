# エージェントチームの並列化パターンと Failure mode

skill `agent-team-startup` の補助資料。判断に必要になった時点で読む。

## 並列化パターン A：spec-test-author × plan-implementer

`tests/` と `experiments/` は disjoint なので、両 agent は同じ計画書を入力に同時起動できる。仕様 first フローでも、2 つを順番に実行する必要はない。

```
implementation-planner
        ↓
   ┌────┴────┐
   ↓         ↓
spec-test-author  plan-implementer   ← 並列起動（同じ計画書を入力に）
   ↓         ↓
   └────┬────┘
        ↓
   合流チェック：just test で、テストが実装に対して期待通り通る／落ちるか確認
        ↓
   code-reviewer →（approve まで反復）→ code-simplifier
```

前提条件は SKILL.md の「並列化の基本条件」を参照。

**合流時の処理**

1. `just format && just type` を通す
2. `just test` を流し、spec-test-author の意図通りに pass / fail しているか確認する
3. 不整合（implementer が IF を変えた、spec-test-author が観点を取りこぼした等）があれば該当 agent を再呼び出しする
4. 全 pass になったら code-reviewer へ

**このパターンを避けるケース**

- 公開 IF がシグネチャレベルまで決まっていない（探索的実装）
- planner の計画が抽象的で、両 agent が異なる解釈をするおそれがある
- 長時間の学習テストが支配的で、両者が同じ GPU を必要とする

## 並列化パターン B：モジュール分割

- **独立したモジュール・機能**ごとに implementer / spec-test-author を分割する
- **インターフェース境界が明確**で、互いの実装詳細に依存しないときに使う
- 例：`experiments/<name>/optim.py` を 1 セット、`experiments/<name>/video.py` を別セットで並行
- パターン A と組み合わせ、各モジュール内でさらに spec-test-author × plan-implementer の並列も可能
- レビューは実装完了後にモジュール単位で並列起動できる

## 並列化を避けるケース（共通）

- インターフェース未確定で並行実装すると衝突するとき
- データフロー上で順序依存があるとき
- `tests/<name>/` の共通 helper を複数 spec-test-author が同時編集するとき（helper の追加は逐次）
- 同じ GPU を占有する学習を同時に走らせることになるとき
- 1 つの中規模タスクを無理に分割することになるとき（並列化は、互いに独立した大きめの作業単位に使う）

## 並列実行の運用

- 並列 implementer は `memory/agents/plan-implementer/<task>-<instance>.md` 形式でファイルを分離する
- 並列 spec-test-author も `memory/agents/spec-test-author/<task>-<instance>.md` で同様に分離する
- 他の implementer / spec-test-author に影響する IF 変更は、同ファイルに「IF 変更通知」セクションで明記する
- 並列 simplifier / reviewer も同様にファイルを分離する
- 並列発火は 1 メッセージにまとめる（skill `maximize-parallels`）

## Failure mode と対処

| 状況                                                | 対処                                                                |
| --------------------------------------------------- | ------------------------------------------------------------------- |
| planner の計画が曖昧                                | 計画書の「確認事項」をユーザーに中継してから先へ進む                |
| spec-test-author が仕様の欠落に気づいた             | planner を再呼出し、計画書を更新してから再開                        |
| spec-test-author 完了後、テストが落ちる（想定済み） | plan-implementer に引き継いで実装で通す                             |
| implementer が「テストが間違っている」と判断        | テストは編集せず spec-test-author に差し戻し、仕様根拠を再確認      |
| implementer がテスト失敗（spec-test-author 不在時） | 再実装、または計画書の見直し（planner 再呼び出し）                  |
| reviewer の must-fix に implementer が異議          | orchestrator が計画書を根拠に裁定。仕様が曖昧なら planner 再呼出    |
| reviewer ⇄ implementer の往復が 2 回を超える        | orchestrator が停止し、ユーザーに判断を仰ぐ                         |
| simplifier が「これ以上簡素化できない」と判断       | ドキュメント同期だけ行い、最終検証へ進む                            |
| 並列 implementer で IF 衝突発覚                     | 並列を中断し、planner で IF を再合意                                |
| サブエージェントが質問を返してきた                  | orchestrator がユーザーに中継する（サブエージェントは直接聞けない） |

## 起動の流れ（例）

ユーザー指示「エージェントチームで continuous_stream に新しい optimizer の対照を足して」に対し、orchestrator は次の順に進める。

1. 要件を確認し、ブランチを作成する
2. `implementation-planner` を起動して計画策定（1 モジュールに収まるなら orchestrator が自分で計画する）
3. 計画を確認する。計画書に「確認事項」があればユーザーに中継する
4. （仕様 first なら）**spec-test-author と plan-implementer を 1 メッセージで並列起動**（パターン A）
    - 計画書が抽象的なら spec-test-author を先行させ、IF 確定後に plan-implementer を起動する
5. 合流時に `just run` で整合確認
6. `code-reviewer` を起動し、報告された指摘を裁定する（must-fix → implementer 差し戻し）
7. approve 後に `code-simplifier` を起動する（should-fix の対応と docstring / README 同期）。大きく書き換えたら 6 で再レビュー
8. 最終検証 → コミット

各 agent 起動時は `subagent_type` で指定し、必要なコンテキスト（計画書パス、対象ファイル、前段ノートのパス）を prompt に含める。並列発火時は 1 メッセージにまとめる（skill `maximize-parallels`）。
