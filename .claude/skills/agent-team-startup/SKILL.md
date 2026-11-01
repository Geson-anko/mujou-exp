---
name: agent-team-startup
description: orchestrator（メインエージェント）が implementation-planner →（任意 spec-test-author）→ plan-implementer → code-reviewer（＋algorithm-reviewer）⇄ code-simplifier を統括する標準フロー、モデル構成（plan-implementer は sonnet、他は inherit + effort で差別化）、委譲と並列化の判断基準。ユーザーが「エージェントチームで進めて」「並列で」と明示したときだけ参照する。指定がない場合の既定は skill solo-dev-cycle。
---

# エージェントチーム起動手順

ユーザーから「エージェントチームで」「並列で」と明示された場合の進行手順。指定がなければチームは起動せず、規模にかかわらず skill [solo-dev-cycle](../solo-dev-cycle/SKILL.md) で単独に進める（CLAUDE.md「エージェント運用」）。

詳細な並列化パターン・Failure mode・起動例は [reference.md](reference.md) に分離してある。判断に必要になった時点で読む。

## 前提

- `settings.json` に `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` が設定済み（必須）
- メインエージェントが `orchestrator` の役割（`.claude/agents/orchestrator.md`）を担う。チーム運用を常用するセッションは `claude --agent orchestrator` で起動してもよい
- 各 agent は `.claude/agents/` 配下に定義済み（orchestrator, implementation-planner, spec-test-author, plan-implementer, code-reviewer, algorithm-reviewer, code-simplifier）
- 中間メモは `memory/agents/<agent-name>/<task-slug>.md` に書く

## 役割とモデル / effort 構成

`plan-implementer` は `model: sonnet`（計画が確定した後の実装は速度・コストを優先する）。他の agent はセッションのモデルを継承し（`model: inherit`）、深さの差別化は **`effort`** で行う。呼び出し時に Agent tool の `model` / `effort` パラメータで一時的に上書きもできる（frontmatter より優先）。

| agent                  | effort       | 役割                                                                     |
| ---------------------- | ------------ | ------------------------------------------------------------------------ |
| orchestrator           | セッション値 | メインエージェント。統括・委譲・レビュー裁定・合流検証・ユーザーとの対話 |
| code-reviewer          | high         | レビュー（仕様準拠・バグ・実験の妥当性・規約・テスト品質）。コードは修正しない |
| algorithm-reviewer     | high         | 更新手法の数理・ノード間の局所性・非同期性・比較の公平さの監査。コードは修正しない |
| implementation-planner | medium       | 要件の仕様化・計画書作成（コードは書かない）                             |
| plan-implementer       | medium       | 計画に基づく実装・テスト・グリーン化                                     |
| spec-test-author       | medium       | 仕様テスト（`tests/` のみ、`experiments/` は触らない）                   |
| code-simplifier        | medium       | 公開 IF 維持の簡素化 + docstring / 実験の README 同期（レビューはしない） |

orchestrator に `effort` を設定していないのは、frontmatter の effort がセッション値を上書きし、ユーザーが `/effort` で変えた値が反映されなくなるため。セッション既定は `settings.json` の `effortLevel` で与える。

**effort の再調整**: 上表は初期値で、固定値ではない。深さが足りなければ 1 段ずつ上げ、`xhigh` / `max` は効果を確かめられたときだけ使う。thinking を減らすには、「考えすぎるな」と指示するより effort を下げる方が確実。

## 標準サイクル

```
0. orchestrator（メイン）
   - 要件確認・ブランチ作成・以降の統括。experiments/ tests/ は自分で編集しない
   - ユーザーへの質問はここだけが行える（サブエージェントは AskUserQuestion を持たない）

1. implementation-planner  [複数モジュールにまたがるとき。1 モジュールに収まるなら orchestrator が自分で計画]
   - 要件を仕様化し、計画書を memory/agents/implementation-planner/<task>.md に書く
   - 実験なら仮説と判定基準、mujou に足すものの切り分けを含める
   - 確認事項があれば計画書に列挙し、orchestrator が中継する
   - mujou に足すものがあれば、orchestrator が skill mujou-upstream の手順で別に進める

2. spec-test-author  [任意 — 仕様 first フロー時のみ]
   - 計画書を読み、tests/ 配下にテスト（生きた仕様書）を書く
   - 実験コード（experiments/）は触らない
   - 期待される失敗と仕様根拠を memory/agents/spec-test-author/<task>.md に記録
   - plan-implementer と並列実行可能（reference.md パターン A）

3. plan-implementer
   - 計画書（必要なら spec-test-author のノートも）を読み、実装＋型＋lint をグリーン化
   - spec-test-author が engagement されていればテストは編集しない（実装だけで通す）
   - 計画外の判断は memory/agents/plan-implementer/<task>.md に記録

4. code-reviewer（学習則・局所性・非同期性に触れる変更なら algorithm-reviewer も並列）
   - diff と計画書を突き合わせ、verdict（approve / request-changes）を
     memory/agents/code-reviewer/<task>.md に出す
   - 見つけた指摘は確信度を添えて全件報告する（ふるいにかけない）
   - orchestrator が裁定：must-fix → 3 に差し戻し／should-fix → 5 へ／
     誤検出 → 却下（理由を orchestrator のノートに記録）
   - must-fix が残る間は verdict が request-changes になり、3 ⇄ 4 を繰り返す

5. code-simplifier
   - 公開 IF を保持しつつ内部を簡素化（reviewer の should-fix も対応対象）
   - 変更で古くなった docstring / 実験の README・PLAN を同期する
   - 内容を memory/agents/code-simplifier/<task>.md に記録
   - 大きく書き換えた場合は 4 で再レビュー
```

approve（should-fix は残ってよい）の後に 5 を行う。5 で大きく書き換えたら 4 で再レビューする。最後に orchestrator が最終検証してコミットする。本番の学習の実行と結果の記録は、コードの approve の後に orchestrator が管理する。

## 委譲するかどうか

チームを明示された後でも、個々の仕事が委譲に見合うかを判断する。委譲すると、サブエージェントは文脈を再構築して探索し直し、報告を返す。orchestrator はその報告を読み直す。この往復のコストを上回る利得があるときだけ委譲する。

- **チームを起動する**：複数ファイル・複数モジュールにまたがる実装や調査。独立して並列に進められる作業がある
- **単独委譲で足りる**：対象が明確な 1 モジュールの実装（`plan-implementer` 単独）
- **委譲しない**：数回のツール呼び出しで orchestrator 自身が終えられる仕事。自分の作業の検証・ダブルチェック（検証は orchestrator のループ内で行う）。1〜数モジュールに収まる変更を通しで仕上げる場合は skill [solo-dev-cycle](../solo-dev-cycle/SKILL.md)

1 体で足りる仕事を分割して複数体に割り振らない。spawn 数は低く保つ。

## spec-test-author を挟むかの判断

| 状況                                                          | 推奨                                     |
| ------------------------------------------------------------- | ---------------------------------------- |
| 公開 API の仕様が明確で、振る舞いの契約を先に固めたい         | **挟む**                                 |
| 学習則・optimizer・指標の数値的オラクルを先に固めたい         | **挟む**                                 |
| 既存仕様への bug fix で、再現テスト → 修正の流れにしたい      | **挟む**（再現テストを spec として固定） |
| 仕様が探索的で、実装しながら詰める必要がある                  | 挟まない（plan-implementer が一括）      |
| trivial な変更・1 関数追加・使い捨ての解析スクリプト          | 挟まない                                 |

挟まない場合は `plan-implementer` がテストと実装を同時に書く。

## 並列化の基本条件

`tests/` と `experiments/` は disjoint なので、`spec-test-author` と `plan-implementer` は同時起動できる。前提条件は次のとおり。

- planner の「公開インターフェース案」が**シグネチャレベル**（関数名・引数・戻り値型）で確定していること
- GPU など占有リソースを奪い合う長時間の学習を同時に走らせないこと
- `tests/<name>/` の共通 helper の更新は spec-test-author 側に寄せること

モジュール分割による並列、並列時のファイル分離規約、避けるべきケースは [reference.md](reference.md) を参照。並列発火は 1 メッセージにまとめる（skill `maximize-parallels`）。

## チェックポイント

orchestrator が各段階で確認する。

| タイミング              | 確認事項                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| planner 完了時          | 計画書に「判定基準／公開 IF 案／実装ステップ／テスト観点／リスク」が揃っているか。確認事項があれば中継したか |
| spec-test-author 完了時 | テストが `tests/` 配下のみに書かれているか、3rd-party モックが無いか、仕様根拠の対応表があるか          |
| implementer 完了時      | `just run` がグリーン。spec-test-author 引継ぎ時はテスト未編集                                          |
| reviewer 完了時         | verdict が明記され、全 must-fix に根拠と確信度が添えてあるか                                            |
| simplifier 完了時       | 公開 IF 不変・全テスト通過・ドキュメントが現状コードと整合                                              |
| 並列完了時              | 全並列分マージ後に 1 回 `just run` を通す                                                               |

学習則・局所性・非同期性に触れる変更では、`code-reviewer` と並列に `algorithm-reviewer`（effort high）を起動する。

## 参照

- 委譲せず単独で回す版：skill [solo-dev-cycle](../solo-dev-cycle/SKILL.md)
- 並列化パターン・Failure mode・起動例：[reference.md](reference.md)
- tool 呼び出しレベルの並列化：skill `maximize-parallels`
- mujou への機能追加：skill `mujou-upstream`
- 各 agent の詳細：`.claude/agents/<name>.md`
