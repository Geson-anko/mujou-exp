# optuna の調整役（`tools/sweep/`）

optuna の study を 1 つ受け持ち、trial の run をフリートのキューに入れ、run の評価を読んで optuna に報告する常駐プロセス。std-00 のコンテナで動く。設計は `memory/agents/orchestrator/design-final.md` の 5 節（初版で省いたものは `design-decisions.md`）。フリートとのやりとりは NAS のファイル（`queue/`、`claimed/`、`live/`、`done/`、`runs/`）と `exp.lib.fleet`（`jobs`、`queue`）だけで行う。ファイルの形は `exp/lib/fleet/README.md` にある。

学習の venv とは別の uv プロジェクトにしてある（Python 3.14 の GIL あり版、`optuna==5.0.0`、`hydra-core==1.3.8`）。optuna・numpy・sqlalchemy は学習のプロセスに入らない。

## コマンド

```bash
just sweep run exp/video/studies/m14_s1_big_trace_bp.toml      # 調整役を起動する（std-00 で切り離して動く）
just sweep status m14-s1-big-trace-bp-v1                       # trial の数、上位 5 つ、走っている trial
just sweep confirm m14-s1-big-trace-bp-v1 --top 3 --learners bp,pc --seeds 0,1,2 \
    --res 4090 --batch m14-s1-confirm-v1                        # 3 x 2 x 3 = 18 本
just sweep bracket m14-s1-big-trace-bp-v1 --best --param optim.lr --factors 0.5,1,2 \
    --learners bp,pc --seeds 0,1,2 --res 4090 --batch m14-s1-bracket-v1
```

Mac で打ったコマンドはすべて std-00 のコンテナに転送され、`code/latest` の `tools/sweep` で動く。venv は `<root>/venv/sweep-<lock の hash>` で、初回に `uv run` が作る。

- `run`: study ごとに 1 プロセス。すでに動いていれば pid を出して終わるので、何度打ってもよい。ログは `<root>/optuna/<name>.log`、storage は `<root>/optuna/<name>.db`（std-00 のローカルの sqlite）。1 時間ごとと終了時に NAS の `studies/<name>/<name>.db` へ写す。`studies/<name>/RUNNING` は study が終わるまで残る。止めるときは `<root>/optuna/<name>.pid` の pid に SIGTERM を送る。次の `run` で続きから再開する。調整役自身のコードは起動したときの `code/latest` なので、その snapshot を `RUNNING` の `coordinator_code` と study の user attribute `coordinators`（起動ごとに 1 件）に残す
- `confirm` と `bracket` は `phase=confirm` の run をキューに入れる。clean で mujou を push 済みの snapshot だけを受け付け、資源の種類は 1 つに固定する（`any_gpu` は拒む。study の `res` が `any_gpu` なら `--res` で決める）。snapshot は既定で study のものを使う。`--snap` で別の clean な snapshot を選べる。優先度は 20 で、探索（60）より先に走る。`--dry` で投入せずに一覧だけを出す
- `--seeds` に探索の seed（100、101）が入っていると拒む。run の hash は `phase` を含まないので、探索の run の重複とみなされて飛ばされるため
- `confirm --top k` は parameter の同じ trial を 1 つに数え、異なる parameter の上位 k 個を確かめる
- `--res` で GPU の種類を替えても、study の `gpu_mem` はそのまま使う。`cpu` に替えたときだけ 0 にする
- 投入は `fleet submit` と同じ経路を通る。study の snapshot の venv で `exp.video.compose` を呼び、`jobs.make_job` で job を作る。同じ run_id が queued、running、completed、diverged のどれかなら `duplicate` として飛ばす（batch をまたいで判定する）。そのため bracket の ×1 は confirm の run と同じものになる

## study の TOML（`exp/video/studies/*.toml`）

```toml
[study]
name = "m14-s1-big-trace-bp-v1"      # batch の名前も兼ねる
overrides = ["+experiment=m14/s1", "learner=bp", "dynamics=big", "dynamics.trace=true", "phase=explore"]
seeds = [100, 101]                   # 探索の seed。trial は全 seed の平均で比べる
parallel = 12                        # 同時に走る trial の数（run の数はこれ x seed の数）
max_trials = 120
res = "any_gpu"                      # 任意: ram, gpu_mem, prio(60), batch, min_resource(300),
                                     #       reduction_factor(3), n_startup_trials(16), sampler_seed(0)
[space]
"optim"          = {categorical = ["sgd", "adamw", "graft_adam"]}
"optim.lr"       = {float = [0.001, 0.3], log = true, when = "optim != 'adamw'"}
"optim.lr@adamw" = {float = [1e-5, 1e-2], log = true, when = "optim == 'adamw'"}

[[enqueue]]                          # 最初に試す trial（M13 の最良）
"optim" = "sgd"
"optim.lr" = 0.03
```

- `[space]` の名前は optuna の parameter 名で、`@` より前が hydra の override のキーになる。同じキーに範囲の違う 2 つの parameter を、排他な `when` の下で置ける。`when` は上に書いた parameter だけを参照できる（`==`、`!=`、`in`、`not in`、比較、`and`、`or`、`not`）。条件が偽の parameter は suggest せず、override も出さない
- 種類は `categorical`、`float`、`int`（`log`、`step` を付けられる）
- study は最初の `run` の snapshot と定義（TOML の中身）に縛られる。コードか TOML を変えたら、名前を `-v2` にした新しい study にする。違う snapshot や定義で `run` すると拒む
- ただし `parallel`、`max_trials`、`prio`、`ram`、`gpu_mem` は束縛に入らない。どれも run_id にも結果にも効かない運用の値なので、TOML で変えて `run` し直せば新しい値で続く（すでにキューにある run の予約は元のまま）
- M14 の study は段 0・1・2 × {小・トレースなし、小・トレースあり、大・トレースあり} の 9 本で、すべて BP で探索する。大のトレースなしは探索せず、大・トレースありで見つかった値で回す（設計 5.4）。`parallel` と `max_trials` は `fleet bench` の前の仮の値

## 目的・報告・打ち切り

- run は `evals.jsonl` の val の行と `result.json` に `objective`（$J$、`exp/video/measure/objective.py`）を書く。調整役はそれを seed で平均するだけ
- 評価は走っている run なら NAS の `live/<run_id>/evals.jsonl` から、終わった run なら `runs/<batch>/<run_id>/evals.jsonl` から読む。途中の報告は trial の全 seed が評価した時刻だけで、seed の平均を `step = round(t)`（仮想時間の秒）で報告する
- 発散した seed は発散した時刻（`result.json` の `t_end`）から先を $J_s = \\log 10$ として数え、その値を tell する。値が有限でないときも同じに扱う。run が failed・corrupt で終わったとき（例外）、誰かが止めたとき、compose できなかったときだけ FAIL にする。FAIL にした trial の残りの run は止める
- 打ち切りは ASHA（`SuccessiveHalvingPruner(min_resource=300, reduction_factor=3)`、段は 300・900 秒）。打ち切った trial は全 seed の claim に `.stop`（`now`、理由 `pruned`）を置き、キューに残る run は `done/` に `stopped` で移す。worker はこれを `stopped`（理由 `pruned`）で `done/` に書く。ほかの走っている trial と共有している run（同じパラメタの trial）は止めない
- 調整役が長く止まっていて、起き直したときに全 seed が終わっていた trial は、途中の値を報告するだけで打ち切らず、最後の値を tell する
- sampler は `TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=16)`。seed は `sampler_seed`（既定 0）に、起動したときの trial の数を足したもの。起き直すたびに同じ乱数の列を引かないようにするため
- trial は suggest した override を user attribute `overrides` に、run_id を `runs` に残す。調整役が止まって起き直すと、RUNNING の trial の報告をそこから続ける。`overrides` はあって `runs` が無い trial は compose からやり直す。`overrides` も無い trial（suggest の前に止まった）は FAIL にする

## 目的を posthoc から取る study（M17）

`[study]` に `objective_from = "posthoc"` を書くと、trial の値を run の評価ではなく、run の後に入れる `task=posthoc` の job から取る。run は目的を書かない（M17 のエンコーダは、デコーダと線形の動力学を後から学んで初めて測れるため）。

```toml
[study]
objective_from = "posthoc"
[study.posthoc]                     # どれも任意
res = "4090"                        # posthoc の job の資源の種類（既定は study の res）
gpu_mem = 12.0                      # 既定は study の gpu_mem
ram = 16.0                          # 既定は study の ram
ckpts = "last"                      # 最後のチェックポイントだけ（"all" は全部）
```

- trial の全 seed の run が completed か diverged になってから、completed の run ごとに posthoc の job を入れる。batch は `<batch>-post`、`after` は親の run_id、`parent` は NAS の `runs/<batch>/<run_id>` の絶対パス、`decode` は `{"ckpts": ...}`。posthoc の run_id は trial の user attribute `post` に残る
- 値は posthoc の `result.json` の `final.decode.val.objective` の seed の平均。diverged の seed は $\\log 10$。posthoc の job が failed などで終わったとき、または目的が無いときは trial を FAIL にする
- 途中の報告をしないので、ASHA は働かない
- `confirm` と `bracket` は、確認の各 run の後に posthoc の job（`ckpts = "all"`、batch `<batch>-post`、資源の種類は確認と同じ）も入れ、run の直後の行に出す

## フリートの側の誤り

trial ではなくフリートの側の誤りでは、trial を FAIL にしない。trial はそのまま待ち、次の step で残りをやり直す。

- compose のプロセス自体が落ちたとき（`RuntimeError`。venv が壊れている、など）と、`OSError`（venv の python が無い、など）: その step を打ち切り、新しい trial も ask しない。直るまで毎 step やり直すので、study の予算は減らない。FAIL にするのは、設定が compose できないとき（`ValueError`）だけ
- NAS の一時的な誤り（`OSError`）: その step を飛ばす。投入の途中で落ちて NAS に無いままの run は、2 回続けて無いのを見たら入れ直す（worker の requeue の rename の途中を見ても、2 つ目のキューファイルを作らないため）
- 同じ parameter の trial が、止めている途中の run（`.stop` の付いた claim）に当たったとき: その run が止まり終わるまで待ち、終わったら新しく入れ直す。止めている run を引き継いで、評価されないまま PRUNED や FAIL になるのを避ける

## テスト

```bash
uv run --project tools/sweep pytest tools/sweep/tests       # optuna の in-memory storage と偽の NAS で
uv run --project tools/sweep pyright -p tools/sweep
```

`just test-sweep` は 1 行目と同じ。`just run` には入っていないので、`exp/lib/fleet` を変えたときも回す。

`tests/test_studies.py` は `exp/video/studies/*.toml` のすべての trial（enqueue と無作為の 6 つ）を本物の `exp.video.compose` で compose する。

## 初版で入れていないもの

`confirm` の監査の run（設計 4.8。`design-decisions.md` で外した）、`fleet up` による調整役の起こし直し（コンテナを作り直した後は `RUNNING` のある study を `just sweep run` で起こす）、optuna-dashboard。
