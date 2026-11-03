# フリート（`exp/lib/fleet/`）

4 台のホスト（std-00〜03）に多数の run を並べて回す仕組み。Mac は snapshot を配って job を投げ、あとで結果を読むだけでよい。job は NAS 上のキューに置かれ、各ホストの worker がそこから取っていく。Mac がスリープしていても run は止まらない。設計は `memory/agents/orchestrator/design-final.md` の 4 節（初版で省いたものは `design-decisions.md`）。

標準ライブラリだけで書いてあり、torch・hydra・`exp.lib` の他のモジュールを import しない。Mac の素の Python 3.14 でも動く。

## 形

```text
Mac   just fleet push / up / doctor / pull / x / logs   ssh でホストへ（ControlMaster つき）
      just fleet <その他>                                std-00 のコンテナへ転送して実行
NAS   /nas/geson/mujou-exp/                             全ホストのコンテナに同じパスでマウント
        queue/<prio>-<ns>-<run_id>.json   claimed/<host>/   done/<run_id>.json
        holds/<batch>   failures/<batch>/<sig>/   hosts/<host>.json   hosts/<host>.cordon
        live/<run_id>/{status.json,evals.jsonl}   runs/<batch>/<run_id>/
ホスト /data/geson/mujou-exp/.fleet/（コンテナ mujou-exp-dev-1 の中で同じパス）
        git/{repo,mujou}.git   code/<snap12>/（読み取り専用）  code/latest
        venv/<envkey>/   runs/<batch>/<run_id>/（実行中の run）  state/   worker.{log,pid,lock}
```

- snapshot: `push` は作業ツリーをそのまま（未追跡のファイルも、mujou の変更も）一時 index で commit にし、HEAD・ブランチ・本物の index は動かさない。ツリーがきれいなら HEAD そのもの。各ホストへ git push し、`code/<snap12>/` に展開して読み取り専用にする
- envkey: `uv.lock`・`pyproject.toml`・mujou の tree・python の版の hash。同じ envkey の snapshot は venv を共有し、mujou が変われば新しい venv ができる。走っている run の下でコードも mujou も変わらない
- run の状態はファイルから決まる。`queue/` にあれば queued、`claimed/<host>/` にあれば running（そのホストの心拍が 10 分より古ければ `lost?`）、どちらにも無ければ `done/<run_id>.json` の `status` が終わり方。キューのファイルか claim があれば、前の attempt が残した `done/` の記録よりそちらを優先する。`failed` や `stopped` で終わった run を投げ直すと、古い記録は消える
- job の claim は `os.rename` で行う。SMB の rename はサーバーの側で排他なので、2 台が同じ job を取ることはない（WP0 で確認）

## 典型の 1 周

```bash
just fleet doctor                     # 4 台の ssh・コンテナ・GPU・NAS・時計・道具・コアを確かめる
just fleet push                       # snapshot を作って 4 台に配る（最後の行が snap）
just fleet up                         # 止まっている worker を起動する（コードを替えたら --restart）
just fleet submit --batch m14-s1-v1 +experiment=m14/s1 --sweep optim.lr=0.01,0.03 --seeds 0,1,2
just fleet status                     # 状態ごとの数、ホストごとの枠、走っている run の進み
just fleet pull --batch m14-s1-v1     # 完了した run を NAS から ./runs/ へ（ckpt を除く）
just fleet table --batch m14-s1-v1 --rows config.optim.lr --cols config.learner.kind --value roll10/cv10
```

`submit` は head（std-00）の `code/<snap>` の中で、その snapshot の venv の python が `exp.video.compose` を呼んで設定を解決する。設定の誤りはキューに入る前に落ちる。同じ (設定, snapshot, 資源の種類, seed) は同じ `run_id` になる。それがキューにあるか、実行中か、completed・diverged で終わっていれば `duplicate` と出して飛ばす。failed などで終わったものは投げ直せる。

## コマンド

| コマンド | どこで | 内容 |
| -- | -- | -- |
| `doctor` | Mac | 各ホストの ssh、コンテナ（running と `unless-stopped`）、user、GPU の並び、NAS への書き込みと rename、root、時計のずれ、道具、物理コアの数 |
| `push` | Mac | snapshot → 4 台へ git push → 展開 → venv（無ければ `uv sync --frozen --no-editable`）→ 検証 → `code/latest` |
| `up [--restart]` | Mac | 各ホストで `up-local`。worker が動いていれば何もしない。`--restart` は止めてから `code/latest` で起動し直す |
| `submit --batch B [overrides...]` | head | `--sweep k=a,b`（直積）、`--seeds 0,1,2`、`--res cpu\|any_gpu\|4090\|4060`、`--ram G`、`--gpu-mem G`、`--prio P`（小さいほど先）、`--after RUN_ID`、`--host H`、`--avoid-host H`、`--tag T`、`--snap S`、`--dry` |
| `status [--batch B] [--json]` | head | 数、ホストの心拍と枠、hold、走っている run の t/of・fps・ETA・最新の roll10/cv10 |
| `ls [--batch B] [--state S]` | head | run の一覧 |
| `logs RUN [--err] [--tail N]` | Mac | 走っていればそのホストのローカルから、終わっていれば NAS から |
| `stop RUN...\|--batch B\|--trial S/N [--now] [--reason R]` | head | 走っている run に停止を要求する。キューにある job は `stopped` で `done/` へ |
| `cancel ...` | head | `stop --now --reason cancel` と同じ |
| `retry RUN [--fresh] [--ram G] [--gpu-mem G]` | head | `done/` の run を attempt を 1 つ進めて投げ直す。`--fresh` は再開せず最初から |
| `requeue --host H` | head | 戻らないホストの claim をキューへ戻す（最初から）。そのホストが後で戻っても二重に走らない（fencing） |
| `hold B` / `release B` | head | batch の保留と解除。`release` は失敗の印も消す |
| `cordon H [--res R]` / `uncordon H [--res R]` | head | ホスト（またはその資源の種類）に新しい run を置かない |
| `pull [--batch B] [--ckpt] [--live RUN] [--dest runs]` | Mac | NAS の `runs/` を tar で持ってくる。`--live` は走っている run をホストから直接 |
| `table --batch B --rows ... --cols ... --value K` | head | NAS の run の表。引数は `python -m exp.lib.runs table` にそのまま渡す（WP6） |
| `x <host\|all> -- cmd ...` | Mac | コンテナの中で、`code/latest` を cwd、その venv を PATH の先頭にして実行する |

`--local` を付けると、その場で `fleet.toml` のパスに対して実行する（コンテナの中とテストで使う）。`worker`・`up-local`・`provision`・`locate` はホストの側の内部コマンド。

## 資源と枠

`fleet.toml` の `[slots]`。1 台で `cpu` 個の物理コアを run に与え、低い番号の残りは worker などに残す（既定は 16 のうち 14）。CPU の run も GPU の run も 1 本が CPU を 1 単位とる（`smt=false` なら物理コア 1 つ。論理 CPU の i と i+16 の両方に `taskset` で縛る）。GPU の run はさらに GPU を 1 枚とる。`any_gpu` は空きメモリの多い方。GPU ごとに本数（`jobs`）とメモリ（`mem_gb`）、ホストごとに RAM（`ram_gb`）の予算を超えない。

run の環境: `OMP_NUM_THREADS=1` など、`CUDA_VISIBLE_DEVICES=<その GPU>`（CPU の run は空）、`PYTHONPYCACHEPREFIX=/tmp/pycache`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`FLEET_HOST`。`PYTHON_GIL` は消す。`job.json` の `config.device` は GPU の run で `cuda`、CPU の run で `cpu` に書き換える。

## run の終わり方（`done/<run_id>.json` の `status`）

| 起きたこと | status | worker の扱い |
| -- | -- | -- |
| exit 0、`result.json` と `manifest.json` あり | `completed` | NAS へ写して manifest で検証（3 回失敗で `corrupt`）、ローカルの `resume/` を消す |
| exit 3（発散）、同上 | `diverged` | 同上。再試行しない |
| exit 0 / 3 なのに result か manifest が無い | `failed`（contract） | 黙った成功を許さない |
| exit 4（環境の誤り） | （キューへ戻る） | そのホストのその資源の種類を cordon し、attempt を進めてキューへ |
| exit 75（停止の要求に従った）、または要求の後に殺した | `stopped`（理由） | `drain`・`move` ならキューへ戻す。`stall` は `failed` |
| 停止の要求なしの SIGKILL | （1 回だけキューへ） | OOM の見込み。RAM を 1.5 倍にして `retry.oom` 回まで。尽きたら `failed`（oom） |
| exit 1 など | `failed` | traceback の最後の行で印を作る。同じ印が batch で 3 回なら `holds/<batch>` |
| `status.json` の `frame` が 20 分進まない | `failed`（stall） | 停止を要求し、猶予の後に殺す。自動の再試行はしない |
| 引き取った run が終わった | 上の表のとおり | worker の子プロセスでないので exit code が分からない。`result.json` と `manifest.json` があれば `completed` / `diverged`。無ければ `status.json` の最後の `state` で決める: `failed` は `failed`、`env` は exit 4 と同じ扱い、`stopped` は `stopped` |
| worker が戻ったら run の pid が死んでいた（`state` も手がかりにならない） | （同じホストで再開） | `lost`。`resume/latest` から attempt を進めて起動し直す（`retry.lost` 回まで） |
| 終わりを片づける途中で worker が例外を出した（壊れた manifest など） | `failed`（`worker: <例外>`） | run を `failed` で閉じ、worker は動き続ける |

停止の要求は `claimed/<host>/<run_id>.stop` に置かれ、worker が run の `stop.request` に写す。120 秒の猶予（`--now` なら即）の後、process group に SIGTERM、10 秒後に SIGKILL を送る。孫のプロセスも死ぬ。終わった run の group に残ったプロセスも SIGKILL で掃除する。

NAS が一時的に読めないとき（EIO など）、worker はそれを claim の消失と取り違えない。run は走り続ける。run の終わりを NAS へ書けなければ、次の tick で書き直す。

NAS の `runs/<batch>/<run_id>/` へ写すときは、前の attempt のファイルを先に消す。`logs/` と `exit/` はファイル名が attempt ごとに違うので、消さずに残す。

各 attempt の終わりは run の `exit/attempt-<n>.json`（`returncode`、`signal`、`started`、`ended`、`stop_reason`、`adopted`）に残る。標準出力と標準エラーは `logs/attempt-<n>.{out,err}`。

## worker の運用

- 1 ホスト 1 本。`.fleet/worker.lock` を flock で持つので、2 本目は起動してすぐ終わる。ログは `.fleet/worker.log`
- 起動: `just fleet up`。worker は `code/latest`（起動した時点の snapshot）から動く。`push` で新しい snapshot を配っても、走っている worker は替わらない。worker のコードを替えるときは `just fleet up --restart`
- 再起動しても run は止まらない（run は別の session で、pid を `.fleet/state/<run_id>.json` に残す）。新しい worker は生きている run を引き取る（`adopted`）。死んでいた run は `resume/latest` から再開する
- コンテナやホストが再起動したとき: コンテナは `restart: unless-stopped` で戻る。その後 Mac から `just fleet up` を叩くと worker が戻り、上の手順で run も戻る
- ホストが戻らないとき: `just fleet requeue --host std-0X`。job は最初からやり直しになる。再開用の checkpoint はそのホストのローカルにしかないため
- worker を止めるだけなら `just fleet x std-0X -- bash -c 'kill $(cat /data/geson/mujou-exp/.fleet/worker.pid)'`（SIGTERM。run はそのまま走り続け、次の `up` で引き取られる）

## 偽の program（`fake.py`）

`--raw --entry exp.lib.fleet.fake` で投げると、run の契約を守る偽の program が走る（設定の compose は飛ばす）。振る舞いは `--set fake.<key>=<json>` か override の `fake.<key>=<json>` で決める（`--sweep fake.exit=0,3` も使える）: `seconds`、`frames`、`exit`（0・3・4・1、または attempt ごとのリスト `[4,0]`）、`kill`、`no_manifest`、`hang_at`、`ignore_stop`、`child`、`resume_every`。

```bash
just fleet submit --batch smoke --raw --entry exp.lib.fleet.fake --name smoke --res 4090 \
  --set fake.seconds=20 --sweep fake.exit=0,3
```

## 他の包みとの約束

- program の起動: `cwd=<root>/code/<snap12>` で `<root>/venv/<envkey>/bin/python -m <entry> --job <run_dir>/job.json --attempt <n>`。`entry` は `job.json` の `entry`（既定は `exp.video.main`）。`resume/latest` があれば program が自分で再開する（worker は `--resume` を渡さない）。新しく始めるときは、worker が `logs/` と `exit/` 以外を消してから起動する
- manifest: `{"schema": 1, "files": {相対パス: {"sha256", "size"}}}`。`{"path", "sha256", "size"}` のリストでも読める。最上位の `manifest.json`・`resume/`・`logs/`・`exit/`・`status.json`・`timing.jsonl`・`stop.request` は含めない
- 心拍: worker は `status.json` の `frame` で止まりを見て、`status.json` と `evals.jsonl` を 60 秒ごとに `live/` へ写す
- 終わりの `state`: program は終わる直前に `status.json` の `state` を書く（例外なら `failed`、環境の誤りなら `env`、停止の要求に従ったなら `stopped`）。worker を再起動すると、引き取った run の exit code は分からなくなる。そのときはこの `state` で終わり方を決める。書かないと `lost` とみなされ、同じホストで起動し直される
- job の形: `jobs.make_job`。`run_id`・`group` の作り方は `jobs.group_of`・`jobs.run_id_of`。投入は `queue.Nas.submit`、停止の要求は `queue.Nas.request_stop`（optuna の調整役が使う）

## 初版で入れていないもの

監査の run（`fleet reproduce`）、canary の自動投入、`fleet gc`、`fleet bench`、止まりの検出での SIGUSR1、optuna の調整役の起動（`up` は worker だけを起こす）。古い `code/` と `venv/` は手で消す（読み取り専用なので `chmod -R u+w` してから）。
