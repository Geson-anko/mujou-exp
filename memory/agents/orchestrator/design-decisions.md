# design-final.md に対する orchestrator の判断（2026-11-03）

design-final.md を全文読んだうえで、土台として採る。ただし初版では次を入れない（要るとわかった時点で足す）。実装の包みはこれに従う。

- 監査の run（4.8）と `fleet reproduce`、`fleet table` の nondeterministic の脚注
- canary の自動投入（4.7）。新しい snapshot では orchestrator が手で短い run を先に流す
- `fleet gc`、`fleet bench` のコマンド。枠の較正は bench の preset を `fleet submit --sweep` で投げて表で見る
- 止まりの検出での SIGUSR1。止まった run は停止して failed（`stall`）にし、自動の再試行はしない
- コンテナを作り直す前の確認は、走っている run が無いことだけを見る

残すもの: NAS のキューと rename の claim、worker（回収・停止・live・心拍・holds・再起動での引き取りと lost の再開）、git の snapshot と envkey の venv、run の契約（終了コード、manifest、再開）、`runs.py` の表（混在の拒否を含む）、optuna の調整役（ASHA、seed の平均、発散は log 10 で tell）、science 案の科学の要（0.3 節）すべて。

ssh の ControlMaster は利用者の `~/.ssh/config` を変えず、fleet のコードが `-o ControlMaster=auto -o ControlPath=... -o ControlPersist=600` を付ける。Mac に Homebrew の rsync は入れない（ssh + tar で写す）。
