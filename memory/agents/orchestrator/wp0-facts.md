# WP0: 実機で確かめた事実（2026-11-03）

- SMB の rename は排他的: 2 台（std-00・std-01）が同じ 200 件を同時に `os.rename` で取り合い、3 回とも重複 0、取り残し 0（94/106、107/93、83/117）
- コンテナの 3.14t で、`import hydra, omegaconf, yaml` の後も、`torch._C._set_deterministic_algorithms(True)`・TF32 の無効化・CUDA の matmul の後も `sys._is_gil_enabled()` は False、`torch._dynamo` は読まれない
- `UV_PROJECT_ENVIRONMENT=/tmp/venv-x uv sync --frozen --no-editable` は 6 秒（ビルドのキャッシュあり）。mujou は wheel として site-packages に入り、cwd が `/tmp` でも import でき、GIL は無効
- CPU は AMD Ryzen 9 7950X。論理 CPU i と i+16 が同じ物理コア（`thread_siblings_list` が `0,16`、`1,17`）
- 4 台のコンテナを `devcontainer up --remove-existing-container` で作り直した: `RestartPolicy=unless-stopped`、vscode が `/nas/geson/mujou-exp` に書ける（compose の `FLEET_NAS`、各ホストの `.env` に `FLEET_NAS=/nas/geson/mujou-exp`）、GPU 2 枚
- `devcontainer up` は作り直しのときも `docker events` で止まる（1 回目は timeout、止めて 2 回目で success）
- hydra-core 1.3.8 を依存に足した（`uv add`）
