---
name: project-mujou-submodule
description: mujou は git submodule（mujou/）で editable install。未 push のコミットを指す submodule ポインタをコミットしない
metadata:
  type: project
---

`mujou/` は https://github.com/Geson-anko/mujou の submodule で、`pyproject.toml` の `[tool.uv.sources] mujou = { path = "mujou", editable = true }` で入れる（PyPI 版は使わない）。

- `uv sync` は Rust 拡張を maturin でビルドする。ホストに Rust が無いときは dev container（mujou 本体のものを下敷きにした `.devcontainer/`）で作業する
- mujou の hook・skill・settings はこのリポジトリのルートでは自動では効かない。変更時は `mujou/CLAUDE.md` に従い、局所性は `mujou/.claude/skills/locality-and-async/SKILL.md` と手で突き合わせる

**Why:** 未 push のコミットを指すポインタをコミットすると、他の clone や container で submodule を取得できずリポジトリが壊れる。

**How to apply:** ポインタのコミット前に `just mujou-status`。手順は skill `mujou-upstream`。関連: [[project-purpose]]
