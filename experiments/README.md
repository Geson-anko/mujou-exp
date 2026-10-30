# experiments/

One directory per experiment, a Python package: `experiments/<name>/` with an `__init__.py`, the code, and a `README.md` that states the hypothesis, how to run it and the result. Run it as `uv run python -m experiments.<name>.<script>`.

- Library code that several experiments need and that is not specific to one goes into mujou (see the skill `mujou-upstream`), not here.
- Checkpoints, logs and plots are not committed (`runs/`, `outputs/`, `logs/` are ignored); write the numbers that matter into the experiment's `README.md`.
