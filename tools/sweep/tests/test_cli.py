"""``sweep <cmd>``: on the Mac every command is forwarded to the head's
container; there (``--local``) it runs against the paths of ``fleet.toml``.

The local commands run here on a host root in a temporary directory, whose
snapshot is this checkout and whose venv python is this interpreter (it has
hydra), so ``exp.video.compose`` composes for real, as in ``fleet submit``.
"""

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import override

import optuna
import pytest
from conftest import SNAP_A, code_info
from sweep.cli import main, study_storage
from sweep.study import StudyConf, open_study

from exp.lib.fleet.contract import read_json
from exp.lib.fleet.queue import Nas
from exp.lib.fleet.transport import Transport

REPO = Path(__file__).resolve().parents[3]
TOML = """\
nas = "{nas}"
root = "{root}"
checkout = "{repo}"
container = "c"
user = "u"
head = "h0"
hosts = ["h0", "h1"]

[slots]
cpu = 2
smt = false
ram_gb = 8
gpus = [{{index = 0, model = "4090", mem_gb = 22.0, jobs = 2}}]
"""
STUDY = """\
[study]
name = "{name}"
overrides = ["+experiment={preset}", "learner=bp", "phase=explore"]
seeds = [100, 101]
parallel = 2
max_trials = 2

[space]
"optim.lr" = {{float = [0.001, 0.3], log = true}}
"""


class Host:
    """A host root with the snapshot ``SNAP_A`` (this checkout) ready."""

    def __init__(self, tmp: Path) -> None:
        self.nas = Nas(tmp / "nas")
        self.root = tmp / "root"
        self.toml = tmp / "fleet.toml"
        self.toml.write_text(TOML.format(nas=self.nas.root, root=self.root, repo=REPO))
        self.code = self.root / "code" / SNAP_A[:12]
        (self.code / "exp" / "video").mkdir(parents=True)
        for name in os.listdir(REPO / "exp"):
            if name != "video":
                (self.code / "exp" / name).symlink_to(REPO / "exp" / name)
        for name in os.listdir(REPO / "exp" / "video"):
            if name != "studies":
                (self.code / "exp" / "video" / name).symlink_to(
                    REPO / "exp" / "video" / name
                )
        (self.code / "exp" / "video" / "studies").mkdir()
        (self.code / "CODE.json").write_text(json.dumps(code_info()))
        (self.root / "code" / "latest").symlink_to(self.code.name)
        python = self.root / "venv" / code_info()["envkey"] / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)

    def study(self, stem: str, name: str, preset: str = "m14/s0") -> str:
        rel = f"exp/video/studies/{stem}.toml"
        (self.code / rel).write_text(STUDY.format(name=name, preset=preset))
        return rel

    def main(self, *argv: str) -> int:
        return main(["--local", "--config", str(self.toml), *argv])


@pytest.fixture
def host(tmp_path: Path) -> Host:
    return Host(tmp_path)


class TestRun:
    def test_a_study_runs_to_its_end_and_leaves_a_copy_on_the_nas(
        self, host: Host, capsys
    ):
        # The preset does not exist: both trials fail to compose, and the
        # study ends at once with max_trials told.
        rel = host.study("broken", "broken-v1", preset="m14/nope")
        assert host.main("run", rel, "--foreground") == 0
        storage = study_storage(host.root, "broken-v1")
        study = optuna.load_study(study_name="broken-v1", storage=storage)
        assert [t.state for t in study.trials] == [optuna.trial.TrialState.FAIL] * 2
        assert study.user_attrs["snap"] == SNAP_A
        nas_dir = host.nas.root / "studies" / "broken-v1"
        assert not (nas_dir / "RUNNING").exists()
        copy = optuna.load_study(
            study_name="broken-v1", storage=f"sqlite:///{nas_dir / 'broken-v1.db'}"
        )
        assert len(copy.trials) == 2

        assert host.main("status", "broken-v1") == 0
        out = capsys.readouterr().out
        assert "FAIL 2" in out
        assert SNAP_A[:12] in out

    def test_a_study_bound_to_one_snapshot_refuses_another(self, host: Host):
        rel = host.study("broken", "broken-v1", preset="m14/nope")
        assert host.main("run", rel, "--foreground") == 0
        other = host.root / "code" / ("f" * 12)
        other.mkdir()
        (other / "exp" / "video" / "studies").mkdir(parents=True)
        (other / rel).write_text((host.code / rel).read_text())
        (other / "CODE.json").write_text(json.dumps(code_info("f" * 40)))
        with pytest.raises(SystemExit, match="snapshot"):
            host.main("run", rel, "--snap", "f" * 12, "--foreground")


class TestConfirm:
    def test_confirm_composes_and_queues_the_best_trial(self, host: Host, capsys):
        rel = host.study("s0", "m14-s0-cli-v1")
        conf = StudyConf.load(host.code / rel)
        storage = study_storage(host.root, conf.name)
        study = open_study(conf, storage, code=code_info())
        for lr, value in ((0.01, -0.2), (0.05, -0.6)):
            study.enqueue_trial({"optim.lr": lr})
            t = study.ask()
            conf.space.suggest(t)
            study.tell(t, value)
        argv = ["confirm", conf.name, "--top", "1", "--learners", "bp,pc"]
        argv += ["--seeds", "0,1", "--batch", "m14-s0-confirm-v1", "--res", "4090"]
        assert host.main(*argv) == 0
        out = capsys.readouterr().out
        assert out.count("new") == 4
        jobs = [read_json(p) for p in host.nas.entries()]
        assert len(jobs) == 4
        for j in jobs:
            assert j["config"]["phase"] == "confirm"
            assert j["config"]["optim"]["lr"] == 0.05
            assert j["config"]["ctl"]["eval_test"] is True  # the confirm preset
            assert j["res"]["class"] == "4090"
        assert {j["config"]["learner"]["kind"] for j in jobs} == {"bp", "pc"}

    def test_another_gpu_class_keeps_the_studys_gpu_memory(self, host: Host):
        rel = "exp/video/studies/big.toml"
        text = STUDY.format(name="m14-s0-big-v1", preset="m14/s0")
        text = text.replace(
            "max_trials = 2\n", 'max_trials = 2\nres = "any_gpu"\ngpu_mem = 3.0\n'
        )
        (host.code / rel).write_text(text)
        conf = StudyConf.load(host.code / rel)
        study = open_study(conf, study_storage(host.root, conf.name), code=code_info())
        study.enqueue_trial({"optim.lr": 0.01})
        t = study.ask()
        conf.space.suggest(t)
        study.tell(t, -0.2)
        argv = ["confirm", conf.name, "--top", "1", "--learners", "bp"]
        argv += ["--seeds", "0", "--batch", "b", "--res", "4090"]
        assert host.main(*argv) == 0
        (job,) = (read_json(p) for p in host.nas.entries())
        assert job["res"] == {"class": "4090", "ram_gb": 2.0, "gpu_mem_gb": 3.0}

    def test_a_class_the_fleet_lacks_is_refused(self, host: Host):
        rel = host.study("s0", "m14-s0-cli-v1")
        open_study(
            StudyConf.load(host.code / rel),
            study_storage(host.root, "m14-s0-cli-v1"),
            code=code_info(),
        )
        argv = ["confirm", "m14-s0-cli-v1", "--top", "1", "--learners", "bp"]
        argv += ["--seeds", "0", "--batch", "b", "--res", "3090"]
        with pytest.raises(SystemExit, match="3090"):
            host.main(*argv)


class _Recorder(Transport):
    """Records the commands it is asked to run in a container."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    @override
    def host(self, host, argv, *, input=None, capture=True, timeout=None):
        raise AssertionError("not on the host")

    @override
    def container(
        self,
        host: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        input: bytes | None = None,
        capture: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((host, list(argv)))
        return subprocess.CompletedProcess(list(argv), 0, b"", b"")

    @override
    def container_argv(self, host: str, argv: Sequence[str]) -> list[str]:
        return list(argv)

    @override
    def git_url(self, host: str, path: Path) -> str:
        return str(path)

    @override
    def git_env(self) -> dict[str, str]:
        return {}


class TestForward:
    def test_the_mac_forwards_to_the_heads_container(self, host: Host):
        t = _Recorder()
        argv = ["--config", str(host.toml), "run", "exp/video/studies/x.toml"]
        assert main(argv, transport=t) == 0
        ((where, cmd),) = t.calls
        assert where == "h0"
        assert cmd[:2] == ["bash", "-c"]
        assert cmd[3:] == ["_", str(host.root), "run", "exp/video/studies/x.toml"]

    def test_the_forwarding_shell_runs_the_local_cli(self, host: Host):
        # The shell of the forward, run here: tools/sweep of code/latest under
        # uv with its own venv; `status` of a study that does not exist fails.
        (host.code / "tools").symlink_to(REPO / "tools")
        (host.code / "fleet.toml").symlink_to(host.toml)
        t = _Recorder()
        main(["--config", str(host.toml), "status", "nope-v1"], transport=t)
        ((_, cmd),) = t.calls
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert proc.returncode != 0
        assert "no study nope-v1" in proc.stderr
