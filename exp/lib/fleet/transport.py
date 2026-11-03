"""How the Mac reaches a host and its container.

``SshTransport`` runs ``ssh <host> <cmd>`` (and ``docker exec`` in the
container) over one master connection per host (``ControlMaster``, kept 10
minutes), with the options on the command line: the user's ``~/.ssh/config``
stays as it is. ``LocalTransport`` runs the same commands here, for the tests.
"""

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, override

from exp.lib.fleet.config import FleetConf

__all__ = ["LocalTransport", "SshTransport", "Transport"]

type Result = subprocess.CompletedProcess[bytes]


class Transport(Protocol):
    def host(
        self,
        host: str,
        argv: Sequence[str],
        *,
        input: bytes | None = None,
        capture: bool = True,
        timeout: float | None = None,
    ) -> Result:
        """Run ``argv`` on ``host`` (outside the container)."""
        ...

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
    ) -> Result:
        """Run ``argv`` in the container of ``host`` as the fleet's user."""
        ...

    def container_argv(self, host: str, argv: Sequence[str]) -> list[str]:
        """The command line that runs ``argv`` in the container of ``host``, for
        a caller that streams its output (``subprocess.Popen``)."""
        ...

    def git_url(self, host: str, path: Path) -> str:
        """The URL to ``git push`` to the repository ``path`` of ``host``."""
        ...

    def git_env(self) -> dict[str, str]:
        """The environment ``git push`` needs to reach the hosts."""
        ...


def _run(
    argv: Sequence[str],
    *,
    input: bytes | None,
    capture: bool,
    timeout: float | None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> Result:
    return subprocess.run(
        list(argv),
        # Not the Mac's terminal: a command reads stdin only when given input.
        stdin=subprocess.DEVNULL if input is None else None,
        input=input,
        capture_output=capture,
        timeout=timeout,
        cwd=cwd,
        env=None if env is None else {**os.environ, **env},
        check=False,
    )


class SshTransport(Transport):
    """``ssh`` to the hosts of ``conf``; control sockets under ``control_dir``
    (short: a socket path has at most ~100 bytes)."""

    def __init__(self, conf: FleetConf, control_dir: Path | None = None) -> None:
        self._conf = conf
        self._dir = control_dir or Path(f"/tmp/fleet-ssh-{os.getuid()}")
        self._dir.mkdir(mode=0o700, exist_ok=True)

    def ssh_argv(self) -> list[str]:
        """``ssh`` with the fleet's options."""
        return [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self._dir}/%C",
            "-o", "ControlPersist=600",
        ]  # fmt: skip

    def _host_argv(self, host: str, argv: Sequence[str]) -> list[str]:
        return [*self.ssh_argv(), host, "--", shlex.join(argv)]

    def _docker_argv(
        self,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        interactive: bool = False,
    ) -> list[str]:
        docker = ["docker", "exec", "-u", self._conf.user]
        if interactive:
            docker.append("-i")
        if cwd is not None:
            docker += ["-w", cwd]
        for k, v in (env or {}).items():
            docker += ["-e", f"{k}={v}"]
        return [*docker, self._conf.container, *argv]

    @override
    def host(self, host, argv, *, input=None, capture=True, timeout=None) -> Result:
        return _run(
            self._host_argv(host, argv), input=input, capture=capture, timeout=timeout
        )

    @override
    def container(
        self, host, argv, *, cwd=None, env=None, input=None, capture=True, timeout=None
    ) -> Result:
        docker = self._docker_argv(
            argv, cwd=cwd, env=env, interactive=input is not None
        )
        return self.host(host, docker, input=input, capture=capture, timeout=timeout)

    @override
    def container_argv(self, host: str, argv: Sequence[str]) -> list[str]:
        return self._host_argv(host, self._docker_argv(argv))

    @override
    def git_url(self, host: str, path: Path) -> str:
        return f"ssh://{host}{path}"

    @override
    def git_env(self) -> dict[str, str]:
        return {"GIT_SSH_COMMAND": shlex.join(self.ssh_argv())}


class LocalTransport(Transport):
    """Every host is this machine and has no container: the tests' transport."""

    @override
    def host(self, host, argv, *, input=None, capture=True, timeout=None) -> Result:
        return _run(argv, input=input, capture=capture, timeout=timeout)

    @override
    def container(
        self, host, argv, *, cwd=None, env=None, input=None, capture=True, timeout=None
    ) -> Result:
        return _run(
            argv, input=input, capture=capture, timeout=timeout, cwd=cwd, env=env
        )

    @override
    def container_argv(self, host: str, argv: Sequence[str]) -> list[str]:
        return list(argv)

    @override
    def git_url(self, host: str, path: Path) -> str:
        return str(path)

    @override
    def git_env(self) -> dict[str, str]:
        return {}
