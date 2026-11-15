"""A study's definition (its TOML, design 5.4) and its optuna study.

::

    [study]
    name = "m14-s1-big-trace-bp-v1"   # also the batch of its runs
    overrides = ["+experiment=m14/s1", "learner=bp", "dynamics=big", ...]
    seeds = [100, 101]                # every trial runs all of them
    parallel = 12                     # trials in flight (x seeds runs)
    max_trials = 120
    res = "any_gpu"                   # optional: ram, gpu_mem, prio, batch,
                                      # min_resource, reduction_factor,
                                      # n_startup_trials, sampler_seed,
                                      # objective_from ("evals" | "posthoc")
    [study.posthoc]                   # with objective_from = "posthoc" (optional)
    res = "4090"                      # res, gpu_mem, ram, ckpts ("last" | "all")
    [space]
    ...                               # sweep.space
    [[enqueue]]
    "optim" = "sgd"                   # trials to ask first (M13's best)

A study is bound to one code snapshot and one definition: the first
``open_study`` records both in the study's user attributes, and a later one
with another snapshot or another definition raises ``StudyError``. Changed
code or a changed TOML is a new study (``-v2``), except for the keys of how it
is run (``parallel``, ``max_trials``, ``prio``, ``ram``, ``gpu_mem``), which
change neither the runs' run_ids nor their results: a later ``open_study``
takes their new values.
"""

import json
import os
import re
import shutil
import sqlite3
import tomllib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

from exp.lib.fleet.jobs import make_job
from exp.lib.fleet.queue import nas_retry
from sweep.space import Space

__all__ = [
    "OBJECTIVE_FROM",
    "StudyConf",
    "StudyError",
    "backup",
    "check_bound",
    "open_study",
]

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
# Where a trial's value comes from: the runs' own val evaluations, or the
# result of a task=posthoc job after each run (its final.decode.val.objective).
OBJECTIVE_FROM = ("evals", "posthoc")
_POSTHOC_KEYS = frozenset({"res", "gpu_mem", "ram", "ckpts"})
# The keys of [study] that do not bind it: how many trials, how soon, and
# what a run reserves (not in its run_id).
_OPERATIONAL = frozenset({"parallel", "max_trials", "prio", "ram", "gpu_mem"})
_STUDY_KEYS = frozenset(
    {
        "name",
        "overrides",
        "seeds",
        "parallel",
        "max_trials",
        "res",
        "ram",
        "gpu_mem",
        "prio",
        "batch",
        "min_resource",
        "reduction_factor",
        "n_startup_trials",
        "sampler_seed",
        "objective_from",
        "posthoc",
    }
)


class StudyError(Exception):
    """The study on the storage is not the one of this definition and snapshot."""


@dataclass(frozen=True, kw_only=True)
class StudyConf:
    """The definition of a study: its runs are ``overrides`` + a trial's
    parameters + ``seed=s`` for each of ``seeds``, queued as the batch
    ``batch`` on slots of ``res``; ``parallel`` trials at a time, ``max_trials``
    in all. ASHA (``min_resource`` seconds of virtual time, then times
    ``reduction_factor``) and TPE (``n_startup_trials`` random ones first).

    With ``objective_from`` ``"posthoc"`` a trial's value is that of the
    ``task=posthoc`` job queued after each of its completed runs (batch
    ``<batch>-post``, on slots of ``post_res``, its ``ckpts`` checkpoints;
    ``sweep.coordinator``), and ASHA never prunes (there is no value before
    the end)."""

    name: str
    overrides: tuple[str, ...]
    seeds: tuple[int, ...]
    parallel: int
    max_trials: int
    space: Space
    enqueue: tuple[dict[str, Any], ...] = ()
    res: str = "cpu"
    ram_gb: float = 2.0
    gpu_mem_gb: float | None = None  # None: 0 for cpu, 2 for a GPU class
    prio: int = 60  # after the confirmations (20): smaller runs first
    batch: str = field(default="")  # "": the name
    min_resource: int = 300
    reduction_factor: int = 3
    n_startup_trials: int = 16
    sampler_seed: int = 0
    objective_from: str = "evals"
    post_res: str = ""  # "": res
    post_gpu_mem_gb: float | None = None  # None: gpu_mem
    post_ram_gb: float | None = None  # None: ram
    post_ckpts: str = "last"

    @classmethod
    def load(cls, path: Path) -> StudyConf:
        """Read the TOML ``path``."""
        with path.open("rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StudyConf:
        """The definition of a parsed TOML; ``ValueError`` names what is wrong."""
        s = dict(d["study"])
        unknown = set(s) - _STUDY_KEYS
        if unknown:
            raise ValueError(f"[study]: unknown keys {sorted(unknown)}")
        if not _NAME.fullmatch(str(s["name"])):
            raise ValueError(f"[study] name {s['name']!r}: letters, digits, _ -")
        post = dict(s.get("posthoc") or {})
        if not set(post) <= _POSTHOC_KEYS:
            raise ValueError(
                f"[study.posthoc]: unknown keys {sorted(set(post) - _POSTHOC_KEYS)}"
            )
        space = Space.from_dict(d.get("space", {}))
        enqueue = tuple(dict(e) for e in d.get("enqueue", []))
        for e in enqueue:
            space.check(e)
        conf = cls(
            name=str(s["name"]),
            overrides=tuple(str(o) for o in s["overrides"]),
            seeds=tuple(int(x) for x in s["seeds"]),
            parallel=int(s["parallel"]),
            max_trials=int(s["max_trials"]),
            space=space,
            enqueue=enqueue,
            res=str(s.get("res", "cpu")),
            ram_gb=float(s.get("ram", 2.0)),
            gpu_mem_gb=None if s.get("gpu_mem") is None else float(s["gpu_mem"]),
            prio=int(s.get("prio", 60)),
            batch=str(s.get("batch") or s["name"]),
            min_resource=int(s.get("min_resource", 300)),
            reduction_factor=int(s.get("reduction_factor", 3)),
            n_startup_trials=int(s.get("n_startup_trials", 16)),
            sampler_seed=int(s.get("sampler_seed", 0)),
            objective_from=str(s.get("objective_from", "evals")),
            post_res=str(post.get("res", "")),
            post_gpu_mem_gb=None
            if post.get("gpu_mem") is None
            else float(post["gpu_mem"]),
            post_ram_gb=None if post.get("ram") is None else float(post["ram"]),
            post_ckpts=str(post.get("ckpts", "last")),
        )
        if conf.objective_from not in OBJECTIVE_FROM:
            raise ValueError(
                f"[study] objective_from must be one of {OBJECTIVE_FROM}, not "
                f"{conf.objective_from!r}"
            )
        if conf.post_ckpts not in ("last", "all"):
            raise ValueError(
                f"[study.posthoc] ckpts: 'last' or 'all', not {conf.post_ckpts!r}"
            )
        if not conf.seeds or conf.parallel < 1 or conf.max_trials < 1:
            raise ValueError("[study]: seeds, parallel and max_trials must be positive")
        if any(o.startswith("seed=") for o in conf.overrides):
            raise ValueError("[study]: the seeds go in seeds, not in overrides")
        return conf

    def to_dict(self) -> dict[str, Any]:
        """The TOML this definition reads from (normalized): what binds the study."""
        study: dict[str, Any] = {
            "name": self.name,
            "overrides": list(self.overrides),
            "seeds": list(self.seeds),
            "parallel": self.parallel,
            "max_trials": self.max_trials,
            "res": self.res,
            "ram": self.ram_gb,
            "gpu_mem": self.gpu_mem_gb,
            "prio": self.prio,
            "batch": self.batch,
            "min_resource": self.min_resource,
            "reduction_factor": self.reduction_factor,
            "n_startup_trials": self.n_startup_trials,
            "sampler_seed": self.sampler_seed,
        }
        if self.objective_from != "evals":
            study["objective_from"] = self.objective_from
            study["posthoc"] = {
                "res": self.post_res,
                "gpu_mem": self.post_gpu_mem_gb,
                "ram": self.post_ram_gb,
                "ckpts": self.post_ckpts,
            }
        return {
            "study": study,
            "space": self.space.to_dict(),
            "enqueue": [dict(e) for e in self.enqueue],
        }

    @classmethod
    def of_study(cls, study: optuna.Study) -> StudyConf:
        """The definition ``open_study`` recorded in ``study``."""
        return cls.from_dict(study.user_attrs["conf"])

    def res_dict(self) -> dict[str, Any]:
        """``res`` of the jobs: ``{"class", "ram_gb", "gpu_mem_gb"}``."""
        gpu_mem = self.gpu_mem_gb
        if gpu_mem is None:
            gpu_mem = 0.0 if self.res == "cpu" else 2.0
        return {"class": self.res, "ram_gb": self.ram_gb, "gpu_mem_gb": gpu_mem}

    def post_res_dict(self, res: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``res`` of the posthoc jobs: ``post_res`` (or the class of ``res``,
        by default the study's), with ``post_gpu_mem_gb`` and ``post_ram_gb``
        where they are set."""
        base = dict(res or self.res_dict())
        if self.post_res:
            base["class"] = self.post_res
        if self.post_gpu_mem_gb is not None:
            base["gpu_mem_gb"] = self.post_gpu_mem_gb
        if self.post_ram_gb is not None:
            base["ram_gb"] = self.post_ram_gb
        return base

    def posthoc_job(
        self,
        run: Mapping[str, Any],
        *,
        nas_runs: Path,
        code: Mapping[str, Any],
        ckpts: str,
        res: Mapping[str, Any] | None = None,
        prio: int | None = None,
        trial: int | None = None,
    ) -> dict[str, Any]:
        """The ``task=posthoc`` job of the run of the job ``run`` (its
        ``job.json``): its config and overrides with ``task=posthoc``, after
        it, of the parent ``nas_runs/<batch>/<run_id>``, in the batch
        ``<batch>-post``, evaluating ``ckpts`` of its checkpoints, on ``res``
        (by default ``post_res_dict()``)."""
        batch = f"{run['batch']}-post"
        return make_job(
            batch=batch,
            config={**run["config"], "task": "posthoc"},
            overrides=[*run["overrides"], "task=posthoc"],
            code=code,
            res=dict(res) if res is not None else self.post_res_dict(),
            prio=self.prio if prio is None else prio,
            after=[run["run_id"]],
            study=None if trial is None else self.name,
            trial=trial,
            tags=["posthoc"],
            extra={
                "parent": str(nas_runs / run["batch"] / run["run_id"]),
                "decode": {"ckpts": ckpts},
            },
        )

    def sampler(self, n_trials: int = 0) -> optuna.samplers.BaseSampler:
        """TPE over the conditional space (design 5.2); ``constant_liar`` keeps
        the trials in flight apart. Its seed is ``sampler_seed + n_trials``
        for a study that has ``n_trials`` already, so a restarted coordinator
        does not draw the same random trials again."""
        with warnings.catch_warnings():  # `group` is still "experimental"
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            return optuna.samplers.TPESampler(
                multivariate=True,
                group=True,
                constant_liar=True,
                n_startup_trials=self.n_startup_trials,
                seed=self.sampler_seed + n_trials,
            )

    def pruner(self) -> optuna.pruners.BasePruner:
        """ASHA over the virtual time: rungs at ``min_resource`` seconds, then
        times ``reduction_factor`` (300, 900, ... by default)."""
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=self.min_resource, reduction_factor=self.reduction_factor
        )


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def open_study(
    conf: StudyConf,
    storage: str | optuna.storages.BaseStorage,
    *,
    code: Mapping[str, Any],
) -> optuna.Study:
    """The study ``conf.name`` on ``storage``: created (bound to the snapshot
    ``code["snap"]`` and to ``conf``, the ``[[enqueue]]`` trials enqueued) or
    loaded, checked by ``check_bound`` and given ``conf``'s operational
    values."""
    if isinstance(storage, str):
        storage = optuna.storages.get_storage(storage)
    names = {s.study_name for s in storage.get_all_studies()}
    kw = {"sampler": conf.sampler(), "pruner": conf.pruner()}
    if conf.name in names:
        study = optuna.load_study(study_name=conf.name, storage=storage, **kw)
        check_bound(study, conf, code)
        study.sampler = conf.sampler(len(study.get_trials(deepcopy=False)))
        if _canonical(study.user_attrs["conf"]) != _canonical(conf.to_dict()):
            study.set_user_attr("conf", conf.to_dict())
        return study
    study = optuna.create_study(
        study_name=conf.name, storage=storage, direction="minimize", **kw
    )
    study.set_user_attr("snap", code["snap"])
    study.set_user_attr("conf", conf.to_dict())
    for params in conf.enqueue:
        study.enqueue_trial(params)
    return study


def check_bound(study: optuna.Study, conf: StudyConf, code: Mapping[str, Any]) -> None:
    """``StudyError`` unless ``study`` was opened with ``conf`` on ``code``."""
    snap = study.user_attrs.get("snap")
    if snap != code["snap"]:
        raise StudyError(
            f"study {study.study_name} is bound to the snapshot {str(snap)[:12]},"
            f" not {code['snap'][:12]}: new code is a new study (-v2)"
        )
    if _canonical(_binding(study.user_attrs.get("conf"))) != _canonical(
        _binding(conf.to_dict())
    ):
        raise StudyError(
            f"study {study.study_name} has another definition than this TOML:"
            " a changed study is a new study (-v2)"
        )


def _binding(d: Mapping[str, Any] | None) -> Any:
    """The part of a definition (``StudyConf.to_dict``) that binds a study."""
    if d is None:
        return None
    study = {k: v for k, v in d["study"].items() if k not in _OPERATIONAL}
    return {**d, "study": study}


def backup(db: Path, dst: Path) -> None:
    """Copy the sqlite database ``db`` to ``dst`` (on the NAS): a consistent
    copy by sqlite's backup API next to ``db``, then its bytes to a temporary
    name beside ``dst`` and a rename (no sqlite locking on the CIFS mount)."""
    tmp = db.with_name(f".{db.name}.backup")
    src, out = sqlite3.connect(db), sqlite3.connect(tmp)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    part = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    nas_retry(lambda: dst.parent.mkdir(parents=True, exist_ok=True))
    nas_retry(lambda: shutil.copyfile(tmp, part))
    nas_retry(lambda: os.replace(part, dst))
    tmp.unlink()
