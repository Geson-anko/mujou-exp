"""After a search: the runs whose numbers go to the README (design 5.4, 0.3).

``confirm`` runs the top ``k`` trials (by their value on the search seeds) for
each learner on the confirmation seeds; ``bracket`` runs one trial with one
of its parameters times each factor (x0.5, x1, x2 of the rate) for each
learner, to see whether the value BP found is also PC-ALM's best. Both use
``phase=confirm`` (the test clips, the checkpoints and the pictures), need a
clean snapshot with mujou pushed, and fix one device class for all their
runs. They queue through the fleet's duplicate check: a run that is queued,
running or finished for good is not queued again, so the x1 runs of a bracket
are the confirm runs.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import optuna

from exp.lib.fleet.jobs import GPU_ANY, make_job
from exp.lib.fleet.queue import Nas
from sweep.coordinator import Composer, code_of
from sweep.fleetio import FleetView
from sweep.study import StudyConf

__all__ = ["FollowupError", "bracket", "confirm"]

CONFIRM_PRIO = 20  # before the searches (60): PC-ALM + trace decides a batch's end


class FollowupError(Exception):
    """The follow-up cannot be queued as asked."""


def confirm(
    study: optuna.Study,
    *,
    nas: Nas,
    code: Mapping[str, Any],
    composer: Composer,
    top: int,
    learners: Sequence[str],
    seeds: Sequence[int],
    batch: str,
    res: Mapping[str, Any] | None = None,
    prio: int = CONFIRM_PRIO,
    dry: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Queue the top ``top`` completed trials of distinct parameters x
    ``learners`` x ``seeds`` as the batch ``batch`` with the code ``code`` (a
    ``CODE.json``), on ``res`` (the study's by default). ``("new" |
    "duplicate", job)`` of each run, in order."""
    conf = StudyConf.of_study(study)
    _check_seeds(conf, seeds)
    trials, seen = [], set()
    for t in _completed(study):
        key = tuple(conf.space.overrides(t.params))
        if key not in seen:
            seen.add(key)
            trials.append(t)
    if top > len(trials):
        raise FollowupError(
            f"{study.study_name} has {len(trials)} completed trials of distinct"
            f" parameters, not {top}"
        )
    runs = [
        (
            _learner_overrides(conf, conf.space.overrides(t.params), learner, s),
            ["confirm", f"{study.study_name}/{t.number}"],
        )
        for t in trials[:top]
        for learner in learners
        for s in seeds
    ]
    return _queue(nas, code, composer, conf, runs, batch, res, prio, dry)


def bracket(
    study: optuna.Study,
    *,
    nas: Nas,
    code: Mapping[str, Any],
    composer: Composer,
    trial: int | None,
    param: str,
    factors: Sequence[float],
    learners: Sequence[str],
    seeds: Sequence[int],
    batch: str,
    res: Mapping[str, Any] | None = None,
    prio: int = CONFIRM_PRIO,
    dry: bool = False,
) -> list[tuple[str, dict[str, Any]]]:
    """Queue the trial number ``trial`` (the best completed one when ``None``)
    with its float parameter of the override key ``param`` times each of
    ``factors``, x ``learners`` x ``seeds``; as ``confirm`` otherwise."""
    conf = StudyConf.of_study(study)
    _check_seeds(conf, seeds)
    trials = _completed(study)
    if not trials:
        raise FollowupError(f"{study.study_name} has no completed trial")
    if trial is None:
        chosen = trials[0]
    else:
        found = [t for t in trials if t.number == trial]
        if not found:
            raise FollowupError(f"trial {trial} of {study.study_name} is not completed")
        chosen = found[0]
    names = [
        p.name
        for p in conf.space.params
        if p.key == param and p.kind == "float" and p.name in chosen.params
    ]
    if len(names) != 1:
        raise FollowupError(f"trial {chosen.number} has no float parameter {param}")
    name = names[0]
    runs = []
    for f in factors:
        params = {**chosen.params, name: chosen.params[name] * f}
        for learner in learners:
            for s in seeds:
                overrides = _learner_overrides(
                    conf, conf.space.overrides(params), learner, s
                )
                tags = ["bracket", f"{study.study_name}/{chosen.number}", f"x{f:g}"]
                runs.append((overrides, tags))
    return _queue(nas, code, composer, conf, runs, batch, res, prio, dry)


def _check_seeds(conf: StudyConf, seeds: Sequence[int]) -> None:
    """``FollowupError`` when ``seeds`` share one with the search: the hash of
    a run leaves ``phase`` out, so such a confirm run would be taken for the
    search's run (no test clips, no checkpoints) and skipped as a duplicate."""
    common = sorted(set(seeds) & set(conf.seeds))
    if common:
        raise FollowupError(
            f"seeds {common} are the search's ({list(conf.seeds)}): confirm on others"
        )


def _completed(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """The completed trials, best (smallest value) first."""
    done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    return sorted(done, key=lambda t: (t.value, t.number))


def _learner_overrides(
    conf: StudyConf, params: Sequence[str], learner: str, seed: int
) -> list[str]:
    """The study's overrides with ``learner`` and ``phase=confirm`` in place of
    its own, then the trial's ``params`` and the seed."""
    base = _set(_set(list(conf.overrides), "learner", learner), "phase", "confirm")
    return [*base, *params, f"seed={seed}"]


def _set(overrides: list[str], key: str, value: str) -> list[str]:
    """``overrides`` with ``key=value`` in place of the override(s) of ``key``
    (appended when there is none)."""
    out, placed = [], False
    for o in overrides:
        if o.partition("=")[0].lstrip("+~") == key:
            if not placed:
                out.append(f"{key}={value}")
                placed = True
            continue
        out.append(o)
    return out if placed else [*out, f"{key}={value}"]


def _queue(
    nas: Nas,
    code: Mapping[str, Any],
    composer: Composer,
    conf: StudyConf,
    runs: Sequence[tuple[list[str], list[str]]],
    batch: str,
    res: Mapping[str, Any] | None,
    prio: int,
    dry: bool,
) -> list[tuple[str, dict[str, Any]]]:
    if code["dirty"] or not code["mujou_pushed"]:
        raise FollowupError(
            "a confirmation needs a clean snapshot with mujou pushed (design 4.4)"
        )
    res = dict(res or conf.res_dict())
    if res["class"] == GPU_ANY:
        raise FollowupError(
            "a confirmation fixes one device class: cpu or a GPU model, not any_gpu"
        )
    try:
        configs = composer([o for o, _ in runs])
    except (ValueError, RuntimeError) as e:
        raise FollowupError(str(e)) from None
    view = FleetView(nas)
    view.refresh()
    out: list[tuple[str, dict[str, Any]]] = []
    for (overrides, tags), config in zip(runs, configs, strict=True):
        job = make_job(
            batch=batch,
            config=config,
            overrides=overrides,
            code=code_of(code),
            res=res,
            prio=prio,
            tags=tags,
        )
        run_id = job["run_id"]
        if view.is_taken(run_id):
            out.append(("duplicate", job))
            continue
        out.append(("new", job))
        if not dry:
            nas.drop_done(run_id)
            view.queued(run_id, nas.submit(job))
    return out
