"""The coordinator of one study (design 5.1, 5.2): ask, queue, watch, tell.

Every ``step`` it reads the NAS and, for each running trial:

1. a run that failed (``failed``, ``corrupt``, stopped by someone else) fails
   the trial (``FAIL``: an exception, the only case that is not a value) and
   stops its other runs; a run stopped as ``pruned`` (by an earlier
   coordinator that died before telling) prunes it;
2. reports the mean $J_s$ at each evaluation time all its seeds reached
   (``sweep.objective.reports``) and asks ASHA after each; a pruned trial's
   runs are stopped at once (``.stop``, reason ``pruned``) and it is told
   ``PRUNED``;
3. once every seed completed or diverged, tells the mean (a diverged seed
   counts ``log 10``).

With ``objective_from = "posthoc"`` (M17) there is no report (and so no
pruning): once every seed ended, each completed run gets a ``task=posthoc``
job (``StudyConf.posthoc_job``, its run_id kept in the trial's user attribute
``post``), and the trial is told the mean of their
``final.decode.val.objective`` (``sweep.objective.posthoc_value``; a diverged
seed ``log 10``) once all completed; a posthoc job that fails fails the trial.

Then it asks new trials until ``parallel`` are in flight or ``max_trials``
were asked. A trial's runs are ``overrides`` + its parameters + ``seed=s``,
composed in the study's snapshot and queued as ``fleet submit`` queues them
(``exp.lib.fleet.jobs.make_job``, the same run_ids, a run already queued,
running or finished for good is not queued again). The trial keeps its
overrides (user attribute ``overrides``, once suggested) and then its run_ids
(``runs``, once composed), so a new coordinator on the same storage goes on
where the last one stopped.

What goes wrong in the fleet rather than in a trial does not cost trials: a
compose that breaks (``RuntimeError``, ``OSError``: the snapshot's venv) or a
NAS error leaves the trial running with what it has, and the next step tries
again (a trial with overrides but no runs is composed again; a run absent
from the NAS at two steps in a row is queued again, as a worker requeues a
claim only after it missed it twice). Only a config that does not compose
(``ValueError``) fails a trial. A trial whose run is being stopped for
another trial (a claim with a ``.stop``) waits for that stop to end, then
queues the run anew.
"""

import logging
import threading
from collections.abc import Callable, Mapping, Sequence, Set
from typing import Any

import optuna

from exp.lib.fleet.contract import read_json
from exp.lib.fleet.jobs import make_job
from exp.lib.fleet.queue import Nas, nas_retry
from sweep.fleetio import FleetView
from sweep.objective import DIVERGED, SeedRun, posthoc_value, reports, trial_value
from sweep.study import StudyConf, check_bound

__all__ = ["CODE_KEYS", "Composer", "Coordinator", "StepError", "code_of"]

log = logging.getLogger("sweep")

# The resolved configs of override lists (ValueError when one does not
# compose), as exp.lib.fleet.jobs.compose_configs in a snapshot.
type Composer = Callable[[Sequence[Sequence[str]]], list[dict[str, Any]]]

CODE_KEYS = ("snap", "base_head", "dirty", "mujou_snap", "mujou_pushed", "envkey")
State = optuna.trial.TrialState
_PENDING = frozenset({"queued", "running", "absent"})
_ENDED = frozenset({"completed", "diverged"})


class StepError(Exception):
    """A step could not be finished for a reason of the fleet (the compose of
    the snapshot broke); ``serve`` tries again at the next step."""


def code_of(info: Mapping[str, Any]) -> dict[str, Any]:
    """The ``code`` of a job from a snapshot's ``CODE.json``."""
    return {k: info[k] for k in CODE_KEYS}


class Coordinator:
    """Runs the study ``study`` of the definition ``conf`` on the fleet whose
    NAS is ``nas``, with the code of the snapshot ``code`` (its ``CODE.json``);
    ``StudyError`` when the study is bound to another snapshot or definition."""

    def __init__(
        self,
        conf: StudyConf,
        study: optuna.Study,
        *,
        nas: Nas,
        code: Mapping[str, Any],
        composer: Composer,
    ) -> None:
        check_bound(study, conf, code)
        self._conf = conf
        self._study = study
        self._nas = nas
        self._code = code_of(code)
        self._compose = composer
        self._view = FleetView(nas)
        # Consecutive steps at which a run was absent from the NAS.
        self._absent: dict[str, int] = {}
        # The jobs of a trial, by number, once composed (no compose again).
        self._jobs: dict[int, list[dict[str, Any]]] = {}

    def step(self) -> None:
        """Follow every running trial, then ask new ones into the free places.
        A running trial without runs is queued again (its compose broke, or a
        run of the same params was being stopped), or failed when it has no
        overrides (a coordinator stopped between asking and suggesting).
        ``StepError`` after following the others when a compose broke; then no
        new trial is asked."""
        self._view.refresh()
        absent, self._absent = self._absent, {}
        broken: StepError | None = None
        for t in self._study.get_trials(deepcopy=False, states=(State.RUNNING,)):
            try:
                if "runs" in t.user_attrs:
                    self._follow(t, absent)
                elif "overrides" in t.user_attrs:
                    self._queue(optuna.Trial(self._study, t._trial_id), t.params)
                else:
                    self._fail(t, [], "the coordinator stopped before suggesting it")
            except StepError as e:
                broken = e
        if broken is not None:
            raise broken
        self._fill()

    def finished(self) -> bool:
        """Whether ``max_trials`` were asked and none is in flight."""
        trials = self._study.get_trials(deepcopy=False)
        asked = [t for t in trials if t.state != State.WAITING]
        return len(asked) >= self._conf.max_trials and not any(
            t.state == State.RUNNING for t in asked
        )

    def serve(
        self,
        stop: threading.Event,
        *,
        every: float = 60.0,
        after_step: Callable[[], None] | None = None,
    ) -> None:
        """``step`` every ``every`` seconds until ``finished`` or ``stop`` is
        set. An ``OSError`` (the NAS) or a ``StepError`` (the compose) skips
        one step; anything else raises."""
        while not stop.is_set():
            try:
                self.step()
            except OSError, StepError:
                log.exception("a step failed; again in %.0f s", every)
            if after_step is not None:
                after_step()
            if self.finished():
                return
            stop.wait(every)

    # -- one trial -------------------------------------------------------------

    def _follow(
        self, frozen: optuna.trial.FrozenTrial, absent: Mapping[str, int]
    ) -> None:
        """Tell or report ``frozen`` from its runs; ``absent`` counts the steps
        before this one at which each run was absent."""
        runs: list[str] = list(frozen.user_attrs["runs"])
        seeds = [self._view.seed_run(r) for r in runs]
        for r, s in zip(runs, seeds, strict=True):
            if s.status == "absent":
                self._absent[r] = absent.get(r, 0) + 1
        again = {r for r in runs if self._absent.get(r, 0) >= 2}
        if again:
            log.info("trial %d: queued again %s", frozen.number, " ".join(again))
            self._queue(
                optuna.Trial(self._study, frozen._trial_id), frozen.params, again
            )
        if any(s.status == "stopped" and s.reason == "pruned" for s in seeds):
            self._stop(frozen, runs, reason="pruned")
            self._tell(frozen, state=State.PRUNED)
            return
        bad = [(r, s) for r, s in zip(runs, seeds, strict=True) if not _fine(s.status)]
        if bad:
            r, s = bad[0]
            self._fail(frozen, runs, f"{r}: {s.status} ({s.reason})")
            return
        if self._conf.objective_from == "posthoc":
            self._follow_posthoc(frozen, runs, seeds, absent)
            return
        trial = optuna.Trial(self._study, frozen._trial_id)
        last = frozen.last_step if frozen.last_step is not None else -1
        ended = all(s.status in _ENDED for s in seeds)
        for step, value in reports(seeds, after=last):
            trial.report(value, step)
            # A trial that ran to its end (seen late) is told its value.
            if not ended and trial.should_prune():
                log.info("trial %d pruned at %d s: %.4f", frozen.number, step, value)
                self._stop(frozen, runs, reason="pruned")
                self._tell(frozen, state=State.PRUNED)
                return
        if ended:
            try:
                value = trial_value(seeds)
            except ValueError as e:
                self._fail(frozen, runs, str(e))
                return
            self._tell(frozen, value=value)

    def _follow_posthoc(
        self,
        frozen: optuna.trial.FrozenTrial,
        runs: Sequence[str],
        seeds: Sequence[SeedRun],
        absent: Mapping[str, int],
    ) -> None:
        """Queue the posthoc job of each completed run of ``frozen`` once all
        its seeds ended, and tell the mean of their objectives once all
        completed (module docstring)."""
        if not all(s.status in _ENDED for s in seeds):
            return
        trial = optuna.Trial(self._study, frozen._trial_id)
        posts: dict[str, str] = dict(frozen.user_attrs.get("post", {}))
        values: list[float] = []
        waiting = False
        for r, s in zip(runs, seeds, strict=True):
            if s.status == "diverged":
                values.append(DIVERGED)
                continue
            post = posts.get(r)
            if post is None:
                posts[r] = self._queue_posthoc(r, frozen.number)
                trial.set_user_attr("post", posts)
                waiting = True
                continue
            seen = self._view.seed_run(post)
            if seen.status == "absent":
                self._absent[post] = absent.get(post, 0) + 1
                if self._absent[post] >= 2:  # queued again, as a run is
                    self._queue_posthoc(r, frozen.number)
            if seen.status in _PENDING:
                waiting = True
            elif seen.status == "completed":
                try:
                    values.append(posthoc_value(seen))
                except ValueError as e:
                    self._fail(frozen, runs, f"{post}: {e}")
                    return
            else:
                self._fail(
                    frozen, runs, f"posthoc {post}: {seen.status} ({seen.reason})"
                )
                return
        if not waiting:
            self._tell(frozen, value=sum(values) / len(values))

    def _queue_posthoc(self, run_id: str, number: int) -> str:
        """Queue the posthoc job of the completed run ``run_id`` (unless it is
        taken already); its run_id."""
        conf = self._conf
        run = nas_retry(
            lambda: read_json(self._nas.runs / conf.batch / run_id / "job.json")
        )
        job = conf.posthoc_job(
            run,
            nas_runs=self._nas.runs,
            code=self._code,
            ckpts=conf.post_ckpts,
            trial=number,
        )
        post = job["run_id"]
        if not self._view.is_taken(post):
            self._nas.drop_done(post)
            self._view.queued(post, self._nas.submit(job))
            log.info("trial %d: posthoc %s of %s", number, post, run_id)
        self._absent.pop(post, None)
        return post

    def _fail(
        self, frozen: optuna.trial.FrozenTrial, runs: Sequence[str], why: str
    ) -> None:
        log.warning("trial %d failed: %s", frozen.number, why)
        optuna.Trial(self._study, frozen._trial_id).set_user_attr("error", why)
        self._stop(frozen, runs, reason="trial-failed")
        self._tell(frozen, state=State.FAIL)

    def _tell(
        self,
        frozen: optuna.trial.FrozenTrial,
        *,
        value: float | None = None,
        state: optuna.trial.TrialState | None = None,
    ) -> None:
        self._study.tell(frozen.number, value, state=state)
        self._jobs.pop(frozen.number, None)
        told = state.name if state is not None else f"COMPLETE {value:+.4f}"
        log.info("trial %d: %s", frozen.number, told)

    def _stop(
        self, frozen: optuna.trial.FrozenTrial, runs: Sequence[str], *, reason: str
    ) -> None:
        """Stop the runs of ``runs`` that no other running trial shares."""
        shared = {
            r
            for t in self._study.get_trials(deepcopy=False, states=(State.RUNNING,))
            if t.number != frozen.number
            for r in t.user_attrs.get("runs", [])
        }
        for r in runs:
            if r not in shared and self._view.seed_run(r).status in _PENDING:
                self._view.stop(r, reason=reason)

    # -- new trials ------------------------------------------------------------

    def _fill(self) -> None:
        while True:
            trials = self._study.get_trials(deepcopy=False)
            asked = [t for t in trials if t.state != State.WAITING]
            in_flight = sum(t.state == State.RUNNING for t in asked)
            if in_flight >= self._conf.parallel or len(asked) >= self._conf.max_trials:
                return
            trial = self._study.ask()
            try:
                params = self._conf.space.suggest(trial)
            except ValueError as e:
                trial.set_user_attr("error", str(e))
                self._study.tell(trial, state=State.FAIL)
                continue
            trial.set_user_attr("overrides", self._conf.space.overrides(params))
            self._queue(trial, params)

    def _queue(
        self,
        trial: optuna.Trial,
        params: Mapping[str, Any],
        only: Set[str] | None = None,
    ) -> None:
        """Compose and queue the runs of ``trial`` (those of ``only``, when
        given, else all; those not taken already). A config that does not
        compose fails the trial; a compose that breaks raises ``StepError``
        (the trial waits for the next step). A trial without runs yet whose
        run is being stopped waits too."""
        jobs = self._jobs.get(trial.number)
        if jobs is None:
            jobs = self._compose_jobs(trial, params)
            if jobs is None:
                return
            self._jobs[trial.number] = jobs
        if only is None:
            stopping = [j["run_id"] for j in jobs if self._view.stopping(j["run_id"])]
            if stopping:
                log.info("trial %d waits for the stop of %s", trial.number, stopping[0])
                return
            # The run_ids go first: a coordinator that stops after this line
            # queues the missing runs at its next start.
            trial.set_user_attr("runs", [j["run_id"] for j in jobs])
        for job in jobs:
            run_id = job["run_id"]
            if (only is not None and run_id not in only) or self._view.is_taken(run_id):
                continue
            # A failed or stopped record of the run goes, so the new one shows.
            self._nas.drop_done(run_id)
            self._view.queued(run_id, self._nas.submit(job))
            self._absent.pop(run_id, None)
        if only is None:
            log.info(
                "trial %d queued: %s", trial.number, " ".join(jobs[0]["overrides"])
            )

    def _compose_jobs(
        self, trial: optuna.Trial, params: Mapping[str, Any]
    ) -> list[dict[str, Any]] | None:
        """The jobs of ``trial``'s seeds; ``None`` (the trial failed) when its
        config does not compose, ``StepError`` when the compose broke."""
        conf = self._conf
        overrides = conf.space.overrides(params)
        lists = [[*conf.overrides, *overrides, f"seed={s}"] for s in conf.seeds]
        try:
            configs = self._compose(lists)
        except ValueError as e:
            log.warning("trial %d does not compose: %s", trial.number, e)
            trial.set_user_attr("error", str(e))
            self._study.tell(trial, state=State.FAIL)
            return None
        except (RuntimeError, OSError) as e:
            raise StepError(f"trial {trial.number}: the compose broke: {e}") from e
        return [
            make_job(
                batch=conf.batch,
                config=config,
                overrides=o,
                code=self._code,
                res=conf.res_dict(),
                prio=conf.prio,
                study=conf.name,
                trial=trial.number,
                tags=["explore"],
            )
            for o, config in zip(lists, configs, strict=True)
        ]


def _fine(status: str) -> bool:
    return status in _PENDING or status in _ENDED
