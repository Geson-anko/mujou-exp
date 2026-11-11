"""The studies of M14 (``exp/video/studies/*.toml``): they load, follow the
rules of design 5.4, and every trial they can ask composes into a ``RunConf``.

Composing needs the experiment preset; a study whose preset is not written yet
is skipped (and says which).
"""

from pathlib import Path
from typing import Any

import optuna
import pytest
from sweep.study import StudyConf

ROOT = Path(__file__).resolve().parents[3]
STUDIES = sorted((ROOT / "exp" / "video" / "studies").glob("*.toml"))
EXPERIMENTS = ROOT / "exp" / "video" / "conf" / "experiment"


def test_the_m14_studies_are_there():
    # Stages 0, 1, 2 x {small without and with the trace, big with it} (design
    # 5.4: the big dynamics is searched with the trace; without, it reruns the
    # same values).
    names = {p.stem for p in STUDIES}
    want = {
        f"m14_s{s}_{size}_{trace}_bp"
        for s in (0, 1, 2)
        for size, trace in (("small", "notrace"), ("small", "trace"), ("big", "trace"))
    }
    assert want <= names


@pytest.mark.parametrize("path", STUDIES, ids=lambda p: p.stem)
class TestStudy:
    def test_the_rules_of_a_search(self, path: Path):
        conf = StudyConf.load(path)
        assert conf.name == path.stem.replace("_", "-") + "-v1"
        assert conf.seeds == (100, 101)  # the search seeds, never 0/1/2
        assert "phase=explore" in conf.overrides
        assert "learner=bp" in conf.overrides  # searched by BP, confirmed by both
        assert conf.batch == conf.name

    def test_every_trial_composes(self, path: Path):
        conf = StudyConf.load(path)
        preset = next(
            o.removeprefix("+experiment=")
            for o in conf.overrides
            if o.startswith("+experiment=")
        )
        if not (EXPERIMENTS / f"{preset}.yaml").exists():
            pytest.skip(f"conf/experiment/{preset}.yaml is not written yet")
        from exp.video.compose import compose

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for params in conf.enqueue:
            study.enqueue_trial(params)
        for _ in range(len(conf.enqueue) + 6):
            params = conf.space.suggest(study.ask())
            run = compose([*conf.overrides, *conf.space.overrides(params), "seed=100"])
            assert run.phase == "explore"
            assert run.learner.kind == "bp"
            assert run.seed == 100
            for key, value in params.items():
                node: Any = run
                for part in key.split("@")[0].split("."):
                    # an entry of a list (vision.context_masks.0.scale)
                    node = node[int(part)] if part.isdigit() else getattr(node, part)
                if key == "optim":
                    assert node.kind == value
                elif isinstance(value, float):
                    assert node == pytest.approx(value)
                else:
                    assert node == value
