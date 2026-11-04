"""The search space of a study: conditional parameters and their overrides."""

import optuna
import pytest
from sweep.space import Space, override_value

SPACE = {
    "optim": {"categorical": ["sgd", "adamw", "graft_adam"]},
    "optim.lr": {"float": [0.001, 0.3], "log": True, "when": "optim != 'adamw'"},
    "optim.lr@adamw": {
        "float": [1e-5, 1e-2],
        "log": True,
        "when": "optim == 'adamw'",
    },
    "optim.tau_m": {"float": [0.03, 3.0], "log": True, "when": "optim != 'sgd'"},
    "optim.warmup": {"categorical": [0.0, 60.0, 300.0]},
    "dynamics.residual": {"categorical": [True, False]},
    "dynamics.open_cycle": {"int": [10, 40], "when": "dynamics.residual == True"},
}


class TestSuggest:
    def test_a_fixed_trial_gives_only_the_active_parameters(self):
        space = Space.from_dict(SPACE)
        fixed = {
            "optim": "adamw",
            "optim.lr@adamw": 1e-4,
            "optim.tau_m": 0.3,
            "optim.warmup": 60.0,
            "dynamics.residual": False,
        }
        # FixedTrial raises on a parameter it was not given: the inactive ones
        # (optim.lr, dynamics.open_cycle) are never asked for.
        params = space.suggest(optuna.trial.FixedTrial(fixed))
        assert params == fixed

    def test_the_conditions_hold_on_every_sampled_trial(self):
        space = Space.from_dict(SPACE)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        kinds = set()
        for _ in range(40):
            params = space.suggest(study.ask())
            kinds.add(params["optim"])
            assert ("optim.lr" in params) == (params["optim"] != "adamw")
            assert ("optim.lr@adamw" in params) == (params["optim"] == "adamw")
            assert ("optim.tau_m" in params) == (params["optim"] != "sgd")
            assert ("dynamics.open_cycle" in params) == params["dynamics.residual"]
            if "optim.lr@adamw" in params:
                assert 1e-5 <= params["optim.lr@adamw"] <= 1e-2
            if "dynamics.open_cycle" in params:
                assert isinstance(params["dynamics.open_cycle"], int)
        assert kinds == {"sgd", "adamw", "graft_adam"}

    def test_in_and_not_and(self):
        space = Space.from_dict(
            {
                "optim": {"categorical": ["sgd", "adamw", "graft_adam"]},
                "a": {
                    "categorical": [1],
                    "when": "optim in ['adamw', 'graft_adam'] and not optim == 'sgd'",
                },
            }
        )
        assert "a" in space.suggest(optuna.trial.FixedTrial({"optim": "adamw", "a": 1}))
        assert "a" not in space.suggest(optuna.trial.FixedTrial({"optim": "sgd"}))


class TestOverrides:
    def test_the_name_after_at_is_dropped_from_the_key(self):
        space = Space.from_dict(SPACE)
        params = {
            "optim": "adamw",
            "optim.lr@adamw": 1e-05,
            "optim.tau_m": 0.3,
            "optim.warmup": 300.0,
            "dynamics.residual": True,
            "dynamics.open_cycle": 20,
        }
        assert space.overrides(params) == [
            "optim=adamw",
            "optim.lr=1e-05",
            "optim.tau_m=0.3",
            "optim.warmup=300.0",
            "dynamics.residual=true",
            "dynamics.open_cycle=20",
        ]

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (True, "true"),
            (False, "false"),
            (None, "null"),
            (3, "3"),
            (0.03, "0.03"),
            (1e-05, "1e-05"),
            (300.0, "300.0"),
            ("graft_adam", "graft_adam"),
        ],
    )
    def test_values_are_written_as_hydra_reads_them(self, value, text):
        assert override_value(value) == text

    def test_a_float_round_trips_through_its_text(self):
        x = 0.012345678901234567
        assert float(override_value(x)) == x

    def test_two_active_parameters_on_one_key_are_refused(self):
        space = Space.from_dict(
            {"optim.lr": {"float": [0.1, 1.0]}, "optim.lr@b": {"float": [0.1, 1.0]}}
        )
        with pytest.raises(ValueError, match="optim.lr"):
            space.suggest(optuna.trial.FixedTrial({"optim.lr": 0.5, "optim.lr@b": 0.5}))


class TestFromDict:
    @pytest.mark.parametrize(
        "bad",
        [
            {"a": {"float": [0.1, 1.0], "when": "b == 1"}},  # unknown name
            {
                "a": {"float": [0.1, 1.0], "when": "b == 1"},
                "b": {"categorical": [1]},
            },  # a later parameter
            {"a": {"float": [1.0]}},  # not [low, high]
            {"a": {"uniform": [0.0, 1.0]}},  # unknown kind
            {"a": {"float": [0.0, 1.0], "log": True}},  # log needs low > 0
            {"a": {"categorical": []}},
            {"a": {"categorical": [1], "when": "__import__('os')"}},  # not a condition
            {"a": {"categorical": [1], "colour": 1}},  # unknown key
            {"a": {"float": [0.1, 1.0], "log": True, "step": 0.1}},  # optuna refuses
            {"a": {"int": [1, 8], "log": True, "step": 2}},  # optuna refuses
        ],
    )
    def test_malformed_spaces_are_refused(self, bad):
        with pytest.raises(ValueError):
            Space.from_dict(bad)

    def test_check_names_the_unknown_parameters(self):
        space = Space.from_dict(SPACE)
        space.check({"optim": "sgd", "optim.lr": 0.03})
        with pytest.raises(ValueError, match="optim.beta"):
            space.check({"optim.beta": 0.9})
