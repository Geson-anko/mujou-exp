"""``exp.vision.jepa_iid`` on the tiny synthetic Walking Tours of
``test_eval`` (videos A, B, C of 20 frames, H held out; stacks of 2 frames of
16 px, patch 4: the 4 x 4 grid of V18; patch 1: the 16 x 16 grid of V18b's
patch 4 at 64 px). The defaults reproduce the values of V18's code."""

import json
import math
from pathlib import Path

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.vision.compose import from_dict, to_dict
from exp.vision.jepa import build_jepa, jepa_mup_lr
from exp.vision.jepa_iid import (
    RUNS,
    V16_RUN,
    _optimizer,
    ckpt_steps,
    main,
    v16_config,
)
from exp.vision.schema import DataConf, ModelConf, OptimConf, RunConf, ToursConf
from tests.vision.test_eval import SIZE, N, random_tours


def conf_of() -> RunConf:
    return RunConf(
        name="test",
        data=DataConf(tours=ToursConf(held_out="H", size=SIZE, frames=N)),
        model=ModelConf(width=128, patch=4, param="mup"),
        optim=OptimConf(kind="rmsprop"),
    )


@pytest.fixture
def root(tmp_path) -> Path:
    return random_tours(tmp_path / "d")


def run(root: Path, tmp_path: Path, name: str, *extra: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(to_dict(conf_of())))
    out = tmp_path / name
    argv = ["--config", str(cfg), "--mask-ratio", "0.5", "--lam", "0.5"]
    argv += ["--lr", "1e-3", "--warmup", "2", "--directions", "64"]
    argv += ["--device", "cpu", "--root", str(root), "--out", str(out), *extra]
    main(argv)
    return out


def rows(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]


class TestCheckpoints:
    def test_a_short_run_saves_at_0_1000_2000_4000_6000(self):
        assert ckpt_steps(6000) == [0, 1000, 2000, 4000, 6000]
        assert ckpt_steps(7) == [0, 7]

    def test_a_long_run_saves_at_the_long_steps(self):
        assert ckpt_steps(60000) == [0, 2000, 6000, *range(10000, 60001, 10000)]

    def test_a_step_is_12_8_s_at_batch_128(self, root, tmp_path):
        out = run(root, tmp_path, "r", "--steps", "1", "--batch", "128")
        times = sorted(float(p.stem) for p in (out / "ckpt").glob("*.pt"))
        assert times == [0.0, 12.8]
        mods = build_jepa(conf_of(), seed=0)
        head = ckpt.load_eval(ckpt.eval_path(out, 12.8), mods.modules())
        assert head["frame"] == 128 and head["program"] == "jepa"


class TestConfig:
    def test_the_config_of_v16s_run_is_read_when_there(self, tmp_path):
        d = to_dict(conf_of())
        (tmp_path / "v14-width" / V16_RUN).mkdir(parents=True)
        (tmp_path / "v14-width" / V16_RUN / "config.yaml").write_text(yaml.safe_dump(d))
        assert v16_config(tmp_path) == d

    def test_without_it_the_schema_gives_the_same_model_and_data(self, tmp_path):
        built = from_dict(v16_config(tmp_path))
        assert built.model.width == 1024 and built.model.param == "mup"
        assert built.model.scale_decoder and built.seed == 0
        local = sorted(RUNS.glob(f"*/{V16_RUN}/config.yaml"))
        if local:  # the run's own config, where runs/ holds it
            conf = from_dict(yaml.safe_load(local[0].read_text()))
            assert built.model == conf.model and built.data == conf.data
            assert built.seed == conf.seed and built.time == conf.time
            assert built.measure.clips == conf.measure.clips
            assert built.measure.windows == conf.measure.windows


class TestOptimizer:
    def test_mup_groups_scale_the_learning_rate_by_jepa_mup_lr(self):
        conf = conf_of()
        mods = build_jepa(conf, seed=0)
        opt = _optimizer(mods, conf, 1e-3)
        mult = jepa_mup_lr(mods, conf.model.width)
        seen = 0
        for g in opt.param_groups:
            for p in g["params"]:
                assert g["lr"] == pytest.approx(1e-3 * mult[p])
                seen += 1
        assert seen == len(mods.parameters())
        # width 128 = 256 / 2: hidden 2; P and embed (outputs of the width) 2
        assert {g["mult"] for g in opt.param_groups} == {1.0, 2.0}


# train.jsonl of run(..., "--steps", "2", "--batch", "4", "--log-every", "1")
# by V18's code (commit 3bf4998, before V18b's arguments), on the CPU
V18_ROWS = (
    {"loss": 3.26409912109375, "pred": 2.3266332149505615, "sigreg": 4.201564788818359},
    {
        "loss": 2.873110294342041,
        "pred": 2.6075186729431152,
        "sigreg": 3.138702154159546,
    },
)
V18_WEIGHT_SUM = 3424.9473013582033  # of the last checkpoint's modules


class TestV18Defaults:
    def test_the_defaults_give_v18s_losses_and_weights(self, root, tmp_path):
        out = run(
            root, tmp_path, "r", "--steps", "2", "--batch", "4", "--log-every", "1"
        )
        for got, want in zip(rows(out), V18_ROWS, strict=True):
            for k, v in want.items():
                assert got[k] == pytest.approx(v, rel=1e-5), k
        w = torch.load(ckpt.eval_path(out, 0.8), weights_only=True)["modules"]
        total = sum(float(v.double().sum()) for m in w.values() for v in m.values())
        assert total == pytest.approx(V18_WEIGHT_SUM, rel=1e-6)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["patch"] == 4 and settings["grid"] == 4
        assert settings["expand"] == 1 and settings["k_fine"] == settings["k"] == 8
        assert settings["token_dim"] == 64 and settings["sigreg_on"] == "flat"
        assert settings["sigreg_dim"] == 1024 and not settings["checkpoint_blocks"]


class TestV18bArms:
    """The arms of V18b on the grid 16 x 16 (patch 1 at 16 px): F (d 4, flat)
    and M (d 64, mean)."""

    @pytest.mark.parametrize(
        ("d", "on", "dim"), [("4", "flat", 1024), ("64", "mean", 64)]
    )
    def test_the_arm_is_recorded_in_the_config_and_settings(
        self, root, tmp_path, d, on, dim
    ):
        extra = ("--patch", "1", "--token-dim", d, "--sigreg-on", on)
        out = run(root, tmp_path, "r", "--steps", "2", "--batch", "4", *extra)
        conf = from_dict(yaml.safe_load((out / "config.yaml").read_text()))
        assert conf.model.patch == 1
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["patch"] == 1 and settings["grid"] == 16
        assert settings["coarse_grid"] == 4 and settings["expand"] == 4
        assert settings["k"] == 8 and settings["k_fine"] == 128
        assert settings["tokens"] == 256 and settings["family"] == 436
        assert settings["token_dim"] == int(d) and settings["sigreg_on"] == on
        assert settings["sigreg_dim"] == dim
        got = rows(out)
        keys = ("loss", "pred", "sigreg", "z_var", "rho_in", "zbar_var")
        assert all(math.isfinite(r[k]) for r in got for k in keys)
        mods = build_jepa(conf, seed=0, token_dim=int(d))
        head = ckpt.load_eval(ckpt.eval_path(out, 0.8), mods.modules())
        assert head["frame"] == 8

    def test_checkpointing_the_blocks_gives_the_same_run(self, root, tmp_path):
        base = ("--steps", "2", "--batch", "4", "--log-every", "1", "--patch", "2")
        a = run(root, tmp_path, "a", *base)
        b = run(root, tmp_path, "b", *base, "--checkpoint-blocks")
        strip = [{k: v for k, v in r.items() if k != "sec"} for r in rows(a)]
        assert strip == [{k: v for k, v in r.items() if k != "sec"} for r in rows(b)]
        assert json.loads((b / "jepa.json").read_text())["checkpoint_blocks"]

    def test_a_grid_of_no_whole_expansion_is_refused(self, root, tmp_path):
        with pytest.raises(ValueError, match="grid"):
            run(root, tmp_path, "r", "--steps", "1", "--patch", "8")


class TestTrain:
    def test_a_tiny_run_logs_finite_means_and_its_settings(self, root, tmp_path):
        out = run(
            root, tmp_path, "r", "--steps", "5", "--batch", "4", "--log-every", "2"
        )
        got = rows(out)
        assert [r["step"] for r in got] == [2, 4, 5]
        keys = ("loss", "pred", "sigreg", "grad_norm", "z_var", "rho_in")
        assert all(math.isfinite(r[k]) for r in got for k in keys)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["k"] == 8 and settings["family"] == 436
        assert settings["units"] == 60 and settings["ckpt_steps"] == [0, 5]
        prov = settings["provenance"]  # CODE.json is there only in a snapshot
        assert {"code", "host", "argv", "env", "torch"} <= set(prov)

    def test_the_same_seed_gives_the_same_run(self, root, tmp_path):
        a, b = (
            run(
                root, tmp_path, name, "--steps", "3", "--batch", "4", "--log-every", "1"
            )
            for name in ("a", "b")
        )
        strip = [{k: v for k, v in r.items() if k != "sec"} for r in rows(a)]
        assert strip == [{k: v for k, v in r.items() if k != "sec"} for r in rows(b)]
        wa = torch.load(ckpt.eval_path(a, 1.2), weights_only=True)["modules"]
        wb = torch.load(ckpt.eval_path(b, 1.2), weights_only=True)["modules"]
        assert all(torch.equal(wa[m][k], wb[m][k]) for m in wa for k in wa[m])

    def test_a_lam_outside_0_1_is_refused(self, root, tmp_path):
        with pytest.raises(ValueError, match="lam"):
            run(root, tmp_path, "r", "--steps", "1", "--lam", "1.5")

    @pytest.mark.slow
    def test_the_loss_goes_down_on_the_tiny_tours(self, root, tmp_path):
        out = run(
            root, tmp_path, "r", "--steps", "200", "--batch", "32", "--log-every", "20"
        )
        got = rows(out)
        assert got[-1]["loss"] < 0.5 * got[0]["loss"]
