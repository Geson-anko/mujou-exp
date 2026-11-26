"""``exp.vision.jepa_iid`` on the tiny synthetic Walking Tours of
``test_eval`` (videos A, B, C of 20 frames, H held out; stacks of 2 frames of
16 px, patch 4: the 4 x 4 grid of V18; patch 1: the 16 x 16 grid of V18b's
patch 4 at 64 px, and of V18c's patch 8 at 64 px with patch 2). The defaults
reproduce the values of V18's code, and of V18c's code before V18d (the
arm ``none``). V18d's arms: their settings, rows and checkpoints, the time
predictor's learning rate by Adam's first step, the arm T leaving the
spatial predictor."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from exp.lib import ckpt
from exp.lib.randproj import gaussian_projection
from exp.vision.compose import from_dict, to_dict
from exp.vision.jepa import (
    LATENT_DIM,
    R_SEED,
    TIME_KNOTS,
    W_SEED,
    build_jepa,
    jepa_mup_lr,
)
from exp.vision.jepa_eval import build_run_jepa
from exp.vision.jepa_iid import (
    DELTA_SEED,
    PERM_SEED,
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


V18C = ("--patch", "2", "--token-dim", "16", "--proj", "fixed", "--sigreg-on", "mix")


class TestV18c:
    """V18c's run on the grid 8 x 8 (patch 2 at 16 px, the grid of patch 8 at
    64 px): the fixed R, W vec(t) of LATENT_DIM for SIGReg."""

    def test_the_settings_and_the_rows_of_a_mix_run(self, root, tmp_path):
        out = run(root, tmp_path, "r", "--steps", "2", "--batch", "4", *V18C)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["grid"] == 8 and settings["expand"] == 2
        assert settings["k_fine"] == 32 and settings["tokens"] == 64
        assert settings["token_dim"] == 16 and settings["proj"] == "fixed"
        assert settings["sigreg_on"] == "mix" and settings["latent_dim"] == LATENT_DIM
        assert settings["sigreg_dim"] == LATENT_DIM
        seeds = settings["seeds"]
        assert seeds["r"] == R_SEED and seeds["w"] == W_SEED and seeds["proj"] is None
        keys = ("loss", "pred", "sigreg", "z_var", "rho_in", "zbar_var", "mix_var")
        assert all(math.isfinite(r[k]) for r in rows(out) for k in keys)

    def test_v18s_settings_record_a_learned_p_and_no_w(self, root, tmp_path):
        out = run(root, tmp_path, "r", "--steps", "1", "--batch", "4")
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["proj"] == "learned" and settings["latent_dim"] is None
        assert settings["seeds"]["r"] is None and settings["seeds"]["w"] is None
        assert "mix_var" not in rows(out)[0]

    def test_r_and_w_are_kept_bit_for_bit_and_rebuilt_from_their_seeds(
        self, root, tmp_path
    ):
        out = run(root, tmp_path, "r", "--steps", "2", "--batch", "4", *V18C)
        r = gaussian_projection(16, 128, R_SEED)
        w = gaussian_projection(LATENT_DIM, 64 * 16, W_SEED)
        for t in (0.0, 0.8):  # before the first step and after the last
            saved = torch.load(ckpt.eval_path(out, t), weights_only=True)["modules"]
            assert set(saved) == {"encoder", "proj", "predictor", "mix"}
            assert torch.equal(saved["proj"]["weight"], r)
            assert torch.equal(saved["mix"]["weight"], w)
        # the run did learn: the encoder moved
        a = torch.load(ckpt.eval_path(out, 0.0), weights_only=True)["modules"]
        b = torch.load(ckpt.eval_path(out, 0.8), weights_only=True)["modules"]
        assert any(
            not torch.equal(a["encoder"][k], b["encoder"][k]) for k in a["encoder"]
        )

    def test_no_optimizer_group_holds_r_or_w(self):
        conf = replace(conf_of(), model=replace(conf_of().model, patch=2))
        mods = build_jepa(
            conf, seed=0, token_dim=16, proj="fixed", latent_dim=LATENT_DIM
        )
        assert mods.mix is not None
        opt = _optimizer(mods, conf, 1e-3)
        held = {id(p) for g in opt.param_groups for p in g["params"]}
        assert held == {id(p) for p in mods.parameters()}
        assert id(mods.proj.weight) not in held and id(mods.mix.weight) not in held

    def test_an_unknown_proj_is_refused(self, root, tmp_path):
        with pytest.raises(SystemExit):
            run(root, tmp_path, "r", "--steps", "1", "--proj", "frozen")


# V18c's tiny run (V18C, 2 steps, batch 4) by the code before V18d
V18C_ROWS = (
    {
        "loss": 3.874279499053955,
        "pred": 3.1206271648406982,
        "sigreg": 4.627932071685791,
        "grad_norm": 17.91151237487793,
        "mix_var": 0.09812088310718536,
    },
    {
        "loss": 5.141622066497803,
        "pred": 6.26303768157959,
        "sigreg": 4.020206451416016,
        "grad_norm": 21.556522369384766,
        "mix_var": 0.0811987817287445,
    },
)
V18C_WEIGHT_SUM = 3405.743044214485
LAGS = ("--time-lags", "1:8")  # the tiny videos are of 20 frames


class TestV18d:
    """V18d's arms on V18c's grid of the tiny tours (lags 1 to 8)."""

    def test_the_arm_none_is_v18cs_run(self, root, tmp_path):
        base = ("--steps", "2", "--batch", "4", "--log-every", "1", *V18C)
        out = run(root, tmp_path, "r", *base)
        for got, want in zip(rows(out), V18C_ROWS, strict=True):
            for k, v in want.items():
                assert got[k] == pytest.approx(v, rel=1e-6), k
            assert "time" not in got
        w = torch.load(ckpt.eval_path(out, 0.8), weights_only=True)["modules"]
        assert set(w) == {"encoder", "proj", "predictor", "mix"}
        total = sum(float(v.double().sum()) for m in w.values() for v in m.values())
        assert total == pytest.approx(V18C_WEIGHT_SUM, rel=1e-9)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["time_arm"] == "none" and settings["time_knots"] is None
        assert settings["seeds"]["delta"] is None

    @pytest.mark.parametrize(
        ("arm", "weight"), [("T", None), ("ST", "0.25"), ("STperm", "1")]
    )
    def test_an_arm_records_its_settings_rows_and_tpred(
        self, root, tmp_path, arm, weight
    ):
        extra = (
            "--time-arm",
            arm,
            *(() if weight is None else ("--time-weight", weight)),
        )
        out = run(
            root,
            tmp_path,
            "r",
            "--steps",
            "3",
            "--batch",
            "6",
            "--log-every",
            "3",
            *V18C,
            *LAGS,
            *extra,
        )
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["time_arm"] == arm and settings["time_lr"] == 1e-3
        assert settings["time_weight"] == (None if weight is None else float(weight))
        assert settings["time_lags"] == [1, 8]
        assert settings["time_knots"] == list(TIME_KNOTS)
        assert settings["seeds"]["delta"] == DELTA_SEED
        assert settings["seeds"]["perm"] == (PERM_SEED if arm == "STperm" else None)
        (row,) = rows(out)
        keys = ["loss", "sigreg", "time", "grad_norm", "tpred_grad_norm", "mix_var"]
        keys += [] if arm == "T" else ["pred"]
        assert all(math.isfinite(row[k]) for k in keys)
        assert ("pred" in row) == (arm != "T")
        # lags 1..8 fall in the first band of the knots 8, 32, 128, 512
        assert row["time_bands"][1:] == [None, None]
        assert math.isfinite(row["time_bands"][0])
        assert set(row["roles"]) == {"minus", "centre", "plus"}
        assert row["roles"]["centre"]["var"][0] > 0
        conf = from_dict(yaml.safe_load((out / "config.yaml").read_text()))
        mods = build_run_jepa(conf, settings)
        ckpt.load_eval(ckpt.eval_path(out, 1.8), mods.modules())
        assert mods.tpred is not None
        assert float(mods.tpred.bias.detach().abs().max()) > 0  # it learned

    def test_the_time_predictors_first_step_is_its_lr(self, root, tmp_path):
        # Adam's first step moves a parameter of a non-zero gradient by the
        # learning rate: time_lr times lr_at(0) = time_lr / warmup
        out = run(
            root,
            tmp_path,
            "r",
            "--steps",
            "1",
            "--batch",
            "6",
            *V18C,
            *LAGS,
            "--time-arm",
            "ST",
            "--time-weight",
            "1",
            "--time-lr",
            "2e-3",
        )
        w = torch.load(ckpt.eval_path(out, 0.6), weights_only=True)["modules"]
        step = w["tpred"]["bias"].abs()
        assert float(step.max()) == pytest.approx(2e-3 / 2, rel=1e-3)
        assert float(step.max()) <= 2e-3 / 2

    def test_the_arm_t_leaves_the_spatial_predictor(self, root, tmp_path):
        out = run(
            root,
            tmp_path,
            "r",
            "--steps",
            "2",
            "--batch",
            "4",
            *V18C,
            *LAGS,
            "--time-arm",
            "T",
        )
        a = torch.load(ckpt.eval_path(out, 0.0), weights_only=True)["modules"]
        b = torch.load(ckpt.eval_path(out, 0.8), weights_only=True)["modules"]
        assert all(torch.equal(a["predictor"][k], v) for k, v in b["predictor"].items())
        assert any(not torch.equal(a["encoder"][k], v) for k, v in b["encoder"].items())

    def test_the_tpred_group_is_apart_and_without_mup(self):
        conf = replace(conf_of(), model=replace(conf_of().model, patch=2))
        mods = build_jepa(
            conf,
            seed=0,
            token_dim=16,
            proj="fixed",
            latent_dim=LATENT_DIM,
            time_knots=TIME_KNOTS,
        )
        assert mods.tpred is not None
        opt = _optimizer(mods, conf, 1e-4, 1e-3)
        tp = {id(p) for p in mods.tpred.parameters()}
        (group,) = (g for g in opt.param_groups if {id(p) for p in g["params"]} & tp)
        assert {id(p) for p in group["params"]} == tp
        assert group["peak"] == 1e-3 and group["mult"] == 1.0
        assert all(g["peak"] == 1e-4 for g in opt.param_groups if g is not group)

    @pytest.mark.parametrize(
        "extra",
        [
            ("--time-arm", "ST"),
            ("--time-arm", "T", "--time-weight", "1"),
            ("--time-weight", "1"),
            ("--time-arm", "T", "--time-lags", "9:4"),
        ],
    )
    def test_a_weight_out_of_its_arm_or_bad_lags_are_refused(
        self, root, tmp_path, extra
    ):
        with pytest.raises(ValueError, match="time_"):
            run(root, tmp_path, "r", "--steps", "1", *V18C, *extra)
