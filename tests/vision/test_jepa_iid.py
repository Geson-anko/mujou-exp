"""``exp.vision.jepa_iid`` on the tiny synthetic Walking Tours of
``test_eval`` (videos A, B, C of 20 frames, H held out; stacks of 2 frames of
16 px, patch 4: the 4 x 4 grid of V18; patch 1: the 16 x 16 grid of V18b's
patch 4 at 64 px, and of V18c's patch 8 at 64 px with patch 2). The defaults
reproduce the values of V18's code. V18d's ``--st``: one time is V18c's run
bit for bit (plans/v18d.md, O1'), and the rows and settings of 4 times and
of ST-perm. The foundation's ``--arm`` (plans/foundation.md, 3; stacks of 8
frames): each arm's settings, rows and validation rows, the same seed's
same run, step 0 the built JEPA with its fixed R and W, and no optimizer
group of them."""

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
from exp.vision.fnd_jepa import build_fnd
from exp.vision.jepa import LATENT_DIM, R_SEED, W_SEED, build_jepa, jepa_mup_lr
from exp.vision.jepa_iid import (
    DELTA_SEED,
    PERM_SEED,
    RUNS,
    SPACING_SEED,
    ST_BANDS,
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


def strip(out: Path) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "sec"} for r in rows(out)]


class TestV18d:
    """V18d's clips of the grid 8 x 8 (V18C's flags) on videos of 20 frames:
    lags of 2..5 stacks fit 4 times (3 x 5 + 1 < 20)."""

    ST = ("--st", "4", "--st-lags", "2:5")

    def test_one_time_is_v18cs_run_bit_for_bit(self, root, tmp_path):
        base = ("--steps", "3", "--batch", "4", "--log-every", "1", *V18C)
        a = run(root, tmp_path, "a", *base)
        b = run(root, tmp_path, "b", *base, "--st", "1")
        got = strip(b)
        for r in got:  # in float64 of each clip's float32 mean
            assert r.pop("pred_now") == pytest.approx(r["pred"], rel=1e-6)
            assert r.pop("sigreg_times") == [r["sigreg"]]
        assert got == strip(a)
        for t in (0.0, 1.2):  # steps 0 and 3 of 4 clips
            wa = torch.load(ckpt.eval_path(a, t), weights_only=True)["modules"]
            wb = torch.load(ckpt.eval_path(b, t), weights_only=True)["modules"]
            assert all(torch.equal(wa[m][k], wb[m][k]) for m in wa for k in wa[m])

    def test_the_settings_and_rows_of_four_times(self, root, tmp_path):
        out = run(root, tmp_path, "r", "--steps", "2", "--batch", "8", *V18C, *self.ST)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["st"] == 4 and settings["st_perm"] is False
        assert settings["st_lags"] == [2, 5] and settings["st_bands"] == list(ST_BANDS)
        assert settings["time_masks"] == [
            [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]
        ]  # fmt: skip
        assert settings["seeds"]["delta"] == DELTA_SEED
        assert settings["seeds"]["perm"] is None
        (r,) = rows(out)
        for k in ("loss", "pred", "sigreg", "pred_now", "pred_time", "mix_var"):
            assert math.isfinite(r[k]), k
        # half the clips each side: L_ST is the mean of the two sides' errors
        assert r["pred"] == pytest.approx((r["pred_now"] + r["pred_time"]) / 2)
        assert len(r["sigreg_times"]) == 4
        assert r["sigreg"] == pytest.approx(sum(r["sigreg_times"]) / 4)
        assert len(r["pred_masks"]) == 6 and len(r["pred_bands"]) == 3
        assert r["pred_bands"][1:] == [None, None]  # lags 2..5: the first band
        assert r["overlap"] == 0.0  # 2 or more apart: no frame shared

    def test_st_perm_records_its_seed_and_shares(self, root, tmp_path):
        out = run(
            root, tmp_path, "r", "--steps", "2", "--batch", "8", *V18C, *self.ST,
            "--st-perm", "--log-every", "1",
        )  # fmt: skip
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["st_perm"] is True
        assert settings["seeds"]["perm"] == PERM_SEED
        assert all(0.0 <= r["overlap"] <= 1.0 for r in rows(out))

    def test_v18cs_settings_record_no_clips(self, root, tmp_path):
        out = run(root, tmp_path, "r", "--steps", "1", "--batch", "4", *V18C)
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["st"] == 0 and settings["st_lags"] is None
        assert settings["seeds"]["delta"] is None
        assert "pred_now" not in rows(out)[0]

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (("--st", "3"), "1 or even"),
            (("--st-perm",), "st_perm"),
            (("--st", "4", "--batch", "5"), "even"),
            (("--st", "4", "--st-lags", "5:2"), "st_lags"),
        ],
    )
    def test_bad_clips_are_refused(self, root, tmp_path, extra, match):
        with pytest.raises(ValueError, match=match):
            run(root, tmp_path, "r", "--steps", "1", *V18C, *extra)


def fnd_run(root: Path, tmp_path: Path, name: str, *extra: str) -> Path:
    """A run of the foundation's arm on the tiny tours: stacks of 8 frames
    (4 times of 2) of 16 px, patch 4 (4 x 4 positions), s up to 2."""
    conf = conf_of()
    tours = replace(conf.data.tours, frames=8)
    conf = replace(conf, data=replace(conf.data, tours=tours))
    cfg = tmp_path / "fnd.yaml"
    cfg.write_text(yaml.safe_dump(to_dict(conf)))
    out = tmp_path / name
    argv = ["--config", str(cfg), "--lam", "0.5", "--lr", "1e-3", "--warmup", "2"]
    argv += ["--directions", "64", "--batch", "4", "--s-max", "2", "--device", "cpu"]
    argv += ["--root", str(root), "--out", str(out), *extra]
    main(argv)
    return out


def vals(out: Path) -> list[dict]:
    return [json.loads(x) for x in (out / "val.jsonl").read_text().splitlines()]


class TestArms:
    @pytest.mark.parametrize("arm", ["T0", "S1", "Ss"])
    def test_an_arm_logs_finite_rows_and_its_settings(self, root, tmp_path, arm):
        out = fnd_run(root, tmp_path, arm, "--arm", arm, "--steps", "3")
        settings = json.loads((out / "jepa.json").read_text())
        assert settings["arm"] == arm
        assert settings["timed"] == (arm != "T0") and settings["scaled"] == (
            arm == "Ss"
        )
        assert settings["s_max"] == (2 if arm == "Ss" else 1)
        assert settings["tokens"] == 64 and settings["times"] == 4
        assert settings["token_dim"] == 16 and settings["latent_dim"] == 64 * 16
        assert settings["seeds"]["spacing"] == (SPACING_SEED if arm == "Ss" else None)
        assert settings["val"]["video"] in ("A", "B", "C")
        (r,) = rows(out)
        keys = ("loss", "pred", "sigreg", "grad_norm", "t_var", "z_var", "time_r2")
        for k in (*keys, "hidden", "pred_short", "pred_long"):
            assert math.isfinite(r[k]), k
        assert 0 < r["hidden"] < 1
        assert r["pred"] == pytest.approx((r["pred_short"] + r["pred_long"]) / 2)
        v = vals(out)
        assert [x["step"] for x in v] == [0, 3]
        for k in ("val_pred", "val_rel", "sigreg_heldin", "rank", "token_r2"):
            assert all(math.isfinite(x[k]) for x in v), k
        got = sorted((out / "ckpt").glob("*.pt"))
        assert got == [ckpt.eval_path(out, 0.0), ckpt.eval_path(out, 1.2)]

    def test_the_same_seed_gives_the_same_run(self, root, tmp_path):
        a = fnd_run(root, tmp_path, "a", "--arm", "Ss", "--steps", "2")
        b = fnd_run(root, tmp_path, "b", "--arm", "Ss", "--steps", "2")

        def bare(rs: list[dict]) -> list[dict]:
            return [{k: v for k, v in r.items() if k != "sec"} for r in rs]

        assert bare(rows(a)) == bare(rows(b)) and bare(vals(a)) == bare(vals(b))
        wa = torch.load(ckpt.eval_path(a, 0.8), weights_only=True)["modules"]
        wb = torch.load(ckpt.eval_path(b, 0.8), weights_only=True)["modules"]
        assert all(torch.equal(wa[m][k], wb[m][k]) for m in wa for k in wa[m])

    def test_step_0_is_the_built_jepa_and_r_w_are_kept(self, root, tmp_path):
        out = fnd_run(root, tmp_path, "r", "--arm", "S1", "--steps", "1")
        conf = from_dict(yaml.safe_load((out / "config.yaml").read_text()))
        mods = build_fnd(conf, seed=conf.seed)
        saved = torch.load(ckpt.eval_path(out, 0.0), weights_only=True)["modules"]
        assert set(saved) == {"encoder", "proj", "predictor", "mix"}
        for name, m in mods.modules().items():
            want = m.state_dict()
            assert all(torch.equal(saved[name][k], want[k]) for k in want), name
        d = 64 * 16
        assert torch.equal(saved["mix"]["weight"], gaussian_projection(d, d, W_SEED))

    def test_no_optimizer_group_holds_r_or_w(self):
        conf = conf_of()
        tours = replace(conf.data.tours, frames=8)
        mods = build_fnd(replace(conf, data=replace(conf.data, tours=tours)), seed=0)
        opt = _optimizer(mods, conf, 1e-3)
        held = {id(p) for g in opt.param_groups for p in g["params"]}
        assert mods.mix is not None
        assert id(mods.mix.weight) not in held and id(mods.proj.weight) not in held
        assert held == {id(p) for p in mods.parameters()}

    def test_an_unknown_arm_or_no_mask_ratio_is_refused(self, root, tmp_path):
        with pytest.raises(SystemExit):
            fnd_run(root, tmp_path, "r", "--arm", "S2", "--steps", "1")
        with pytest.raises(SystemExit):
            fnd_run(root, tmp_path, "r", "--steps", "1")
