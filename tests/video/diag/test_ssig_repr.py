"""The check of the mechanism of the stream SIGReg of M17 (diagnosis): the lags
whose stacks share frames, the arm of a curve's label, and the mean pair
correlation of a curve read beside no ``corr.json``."""

import json
from pathlib import Path

from exp.video.diag.ssig_repr import arm_of, lag_table, rbar_table, split_lags
from exp.video.diag.ssig_window import load_curves, windows


def test_stacks_of_16_frames_share_frames_up_to_15_steps_apart():
    lags = [0, 1, 15, 16, 32]
    shared, apart = split_lags(lags, [1.0, 0.5, 0.15, 0.16, 0.32])
    assert shared == [0.5, 0.15] and apart == [0.16, 0.32]


def test_the_arm_of_a_label_is_its_first_field():
    assert arm_of("t512-k6-l0.0442-s100/tokens/white") == "t512"
    assert arm_of("t64/tokens/white") == "t64"


def _repr(path: Path, label: str, rho: list[float]) -> None:
    lags = [3, 6, 45, 48, 96]  # video frames, stride 3: steps 1 2 15 16 32
    curve = {"rho": rho, "se": [0.0] * len(rho)}
    code = {"stride": 3, "lags": lags, "t95": 2.0, "curves": {"white": curve}}
    path.write_text(json.dumps({"codes": {f"{label}/tokens": code}}))


def test_an_uncorrelated_curve_has_no_pair_correlation_and_an_open_window(
    tmp_path: Path,
):
    _repr(tmp_path / "repr-a.json", "t512-k1-l0.0442-s100", [0.0] * 5)
    name = "t512-k1-l0.0442-s100/tokens/white"
    rows = windows(tmp_path, [name], [0.122], [1], 256, 2000)
    shown = rbar_table(rows)
    # Only the curve's own arm is shown, with rbar 0 below rbar_max.
    assert shown.count(f"`{name}`") == 1
    assert "| 0.0000 (0.0000〜0.0000) |" in shown and shown.endswith("○ |")


def test_lag_table_puts_the_untrained_curve_first_and_means_each_band(
    tmp_path: Path,
):
    _repr(tmp_path / "repr-t.json", "t64", [0.4, 0.2, 0.1, 0.05, 0.01])
    _repr(tmp_path / "repr-k.json", "t64-k24-l0.233-s100", [0.2] + [0.0] * 4)
    names = ["t64/tokens/white", "t64-k24-l0.233-s100/tokens/white"]
    t = lag_table(load_curves(tmp_path, names), "t64", [1, 16])
    lines = t.splitlines()
    assert lines[0] == "| lag | `t64` | `t64-k24-l0.233-s100` |"
    assert lines[2] == "| 0.1 秒 | 0.4000 | 0.2000 |"
    # Shared lags 1, 2, 15: (0.4 + 0.2 + 0.1) / 3; apart 16, 32: 0.03.
    assert lines[4] == "| 1〜15 の平均 | 0.2333 | 0.0667 |"
    assert lines[5] == "| 16 以上の平均 | 0.0300 | 0.0000 |"
