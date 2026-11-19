"""``python -m exp.vision.coord_check``: the slope of a power law, and a tiny
check of synthetic stacks whose rows are those of the definition."""

import json
import math

import pytest

from exp.vision.coord_check import NODES, loglog_slope, main


class TestSlope:
    @pytest.mark.parametrize("a", [0.0, 0.5, -1.0, 2.0])
    def test_a_power_law_has_its_exponent(self, a):
        widths = [128, 256, 512, 1024]
        values = [3.0 * w**a for w in widths]
        assert loglog_slope(widths, values) == pytest.approx(a, abs=1e-12)

    def test_a_value_not_positive_has_none(self):
        assert loglog_slope([1, 2], [0.0, 1.0]) is None
        assert loglog_slope([1, 2], [math.nan, 1.0]) is None


class TestTiny:
    def test_rows_and_slopes(self, tmp_path, capsys):
        """v0/tiny of widths 128 and 256, SP and muP, frames 0 and 2 of
        synthetic stacks: a row per node, width, parametrization and frame,
        no change at frame 0 and some at 2; at width 256 muP's rows are SP's
        exactly; a slope per node, parametrization and frame after 0."""
        out = tmp_path / "coord.json"
        argv = ["--random", "--experiment", "v0/tiny", "--widths", "128,256"]
        argv += ["--frames", "0,2", "--out", str(out), "model.heads=null"]
        main(argv)
        got = json.loads(out.read_text())
        rows, slopes = got["rows"], got["slopes"]
        assert len(rows) == 2 * 2 * 2 * len(NODES)
        for r in rows:
            assert r["rms"] > 0
            assert (r["delta"] == 0) == (r["frame"] == 0)

        def at(param: str) -> list[tuple]:
            return [
                (r["node"], r["frame"], r["rms"], r["delta"])
                for r in rows
                if r["param"] == param and r["width"] == 256
            ]

        assert at("sp") == at("mup")
        assert {(s["param"], s["node"], s["frame"]) for s in slopes} == {
            (p, n, 2) for p in ("sp", "mup") for n in NODES
        }
        assert "slope" in capsys.readouterr().out
