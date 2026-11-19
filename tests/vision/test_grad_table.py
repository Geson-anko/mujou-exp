"""``python -m exp.vision.grad_table`` on a synthetic ``grad.jsonl``."""

import json

import pytest

from exp.vision import grad_table


def _row(t, node, **kw):
    base = {
        "t": t,
        "node": node,
        "g": 1.0,
        "cos_prev": None,
        "age_prev": None,
        "cos_1": None,
        "age_1": None,
        "cos_10": None,
        "age_10": None,
        "cos_100": None,
        "age_100": None,
        "rho_0.1": None,
        "rho_1": None,
        "rho_10": None,
    }
    return {**base, **kw}


ROWS = [
    _row(
        100.0,
        "dec1",
        cos_prev=0.9,
        age_prev=0.1,
        cos_1=0.5,
        age_1=1.0,
        cos_10=0.3,
        age_10=5.0,  # the bin of 10
        cos_100=0.1,
        age_100=50.0,  # the bin of 100
        **{"rho_0.1": 0.2, "rho_1": 0.5, "rho_10": 0.9},
    ),
    _row(
        300.0,  # still the first window
        "dec1",
        cos_prev=0.7,
        age_prev=0.1,
        cos_1=0.3,
        age_1=1.0,
        cos_10=0.5,
        age_10=2.0,  # the bin of 1
        **{"rho_0.1": 0.4},
    ),
    _row(400.0, "dec1", cos_prev=0.2, age_prev=0.1, **{"rho_0.1": 0.6}),
    _row(100.0, "enc1", cos_prev=-0.5, age_prev=0.1),
    _row(1900.0, "enc1", cos_prev=1.0, age_prev=0.1),  # past 1800: dropped
]


class TestTable:
    def test_the_means_by_node_window_and_age(self):
        got = {(r["node"], r["window"]): r for r in grad_table.table(ROWS)}
        early = got["dec1", "0-300"]
        assert early["cos_0.1"] == pytest.approx(0.8)
        assert early["cos_1"] == pytest.approx((0.5 + 0.3 + 0.5) / 3)
        assert early["cos_10"] == pytest.approx(0.3)
        assert early["cos_100"] == pytest.approx(0.1)
        assert early["rho_0.1"] == pytest.approx(0.3)
        assert early["rho_1"] == pytest.approx(0.5)
        assert early["rho_10"] == pytest.approx(0.9)
        assert early["rows"] == 2
        late = got["dec1", "300-1800"]
        assert late["cos_0.1"] == pytest.approx(0.2)
        assert late["cos_1"] is None and late["rho_0.1"] == pytest.approx(0.6)
        assert got["enc1", "0-300"]["cos_0.1"] == pytest.approx(-0.5)
        assert ("enc1", "300-1800") not in got

    def test_main_prints_a_row_per_run_node_and_window(self, tmp_path, capsys):
        run = tmp_path / "run-a"
        run.mkdir()
        (run / "grad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS))
        grad_table.main([str(run)])
        lines = [x for x in capsys.readouterr().out.splitlines() if "run-a" in x]
        assert len(lines) == 3
        assert any("dec1" in x and "0-300" in x and "0.800" in x for x in lines)
