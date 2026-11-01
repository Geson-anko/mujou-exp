from dataclasses import dataclass, field

import pytest

from exp.lib.config import from_dict


@dataclass(frozen=True)
class _Inner:
    tau: float = 1.0


@dataclass(frozen=True)
class _Outer:
    optim: _Inner = field(default_factory=_Inner)


class TestFromDict:
    def test_an_int_in_json_reads_as_float(self):
        config = from_dict(_Outer, {"optim": {"tau": 2}})
        assert isinstance(config.optim.tau, float)
        assert config.optim.tau == 2.0

    def test_an_unknown_key_is_an_error(self):
        with pytest.raises(TypeError, match="lr"):
            from_dict(_Outer, {"optim": {"lr": 0.1}})
