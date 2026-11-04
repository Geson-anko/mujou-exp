import tomllib

import pytest

from exp.lib.fleet.config import FleetConf
from tests.lib.fleet.fleetkit import REPO


class TestConfig:
    def test_the_fleet_toml_of_the_repo(self):
        conf = FleetConf.load(REPO / "fleet.toml")
        assert conf.head in conf.hosts
        assert conf.slots.models == ("4090", "4060")
        assert [g.index for g in conf.slots.gpus] == [0, 1]

    def test_the_head_must_be_a_host(self):
        d = tomllib.loads((REPO / "fleet.toml").read_text())
        with pytest.raises(ValueError, match="head"):
            FleetConf.from_dict({**d, "head": "elsewhere"})
