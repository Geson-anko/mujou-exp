import tomllib

import pytest

from exp.lib.fleet.config import FleetConf, Gpu
from tests.lib.fleet.fleetkit import REPO


def _toml(more: str = "") -> dict:
    return tomllib.loads(
        'nas = "/n"\nroot = "/r"\ncheckout = "/c"\ncontainer = "c"\nuser = "u"\n'
        'head = "a"\nhosts = ["a", "b"]\n\n'
        "[slots]\ncpu = 14\nsmt = true\nram_gb = 52\n"
        'gpus = [{index = 0, model = "4090", mem_gb = 22.0, jobs = 6}]\n' + more
    )


class TestConfig:
    def test_the_fleet_toml_of_the_repo(self):
        conf = FleetConf.load(REPO / "fleet.toml")
        assert conf.head in conf.hosts
        assert set(conf.slots) == set(conf.hosts)
        for host in ("std-00", "std-01", "std-02", "std-03"):
            s = conf.slots[host]
            assert (s.cpu, s.smt, s.ram_gb) == (14, False, 52.0)
            assert s.models == ("4090", "4060")
            assert [g.index for g in s.gpus] == [0, 1]
        mini = conf.slots["mini-00"]
        assert (mini.cpu, mini.smt, mini.ram_gb) == (6, False, 50.0)
        assert mini.gpus == (Gpu(index=0, model="5070ti", mem_gb=13.9, jobs=4),)
        assert conf.models == ("4090", "4060", "5070ti")

    def test_a_host_table_replaces_only_its_keys(self):
        conf = FleetConf.from_dict(_toml("[slots.b]\ncpu = 6\ngpus = []\n"))
        a, b = conf.slots["a"], conf.slots["b"]
        assert (a.cpu, a.smt, a.ram_gb, a.models) == (14, True, 52.0, ("4090",))
        assert (b.cpu, b.smt, b.ram_gb, b.models) == (6, True, 52.0, ())

    def test_a_host_table_of_no_host_is_refused(self):
        with pytest.raises(ValueError, match="'c'"):
            FleetConf.from_dict(_toml("[slots.c]\ncpu = 6\n"))

    def test_the_head_must_be_a_host(self):
        d = tomllib.loads((REPO / "fleet.toml").read_text())
        with pytest.raises(ValueError, match="head"):
            FleetConf.from_dict({**d, "head": "elsewhere"})
