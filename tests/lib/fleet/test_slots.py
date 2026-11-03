from collections.abc import Collection, Sequence
from pathlib import Path

import pytest

from exp.lib.fleet.config import Gpu, SlotsConf
from exp.lib.fleet.slots import Use, allocate, physical_cores, units

# 4 physical cores, SMT 2: logical i and i+4 share a core (as i and i+16 on the hosts)
CORES = [(0, 4), (1, 5), (2, 6), (3, 7)]
GPUS = (
    Gpu(index=0, model="4090", mem_gb=22.0, jobs=3),
    Gpu(index=1, model="4060", mem_gb=7.0, jobs=2),
)


def _conf(cpu: int = 3, smt: bool = False, ram: float = 100.0) -> SlotsConf:
    return SlotsConf(cpu=cpu, smt=smt, ram_gb=ram, gpus=GPUS)


def _fill(
    conf: SlotsConf,
    res: str,
    ram: float = 1.0,
    gpu_mem: float = 0.0,
    used: Sequence[Use] = (),
    cordoned: Collection[str] = (),
) -> list[Use]:
    used = list(used)
    while s := allocate(
        conf,
        CORES,
        used,
        res_class=res,
        ram_gb=ram,
        gpu_mem_gb=gpu_mem,
        cordoned=cordoned,
    ):
        used.append(Use(s, ram, gpu_mem))
    return used


class TestTopology:
    def test_reads_both_list_forms(self, tmp_path: Path):
        for cpu, siblings in ((0, "0,2"), (2, "0,2"), (1, "1-1"), (3, "3")):
            d = tmp_path / f"cpu{cpu}" / "topology"
            d.mkdir(parents=True)
            (d / "thread_siblings_list").write_text(siblings + "\n")
        assert physical_cores(tmp_path) == [(0, 2), (1,), (3,)]

    def test_runs_get_the_last_cores(self):
        assert units(_conf(cpu=3), CORES) == [(1, 5), (2, 6), (3, 7)]
        assert units(_conf(cpu=2, smt=True), CORES) == [(2,), (6,), (3,), (7,)]


class TestAllocate:
    @pytest.mark.parametrize("smt", [False, True])
    def test_no_physical_core_or_thread_twice(self, smt):
        conf = _conf(cpu=3, smt=smt)
        used = _fill(conf, "cpu")
        cpus = [c for u in used for c in u.slot.cpus]
        assert len(cpus) == len(set(cpus))
        assert len(used) == (6 if smt else 3)
        if not smt:  # a whole core per run: no two runs on the threads of one core
            assert {u.slot.cpus for u in used} == {(1, 5), (2, 6), (3, 7)}
        assert not {0, 4} & set(cpus)  # left to the worker

    def test_gpu_count_and_memory_stay_in_budget(self):
        conf = _conf(cpu=4)
        used = _fill(conf, "4090", gpu_mem=8.0)  # 22 GB: two of 8 GB, not three
        assert [u.slot.gpu for u in used] == [0, 0]
        used = _fill(conf, "4060", gpu_mem=1.0, used=used)  # 2 jobs at most
        assert [u.slot.gpu for u in used] == [0, 0, 1, 1]
        assert (
            allocate(conf, CORES, used, res_class="any_gpu", ram_gb=1, gpu_mem_gb=1)
            is None
        )

    def test_any_gpu_takes_the_most_memory_left(self):
        conf = _conf(cpu=4)
        used = _fill(conf, "any_gpu", gpu_mem=6.0)
        # 22 vs 7 -> 4090; 16 vs 7 -> 4090; 10 vs 7 -> 4090; 4090 is full (3 jobs) -> 4060
        assert [u.slot.gpu for u in used] == [0, 0, 0, 1]
        assert [u.slot.model for u in used] == ["4090", "4090", "4090", "4060"]

    def test_every_run_takes_a_cpu(self):
        conf = _conf(cpu=2)
        used = _fill(conf, "any_gpu", gpu_mem=1.0)
        assert len(used) == 2
        assert allocate(conf, CORES, used, res_class="cpu", ram_gb=1) is None

    def test_ram_budget(self):
        used = _fill(_conf(cpu=3, ram=5.0), "cpu", ram=2.0)
        assert len(used) == 2

    def test_cordons(self):
        conf = _conf()
        assert (
            allocate(conf, CORES, [], res_class="cpu", ram_gb=1, cordoned={"cpu"})
            is None
        )
        slot = allocate(
            conf, CORES, [], res_class="any_gpu", ram_gb=1, cordoned={"4090"}
        )
        assert slot is not None and slot.model == "4060"
        assert (
            allocate(conf, CORES, [], res_class="4090", ram_gb=1, cordoned={"4090"})
            is None
        )
        assert (
            allocate(conf, CORES, [], res_class="4060", ram_gb=1, cordoned={"all"})
            is None
        )

    def test_deterministic(self):
        conf = _conf(cpu=4)
        assert _fill(conf, "any_gpu", gpu_mem=3.0) == _fill(
            conf, "any_gpu", gpu_mem=3.0
        )
