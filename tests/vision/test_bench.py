import json

import torch

from exp.vision import bench
from exp.vision.compose import compose


class TestRandomStacks:
    def test_the_stacks_of_a_call_are_fixed_by_the_seed_and_resume(self):
        conf = compose(["+experiment=v0/tiny", "data.streams=2"])
        a, b = bench.RandomStacks(conf), bench.RandomStacks(conf)
        x, ready, reset = a.next()
        assert x.shape == (2, 2, 3, 32, 32) and x.dtype == torch.uint8
        assert ready.all() and reset.all()
        b.load_state_dict(a.state_dict())
        y, _, reset = a.next()
        assert torch.equal(b.next()[0], y) and not reset.any()
        assert not torch.equal(x, y)


def test_main_times_every_arch_and_size(tmp_path):
    """A row per arch and size, at B = 1, written to --out."""
    out = tmp_path / "bench.json"
    rows = bench.main(
        ["--frames", "1", "--warmup", "1", "--experiment", "v0/tiny", "--out", str(out)]
    )
    assert [(r["arch"], r["size"]) for r in rows] == [
        (a, s) for a in ("vit", "resnet") for s in ("S", "M", "L")
    ]
    assert all(r["units_s"] > 0 and r["gpu_mem_mb"] == 0.0 for r in rows)
    assert json.loads(out.read_text()) == rows
