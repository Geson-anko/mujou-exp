import re
from dataclasses import replace

import pytest

from exp.lib.optim import fan_ins
from exp.video.compose import compose
from exp.video.dyn.nets import make_optimizer
from exp.video.jepa.nets import VisionBP, VisionPC, build_vision_net, vision_modules
from exp.video.jepa.vit import build_vision
from exp.video.schema.learn import OptimConf
from tests.video.jepa.test_nodes import _conf


def test_the_learner_picks_the_network():
    assert isinstance(build_vision_net(_conf("pc")), VisionPC)
    assert isinstance(build_vision_net(_conf("bp")), VisionBP)


def test_none_learns_no_vision():
    with pytest.raises(ValueError, match="learns no vision"):
        build_vision_net(_conf("none"))


@pytest.mark.parametrize("nodes", [1, 2])
def test_one_encoder_node_per_stage_holding_the_live_modules(nodes):
    net = build_vision_net(_conf(depth=2, nodes=nodes))
    assert isinstance(net, VisionPC)
    mods = vision_modules(net)
    assert net.names == tuple(f"enc{i + 1}" for i in range(nodes))
    for name, stage in zip(net.names, mods.encoder.stages, strict=True):
        assert getattr(net, name).module is stage
    assert net.head.module is mods.predictor


def _scaled(learner: str):
    conf = _conf(learner)
    rms = OptimConf(kind="rmsprop", lr=1e-3, warmup=0.0, scale="fan_in")
    return replace(conf, optim_vision=rms)


def test_the_bp_vision_learns_at_the_rates_of_its_modules_fan_ins():
    """``JEPABP`` binds every parameter it learns, and binding one without a
    fan-in raises: each has its fan-in from the modules."""
    assert isinstance(build_vision_net(_scaled("bp")), VisionBP)
    with pytest.raises(ValueError, match="needs the modules"):
        build_vision_net(_scaled("pc"))


# The fan-in of each parameter of m17/ssig_t128 (results/m17.md, 「層ごとの
# 学習率」), by name: the first pattern that matches; every other is a vector.
M17_FAN_INS = (
    (r"encoder\.stages\.0\.embed\.proj\.weight", 6144),  # 2 x 3 x 32 x 32 pixels
    (r".*blocks\.\d+\.(qkv|proj|fc1)\.weight", 256),  # qkv fused: still 256
    (r".*blocks\.\d+\.fc2\.weight", 1024),
    (r"encoder\.stages\.3\.out\.weight", 256),  # tokens, 256 to d = 8
    (r"predictor\.embed\.weight", 8),  # context tokens, 8 to 256
    (r"predictor\.out\.weight", 256),  # 256 to 8
)


def test_the_fan_ins_of_the_m17_ssig_t128_modules():
    conf = compose(
        [
            "+experiment=m17/ssig_t128",
            "learner=bp",
            "optim_vision.warmup=0",
            "optim_vision.scale=fan_in",
            "device=cpu",
        ]
    )
    mods = build_vision(conf, seed=100)
    named = mods.named()
    assert set(named) == {"encoder", "predictor"}
    n = fan_ins(*named.values())
    got, want = {}, {}
    for key, module in named.items():
        for name, p in module.named_parameters():
            full = f"{key}.{name}"
            got[full] = n[p]
            rule = (f for pattern, f in M17_FAN_INS if re.fullmatch(pattern, full))
            want[full] = next(rule, None)
            if want[full] is None:  # a vector: a bias, a gain or a token
                assert p.dim() == 1, full
                want[full] = 1
    assert got == want
    assert sum(f == 6144 for f in got.values()) == 1
    assert sum(f == 1024 for f in got.values()) == 10  # 8 encoder + 2 predictor blocks
    assert set(got.values()) == {1, 8, 256, 1024, 6144}
    assert conf.optim_vision is not None
    make_optimizer(conf.optim_vision, conf.time.frame, list(named.values())).bind(
        [p for m in named.values() for p in m.parameters()]
    )
