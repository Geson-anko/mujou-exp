import pytest

from exp.video.jepa.nets import VisionBP, VisionPC, build_vision_net, vision_modules
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
