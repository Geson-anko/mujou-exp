import pytest

from exp.video.schema.vision import MaskConf, VisionConf

MASKS = [MaskConf(blocks=8, scale=0.15), MaskConf(blocks=2, scale=0.7)]


def _packed(**kw) -> VisionConf:
    base = {
        "layout": "packed",
        "dim": 32,
        "heads": 2,
        "frames": 4,
        "tubelet": 2,
        "views": 3,
        "context_masks": MASKS,
    }
    return VisionConf(**(base | kw))


class TestPacked:
    def test_a_valid_config(self):
        v = _packed(z_dim=64, sigreg_on="both")
        assert v.projected and v.z_width == 64
        assert not _packed(z_dim=32).projected  # z_dim = dim: no projection
        assert _packed().z_width == 32

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"tubelet": 3}, "divide"),
            ({"tubelet": 0}, "divide"),
            ({"views": 2}, "context_masks"),
            ({"z_pool": "cls"}, "mean"),
            ({"dim": 40, "heads": 2}, "multiple of 16"),
            ({"z_dim": 1025}, "z_dim"),
            ({"z_dim": 0}, "z_dim"),
            ({"z_dim": 64, "sigreg_on": "tokens"}, "projection"),
            ({"stride": 0}, "stride"),
            ({"pred_loss": "huber"}, "pred_loss"),
        ],
    )
    def test_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _packed(**kw)

    def test_the_pixels_layout_is_not_checked_so(self):
        VisionConf(frames=3, tubelet=0, views=2)  # M15's defaults stay valid

    @pytest.mark.parametrize(
        "kw", [{"blocks": 0, "scale": 0.1}, {"blocks": 1, "scale": 1.5}]
    )
    def test_a_mask_is_checked(self, kw):
        with pytest.raises(ValueError, match="context mask"):
            MaskConf(**kw)
