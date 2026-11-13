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
            ({"token_sigreg": "half"}, "token_sigreg"),
            ({"token_weight": 0.0}, "both be 0"),
            ({"z_weight": -1.0}, "z_weight"),
            ({"projector": [16]}, "z_weight > 0"),
            ({"zpred_hidden": 8}, "z_weight > 0"),
            ({"z_weight": 1.0, "projector": [0]}, "widths"),
            ({"pred_dim": 24}, "multiple of 16"),
            ({"pred_dim": 32, "pred_heads": 3}, "heads"),
            ({"ema": 1.5}, "ema"),
            ({"ema": -0.1}, "ema"),
            ({"ema": 0.9, "token_weight": 0.0, "z_weight": 1.0}, "token_weight 0"),
            ({"token_dim": 0}, "token_dim"),
            ({"token_dim": 16, "z_dim": 16}, "z_dim None"),
            ({"token_buffer": -1}, "token_buffer"),
            ({"token_insert": 0, "token_buffer": 8}, "token_insert"),
            ({"token_buffer": 8, "sigreg_on": "tokens"}, "'raw'"),
            ({"token_buffer": 8, "token_sigreg": "raw"}, "'raw'"),
        ],
    )
    def test_refused(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _packed(**kw)

    def test_the_target_encoder_is_off_by_default_and_m_in_0_1(self):
        assert _packed().ema == 0.0 and VisionConf().ema == 0.0  # 0: none
        assert [_packed(ema=m).ema for m in (0.0, 0.996, 1.0)] == [0.0, 0.996, 1.0]

    def test_the_tokens_as_the_representation(self):
        v = _packed(token_dim=16, sigreg_on="tokens", token_sigreg="raw")
        assert v.token_width == v.z_width == 16 and not v.projected
        assert _packed().token_width == 32
        _packed(token_buffer=64, sigreg_on="both", token_sigreg="raw")

    @pytest.mark.parametrize("kw", [{"token_dim": 8}, {"token_buffer": 8}])
    def test_the_tokens_are_of_the_packed_layout_only(self, kw):
        with pytest.raises(ValueError, match="packed layout only"):
            VisionConf(**kw)

    def test_the_pixels_layout_is_not_checked_so(self):
        VisionConf(frames=3, tubelet=0, views=2)  # M15's defaults stay valid

    def test_the_space_time_completion(self):
        v = _packed(z_weight=1.0, projector=[64, 16], zpred_hidden=32)
        assert v.embeds_views and v.u_width == 16
        views = _packed(sigreg_on="views", z_dim=8)
        assert views.embeds_views and views.u_width == 8
        assert not _packed().embeds_views
        with pytest.raises(ValueError, match="packed"):
            VisionConf(z_weight=1.0)
        with pytest.raises(ValueError, match="packed"):
            VisionConf(pred_dim=64)
        with pytest.raises(ValueError, match="packed"):
            VisionConf(target_norm=True)
        with pytest.raises(ValueError, match="packed"):
            VisionConf(ema=0.9)

    def test_the_token_sigreg_is_of_the_packed_layout(self):
        assert _packed(sigreg_on="both", token_sigreg="clip").token_sigreg == "clip"
        with pytest.raises(ValueError, match="packed"):
            VisionConf(token_sigreg="clip")

    def test_a_future_must_leave_a_slot(self):
        future = [MaskConf(blocks=8, scale=0.15), MaskConf(blocks=0, scale=1, future=1)]
        assert _packed(context_masks=future).context_masks[1].future == 1
        with pytest.raises(ValueError, match="future"):
            _packed(context_masks=[MASKS[0], MaskConf(blocks=1, scale=0.5, future=2)])

    @pytest.mark.parametrize(
        "kw",
        [
            {"blocks": 0, "scale": 0.1},
            {"blocks": 1, "scale": 1.5},
            {"blocks": 1, "scale": 0.5, "future": -1},
        ],
    )
    def test_a_mask_is_checked(self, kw):
        with pytest.raises(ValueError, match="context mask"):
            MaskConf(**kw)
