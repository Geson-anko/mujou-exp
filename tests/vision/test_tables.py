"""The registered rule of 'learned' (exp/vision/README.md, V2-V5)."""

from exp.vision.tables import learned, stream_learned


def _row(stack_mse: float, blur: float, sd: float, dyn: float) -> dict[str, object]:
    return {
        "stack_mse": stack_mse,
        "recon_blur": blur,
        "recon_blur_sd": sd,
        "recon_blur_k": [0.0] * 4,
        "dyn_1/copy_1": dyn,
    }


class TestLearned:
    def test_all_four_down_is_learned(self) -> None:
        c = learned(_row(6.7, 0.28, 0.001, 0.88), _row(0.9, 0.20, 0.003, 0.80))
        assert c["all"]

    def test_stack_mse_above_one_is_not_learned(self) -> None:
        """V1's tau 1e5: three conditions hold but the system decoder is
        worse than the mean stack."""
        c = learned(_row(6.7, 0.282, 0.0006, 0.880), _row(1.518, 0.2175, 0.0027, 0.796))
        assert c["stack_mse_down"] and c["recon_blur_down"] and c["dyn_down"]
        assert not c["stack_mse_lt1"] and not c["all"]

    def test_the_blur_threshold_is_twice_the_combined_standard_error(self) -> None:
        # s0 = s1 = 0.01, k = 4: threshold 2 sqrt(2e-4 / 4) = 0.01414...
        r0 = _row(2.0, 0.300, 0.01, 0.9)
        assert not learned(r0, _row(0.5, 0.286, 0.01, 0.8))["recon_blur_down"]
        assert learned(r0, _row(0.5, 0.285, 0.01, 0.8))["recon_blur_down"]


def _summary(span: float, after: float) -> dict[str, object]:
    return {"span": span, "after": after}


def _probe(blur: float, sd: float, dyn: float) -> dict[str, object]:
    return {
        "recon_blur": blur,
        "recon_blur_sd": sd,
        "recon_blur_k": [0.0] * 4,
        "dyn_1/copy_1": dyn,
    }


class TestStreamLearned:
    def test_all_five_down_is_learned(self) -> None:
        c = stream_learned(
            _summary(6.0, 6.5),
            _summary(0.7, 0.9),
            _probe(0.28, 0.001, 0.88),
            _probe(0.22, 0.002, 0.80),
        )
        assert c["all"]

    def test_a_span_above_one_is_not_learned(self) -> None:
        """SGD's decoder: the span is worse than the mean stack at the end."""
        c = stream_learned(
            _summary(6.0, 6.5),
            _summary(1.4, 1.5),
            _probe(0.28, 0.001, 0.88),
            _probe(0.22, 0.002, 0.80),
        )
        assert c["span_down"] and c["after_down"] and not c["span_lt1"]
        assert not c["all"]

    def test_the_after_span_must_fall_too(self) -> None:
        c = stream_learned(
            _summary(6.0, 0.8),
            _summary(0.7, 0.9),
            _probe(0.28, 0.001, 0.88),
            _probe(0.22, 0.002, 0.80),
        )
        assert not c["after_down"] and not c["all"]
