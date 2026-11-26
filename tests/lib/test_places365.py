r"""``exp.lib.places365``: the vendored hierarchy's shape and order, the
groups' probabilities summing to 1 with a class of two groups split evenly,
the official transform of a flat frame by hand, the features before the last
layer giving the logits through it, and the weights refused on another
sha256."""

import pytest
import torch
from torchvision import models

from exp.lib.places365 import (
    class_names,
    group_names,
    group_probs,
    load_places365,
    logits,
    membership,
    penultimate,
    preprocess,
)


class TestHierarchy:
    def test_365_classes_in_16_groups_each_in_one_or_more(self):
        m = membership()  # its rows' names checked against class_names()
        assert m.shape == (365, 16)
        assert bool((m.sum(1) >= 1).all())
        assert len(class_names()) == 365 and class_names()[0] == "/a/airfield"
        assert len(group_names()) == 16

    def test_the_groups_probabilities_sum_to_1_and_split_a_shared_class(self):
        m = membership()
        g = torch.Generator().manual_seed(0)
        q = group_probs(torch.randn(8, 365, generator=g), m)
        torch.testing.assert_close(q.sum(1), torch.ones(8, dtype=torch.float64))
        shared = int((m.sum(1) == 2).nonzero()[0])
        one = torch.full((1, 365), -1e4)
        one[0, shared] = 0.0  # all the probability on the shared class
        q1 = group_probs(one, m)[0]
        groups = m[shared].nonzero().flatten()
        torch.testing.assert_close(
            q1[groups], torch.full((2,), 0.5, dtype=torch.float64)
        )


class TestTransform:
    def test_a_flat_frame_is_its_normalized_colour(self):
        x = torch.tensor([10, 128, 250], dtype=torch.uint8)[None, :, None, None]
        got = preprocess(x.expand(2, 3, 320, 320))
        assert got.shape == (2, 3, 224, 224)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        want = (torch.tensor([10, 128, 250]) / 255 - mean) / std
        torch.testing.assert_close(got[:, :, 100, 100], want.expand(2, 3))


class TestNetwork:
    def test_the_last_layer_of_the_features_gives_the_logits(self):
        torch.manual_seed(0)  # torchvision's initialization takes no generator
        net = models.resnet50(num_classes=365).eval()
        g = torch.Generator().manual_seed(1)
        x = torch.randn(2, 3, 64, 64, generator=g)
        f = penultimate(net, x)
        assert f.shape == (2, 2048)
        torch.testing.assert_close(net.fc(f), logits(net, x), rtol=1e-4, atol=1e-5)

    def test_weights_of_another_sha256_are_refused(self, tmp_path):
        path = tmp_path / "w.pth.tar"
        path.write_bytes(b"not the weights")
        with pytest.raises(ValueError, match="sha256"):
            load_places365(path, "cpu")
