"""The readout check of M18 stage 1 (diagnosis): the movement of a parameter
is its known change, taking out the whole readout leaves the decoder's
constant, the removed term is linear in the weights taken out, and the errors
of a perfect and a constant reconstruction are 0 and 1."""

import torch
from torch import nn

from exp.video.diag.m18_readout import movement, readout_errors, without_init_readout
from exp.video.vae.loss import target
from exp.video.vae.modules import TokenDecoder

MU, SIGMA = 0.5, 0.25


def _decoder() -> TokenDecoder:
    torch.manual_seed(0)
    return TokenDecoder(
        token_dim=2,
        dim=16,
        heads=2,
        depth=1,
        grid3=(1, 2, 2),
        patch=2,
        tubelet=1,
        channels=3,
        mean=MU,
        std=SIGMA,
    )


def _tokens() -> torch.Tensor:
    return torch.randn(3, 4, 2, generator=torch.Generator().manual_seed(1))


def test_movement_is_the_known_change_with_the_fan_in():
    before = nn.Linear(4, 3)
    after = nn.Linear(4, 3)
    with torch.no_grad():
        before.weight.fill_(0.5)
        before.bias.zero_()
        after.weight.copy_(before.weight)
        after.weight[0, 0] += 0.3  # one element moves by 0.3
        after.bias.copy_(torch.tensor([1.0, -1.0, 1.0]))
    rows = {r["name"]: r for r in movement(before, after)}
    w, b = rows["weight"], rows["bias"]
    assert (w["n"], b["n"]) == (4, 1)  # the matrix's inputs; a vector's 1
    assert w["rms0"] == 0.5 and b["rms0"] == 0.0
    assert abs(w["max_delta"] - 0.3) < 1e-6
    assert abs(w["rms_delta"] - (0.3**2 / 12) ** 0.5) < 1e-6
    assert b["rms_delta"] == 1.0 and b["max_delta"] == 1.0


def test_without_the_whole_readout_the_decoder_is_its_constant():
    dec = _decoder()
    last = dec.stage.out
    assert last is not None
    with torch.no_grad():
        last.bias.fill_(0.4)
    ablated = without_init_readout(dec, last.weight.detach().clone())
    out = ablated(_tokens())
    torch.testing.assert_close(out, torch.full_like(out, MU + SIGMA * 0.4))
    assert not torch.equal(dec(_tokens()), out)  # the original is unchanged


def test_the_removed_term_is_linear_in_the_weights_taken_out():
    dec = _decoder()
    last = dec.stage.out
    assert last is not None
    w0 = torch.randn_like(last.weight)
    x = _tokens()
    d1 = dec(x) - without_init_readout(dec, w0)(x)
    d2 = dec(x) - without_init_readout(dec, 2 * w0)(x)
    torch.testing.assert_close(d2, 2 * d1)


def test_errors_of_a_perfect_and_of_the_mean_reconstruction():
    g = torch.Generator().manual_seed(2)
    stacks = torch.randint(0, 256, (5, 2, 3, 4, 4), generator=g, dtype=torch.uint8)
    mean = target(stacks).mean(0)
    r = readout_errors(
        target, lambda x: mean.expand(len(x), -1, -1, -1, -1), stacks, mean, batch=2
    )
    assert abs(r["stack_mse"]) < 1e-12  # g(f(x)) = y
    assert abs(r["ablated"] - 1) < 1e-6  # the mean stack is 1 by definition
    assert abs(r["e0"] - 1) < 1e-6  # y - m over sum ||y - m||^2
