import torch

from fpopp.loss import modified_huber, ste_clamp, surrogate_loss
from fpopp.buffer import FPOBuffer, chunk_gae


def test_modified_huber_and_ste_gradient():
    error = torch.tensor([-1.0, -0.2, 0.0, 0.2, 1.0], requires_grad=True)
    assert torch.allclose(modified_huber(error), torch.tensor([0.75, 0.04, 0, 0.04, 0.75]))
    modified_huber(error).sum().backward()
    assert torch.allclose(error.grad, torch.tensor([-1.0, -0.4, 0, 0.4, 1.0]))
    x = torch.tensor([-9.0, 9.0], requires_grad=True)
    bounded = ste_clamp(x, -5, 5)
    bounded.sum().backward()
    assert bounded.tolist() == [-5, 5] and x.grad.tolist() == [1, 1]


def test_matches_official_order_keeps_mc_and_masks_execution():
    old = torch.tensor([[[8.0, 1.0, 100.0], [1.0, 3.0, 100.0]]], requires_grad=True)
    current = torch.tensor([[[3.9, 1.2, 700.0], [1.2, 2.4, 700.0]]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    advantage = torch.tensor([2.0])
    loss, _ = surrogate_loss(old, current, mask, advantage)
    # Independently express the official per-position cap -> chunk sum -> per-MC PPO.
    ratios = torch.exp(torch.tensor([4+1-3.9-1.2, 1+3-1.2-2.4]))
    expected = torch.maximum(-2*ratios, -2*ratios.clamp(0.99, 1.01)).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    assert old.grad is None
    assert current.grad[..., 2].abs().sum() == 0


def test_identical_losses_unit_ratio_except_official_old_only_cap():
    current = torch.ones(2, 3, 4, requires_grad=True)
    mask = torch.ones(2, 4)
    _, stats = surrogate_loss(current.detach(), current, mask, torch.ones(2))
    assert stats["ratio_mean"] == 1
    _, stats = surrogate_loss(current.detach()*6, current*6, mask, torch.ones(2))
    assert torch.allclose(stats["ratio_mean"], torch.tensor(-5.0).exp())
    loss, _ = surrogate_loss(current.detach(), current, mask*0, torch.ones(2))
    loss.backward()
    assert loss == 0 and torch.isfinite(current.grad).all()


def test_gae_cuts_success_timeout_and_reset_value():
    rewards = torch.tensor([[1., 2.], [3., 4.]])
    values = torch.tensor([[.5, .5], [100., 200.]])
    terminated = torch.tensor([[True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False]])
    advantages, returns = chunk_gae(rewards, values, terminated, truncated, torch.tensor([2., 3.]), .9, .8)
    assert torch.allclose(advantages[0], torch.tensor([.5, 1.5]))
    assert torch.allclose(returns[-1], torch.tensor([4.8, 6.7]), atol=1e-5)


def test_buffer_owns_shared_memory_and_fixed_mc_samples():
    source = torch.ones(1, 2, 3)
    buffer = FPOBuffer()
    buffer.add(obs={"x": source}, time=source)
    source.fill_(12)
    assert buffer.records[0]["obs"]["x"].eq(1).all()
    assert buffer.records[0]["time"].eq(1).all()
