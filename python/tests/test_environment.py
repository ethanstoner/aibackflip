"""Preconditions the training stack depends on.

These are not placeholder tests. Every one of them has a failure mode that would
otherwise show up hours into a training run: a CPU-only torch silently training
100x slower, a float64 default dtype doubling every packet, or an autograd graph
that does not actually produce gradients.
"""

import numpy as np
import pytest
import torch


def test_torch_sees_the_gpu():
    assert torch.cuda.is_available(), "CUDA unavailable; PPO updates would fall back to CPU"
    assert torch.cuda.device_count() >= 1
    torch.zeros(8, device="cuda").sum().item()  # a real allocation, not just a query


def test_default_dtype_is_float32():
    # The UDP protocol packs float32. A float64 default would make every
    # observation tensor silently widen and mismatch the wire format.
    assert torch.get_default_dtype() is torch.float32
    assert np.zeros(3, dtype=np.float32).dtype.itemsize == 4


def test_gradients_flow_through_a_tanh_mlp():
    # Exactly the policy shape from the plan: Linear -> Tanh -> Linear -> Tanh -> Linear.
    net = torch.nn.Sequential(
        torch.nn.Linear(12, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 5),
    )
    out = net(torch.randn(4, 12))
    assert out.shape == (4, 5)
    out.pow(2).mean().backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_cuda_and_cpu_agree_on_a_forward_pass():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    x = torch.randn(8, 16)
    cpu_out = net(x)
    gpu_out = net.cuda()(x.cuda()).cpu()
    assert torch.allclose(cpu_out, gpu_out, atol=1e-5)


def test_seeding_is_reproducible():
    torch.manual_seed(1234)
    a = torch.randn(64)
    torch.manual_seed(1234)
    b = torch.randn(64)
    assert torch.equal(a, b)


def test_tensorboard_writer_is_importable():
    from torch.utils.tensorboard import SummaryWriter

    assert SummaryWriter is not None


@pytest.mark.parametrize("shape", [(1,), (32, 46), (25, 186)])
def test_numpy_float32_roundtrips_through_bytes(shape):
    # The exact operation the UDP client performs on every received packet.
    rng = np.random.default_rng(0)
    original = rng.standard_normal(shape).astype(np.float32)
    restored = np.frombuffer(original.tobytes(), dtype="<f4").reshape(shape)
    assert np.array_equal(original, restored)
