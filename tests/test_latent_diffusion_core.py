"""Synthetic algebra and interface tests; no CIF or pretrained files required."""

import subprocess
import sys

import pytest
import torch

from latent_diffusion import ConditionalDenoiser, GaussianDiffusion


@pytest.mark.parametrize("mode", ["additive", "film"])
def test_denoiser_shapes_gradients_and_condition_presence(mode):
    torch.manual_seed(2)
    model = ConditionalDenoiser(8, width=32, blocks=2, expansion=2, context_mode=mode)
    u = torch.randn(4, 8, requires_grad=True)
    t = torch.tensor([0, 1, 500, 999])
    conditions = torch.tensor([[0., 0.], [1., -1.], [2., 3.], [-4., 5.]])
    absent = torch.zeros(4, dtype=torch.bool)
    prediction = model(u, t, conditions)
    assert prediction.shape == u.shape and torch.isfinite(prediction).all()
    assert torch.equal(model(u, t, conditions, absent), model(u, t, torch.zeros_like(conditions), absent))
    assert torch.equal(model(u, t, conditions, absent), model(u, t, None))
    assert not torch.equal(model(u, t, torch.zeros_like(conditions)), model(u, t, None))
    prediction.square().mean().backward()
    assert torch.isfinite(u.grad).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())


def test_default_architecture_is_not_silent_tiny_fixture():
    model = ConditionalDenoiser(128)
    assert (model.latent_dim, model.width, model.blocks, model.expansion, model.context_mode) \
        == (128, 256, 4, 2, "film")
    assert model(torch.zeros(2, 128), torch.tensor([0, 999]), torch.zeros(2, 2)).shape == (2, 128)


def test_schedule_buffers_and_zero_based_batch_indexing():
    process = GaussianDiffusion()
    assert process.timesteps == 1000
    assert set(dict(process.named_buffers())) == {
        "betas", "alpha_bars", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"
    }
    expected = torch.cumprod(1 - process.betas.double(), dim=0).float()
    assert torch.allclose(process.alpha_bars, expected, atol=1e-7, rtol=1e-6)
    assert 3e-5 < process.alpha_bars[-1].item() < 5e-5
    u0 = torch.tensor([[1., 2.], [3., -4.], [5., 6.]])
    eps = torch.tensor([[-1., 2.], [3., 4.], [0., -2.]])
    t = torch.tensor([0, 333, 999])
    noisy = process.q_sample(u0, t, eps)
    for row, level in enumerate(t):
        a = process.alpha_bars[level]
        assert torch.allclose(noisy[row], a.sqrt() * u0[row] + (1 - a).sqrt() * eps[row])
    assert torch.allclose(process.predict_x0(noisy, t, eps), u0, atol=4e-5)


def test_diffusion_dtype_and_invalid_timestep():
    process = GaussianDiffusion()
    u = torch.ones(2, 3, dtype=torch.float64)
    t = torch.tensor([0, 999], dtype=torch.long)
    noisy = process.q_sample(u, t, torch.zeros_like(u))
    assert noisy.dtype == torch.float64 and noisy.device.type == "cpu"
    assert process.predict_x0(noisy, t, torch.zeros_like(u)).dtype == torch.float64
    with pytest.raises(ValueError, match="0-based"):
        process.q_sample(u, torch.tensor([0, 1000]), torch.zeros_like(u))
    with pytest.raises(ValueError, match="int64"):
        process.q_sample(u, torch.tensor([0., 1.]), torch.zeros_like(u))


def test_fixed_noise_loss_and_finite_gradients():
    model = ConditionalDenoiser(5, width=16, blocks=1)
    process = GaussianDiffusion()
    clean = torch.randn(3, 5)
    noise = torch.randn_like(clean)
    t = torch.tensor([0, 40, 999])
    condition = torch.randn(3, 2)
    loss = process.training_loss(model, clean, t, condition, noise=noise)
    direct = (model(process.q_sample(clean, t, noise), t, condition) - noise).square().mean()
    assert torch.equal(loss, direct)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


class _PresenceOracle(torch.nn.Module):
    def forward(self, u_t, t, conditions, condition_present=None):
        return condition_present[:, None].to(u_t.dtype).expand_as(u_t) * 2 + 1


def test_classifier_free_guidance_zero_and_one():
    process = GaussianDiffusion()
    model = _PresenceOracle()
    u = torch.zeros(2, 4)
    t = torch.tensor([10, 80])
    c = torch.zeros(2, 2)
    assert torch.equal(process.guided_epsilon(model, u, t, c, guidance=0), torch.ones_like(u))
    assert torch.equal(process.guided_epsilon(model, u, t, c, guidance=1), 3 * torch.ones_like(u))
    assert torch.equal(process.guided_epsilon(model, u, t, c, guidance=2), 5 * torch.ones_like(u))


class _NoiseOracle(torch.nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.register_buffer("epsilon", epsilon)

    def forward(self, u_t, t, conditions, condition_present=None):
        return self.epsilon


@pytest.mark.parametrize("steps", [20, 50, 100, 1000])
def test_ddim_oracle_reverse_and_clean_endpoint(steps):
    process = GaussianDiffusion()
    levels = process.ddim_timesteps(steps)
    assert levels[0] == 999 and levels[-1] == 0
    assert bool((levels[:-1] > levels[1:]).all())
    # Float64 isolates the full-1000-step algebra from accumulated float32
    # roundoff; the 20/50/100-step parameterizations also exercise the API.
    dtype = torch.float64 if steps == 1000 else torch.float32
    clean = torch.tensor([[0.1, -0.4], [1.2, 0.5]], dtype=dtype)
    epsilon = torch.tensor([[1., 0.2], [-0.7, 0.3]], dtype=dtype)
    start = process.q_sample(clean, torch.full((2,), 999, dtype=torch.long), epsilon)
    result = process.ddim_sample(_NoiseOracle(epsilon), torch.zeros(2, 2), start,
                                 steps=steps, guidance=1)
    assert torch.allclose(result, clean, atol=3e-5)
    near_clean = process.q_sample(clean, torch.zeros(2, dtype=torch.long), epsilon)
    assert torch.allclose(process.ddim_step(near_clean, 0, -1, epsilon), clean, atol=3e-5)


def test_sampler_replay_and_nonfinite_failure():
    torch.manual_seed(7)
    process = GaussianDiffusion()
    model = ConditionalDenoiser(4, width=16, blocks=2).eval()
    condition = torch.randn(3, 2)
    initial = torch.randn(3, 4)
    original = initial.clone()
    first = process.ddim_sample(model, condition, initial, steps=20)
    second = process.ddim_sample(model, condition, initial, steps=20)
    assert torch.equal(first, second)
    assert torch.equal(initial, original)
    with pytest.raises(FloatingPointError, match="Nonfinite DDIM input"):
        process.ddim_sample(model, condition, torch.full_like(initial, float("nan")))


def test_core_import_does_not_load_materials_packages():
    code = ("import latent_diffusion; import sys; "
            "assert 'pymatgen' not in sys.modules and 'ase' not in sys.modules and 'meidnet' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
