import torch
import pytest
from smol_sae.base import Config
from smol_sae.vanilla import VanillaSAE
from transformer_lens import HookedTransformer

@pytest.fixture 
def device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

@pytest.fixture 
def model(device):
    return HookedTransformer.from_pretrained("gelu-1l").to(device)

@pytest.fixture
def config(device):
    return Config(
        n_buffers=100,
        expansion=4,
        buffer_size=2**8,
        sparsities=(0.1, 1.0),
        device = device
    )


def test_vanilla(config, model, device):
    sae = VanillaSAE(config, model)
    n_batch = 32
    input = torch.randn(
        n_batch, sae.n_instances, model.cfg.d_model
    ).to(device)
    output = sae(input)
    assert output.shape == input.shape
    