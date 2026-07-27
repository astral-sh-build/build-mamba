from copy import deepcopy
from importlib.metadata import version

from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn,
    selective_scan_ref,
)
import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def scan_inputs(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batch, channels, states, length = 2, 8, 4, 32
    return {
        "u": torch.randn((batch, channels, length), device=device, dtype=dtype),
        "delta": 0.5
        * torch.rand((batch, channels, length), device=device, dtype=dtype),
        "A": -0.5 * torch.rand((channels, states), device=device),
        "B": torch.randn((batch, states, length), device=device, dtype=dtype),
        "C": torch.randn((batch, states, length), device=device, dtype=dtype),
        "D": torch.randn(channels, device=device),
        "z": torch.randn((batch, channels, length), device=device, dtype=dtype),
        "delta_bias": 0.5 * torch.rand(channels, device=device),
    }


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    elif actual.dtype == torch.float16:
        rtol, atol = 3e-3, 5e-3
    else:
        rtol, atol = 6e-4, 2e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_published_cuda_wheels(device: torch.device) -> None:
    assert version("mamba-ssm") == "2.3.1+cu.12.8.torch.2.10"
    assert version("causal-conv1d") == "1.6.2.post1+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("delta_softplus", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_selective_scan(
    device: torch.device,
    dtype: torch.dtype,
    delta_softplus: bool,
) -> None:
    inputs = scan_inputs(device, dtype)

    actual = selective_scan_fn(**inputs, delta_softplus=delta_softplus)
    expected = selective_scan_ref(**inputs, delta_softplus=delta_softplus)

    assert_close(actual, expected)


def test_selective_scan_final_state(device: torch.device) -> None:
    inputs = scan_inputs(device)

    actual, state = selective_scan_fn(
        **inputs,
        delta_softplus=True,
        return_last_state=True,
    )
    expected, expected_state = selective_scan_ref(
        **inputs,
        delta_softplus=True,
        return_last_state=True,
    )

    assert_close(actual, expected)
    assert_close(state, expected_state)


def test_selective_scan_backward(device: torch.device) -> None:
    inputs = {
        name: value.detach().clone().requires_grad_()
        for name, value in scan_inputs(device).items()
    }
    expected_inputs = {
        name: value.detach().clone().requires_grad_()
        for name, value in inputs.items()
    }

    actual = selective_scan_fn(**inputs, delta_softplus=True)
    expected = selective_scan_ref(**expected_inputs, delta_softplus=True)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)

    assert_close(actual, expected)
    for name, value in inputs.items():
        expected_gradient = expected_inputs[name].grad
        assert value.grad is not None, f"Missing gradient for {name}"
        assert expected_gradient is not None, f"Missing reference gradient for {name}"
        torch.testing.assert_close(
            value.grad,
            expected_gradient,
            rtol=3e-3,
            atol=1e-2,
            msg=f"Incorrect selective-scan gradient for {name}",
        )


def test_fused_mamba_model(device: torch.device) -> None:
    torch.manual_seed(0)
    model = Mamba(
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        device=device,
    ).eval()
    reference = deepcopy(model)
    reference.use_fast_path = False
    source = torch.randn((2, 32, 32), device=device)

    with torch.no_grad():
        actual = model(source)
        expected = reference(source)

    assert actual.shape == source.shape
    assert_close(actual, expected)


def test_mamba_streaming_state(device: torch.device) -> None:
    torch.manual_seed(0)
    model = Mamba(
        d_model=32,
        d_state=8,
        d_conv=4,
        expand=2,
        use_fast_path=False,
        device=device,
    ).eval()
    source = torch.randn((2, 16, 32), device=device)
    convolution_state, selective_state = model.allocate_inference_cache(
        batch_size=2,
        max_seqlen=16,
    )

    with torch.no_grad():
        expected = model(source)
        outputs = []
        for token in source.split(1, dim=1):
            output, convolution_state, selective_state = model.step(
                token,
                convolution_state,
                selective_state,
            )
            outputs.append(output)

    assert_close(torch.cat(outputs, dim=1), expected)
