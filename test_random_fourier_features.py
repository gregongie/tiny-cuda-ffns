#!/usr/bin/env python3
"""
Simple test for the RandomFourierFeatures encoding.

Run after installing tinycudann:
    cd bindings/torch && pip install .
    cd ../.. && python test_random_fourier_features.py
"""

import torch
import tinycudann_rff as tcnn
import numpy as np


def test_basic_forward():
    """Test basic forward pass and output dimensions."""
    print("Testing basic forward pass...")

    n_input_dims = 3
    n_features = 64
    batch_size = 1024

    encoding = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 10.0,
            "seed": 1337,
        },
        dtype=torch.float32,
    )

    # Check output dimensions
    expected_output_dims = n_features * 2  # cos and sin for each feature
    assert encoding.n_output_dims == expected_output_dims, \
        f"Expected {expected_output_dims} output dims, got {encoding.n_output_dims}"

    # Run forward pass
    x = torch.rand(batch_size, n_input_dims, device="cuda")
    y = encoding(x)

    assert y.shape == (batch_size, expected_output_dims), \
        f"Expected shape {(batch_size, expected_output_dims)}, got {y.shape}"

    # Check output is bounded (cos and sin are in [-1, 1])
    assert y.min() >= -1.1 and y.max() <= 1.1, \
        f"Output should be bounded, got min={y.min():.3f}, max={y.max():.3f}"

    print(f"  Output shape: {y.shape}")
    print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")
    print("  PASSED")


def test_backward():
    """Test backward pass (gradient computation)."""
    print("\nTesting backward pass...")

    n_input_dims = 3
    n_features = 32
    batch_size = 256

    encoding = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 10.0,
            "seed": 42,
        },
        dtype=torch.float32,
    )

    x = torch.rand(batch_size, n_input_dims, device="cuda", requires_grad=True)
    y = encoding(x)

    # Compute loss and backward
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "Input gradient should not be None"
    assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x.shape}"
    assert not torch.isnan(x.grad).any(), "Gradient contains NaN"
    assert not torch.isinf(x.grad).any(), "Gradient contains Inf"

    print(f"  Gradient shape: {x.grad.shape}")
    print(f"  Gradient range: [{x.grad.min():.3f}, {x.grad.max():.3f}]")
    print("  PASSED")


def test_reproducibility():
    """Test that same seed produces same results."""
    print("\nTesting reproducibility (same seed)...")

    n_input_dims = 2
    n_features = 16
    batch_size = 128
    seed = 12345

    config = {
        "otype": "RandomFourierFeatures",
        "n_features": n_features,
        "scale": 5.0,
        "seed": seed,
    }

    encoding1 = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=config, dtype=torch.float32)
    encoding2 = tcnn.Encoding(n_input_dims=n_input_dims, encoding_config=config, dtype=torch.float32)

    x = torch.rand(batch_size, n_input_dims, device="cuda")

    y1 = encoding1(x)
    y2 = encoding2(x)

    assert torch.allclose(y1, y2, atol=1e-5), \
        f"Same seed should produce same results, max diff: {(y1 - y2).abs().max()}"

    print(f"  Max difference: {(y1 - y2).abs().max():.2e}")
    print("  PASSED")


def test_different_seeds():
    """Test that different seeds produce different results."""
    print("\nTesting different seeds produce different results...")

    n_input_dims = 2
    n_features = 16
    batch_size = 128

    encoding1 = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 5.0,
            "seed": 1111,
        },
        dtype=torch.float32,
    )

    encoding2 = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 5.0,
            "seed": 2222,
        },
        dtype=torch.float32,
    )

    x = torch.rand(batch_size, n_input_dims, device="cuda")

    y1 = encoding1(x)
    y2 = encoding2(x)

    # Results should be different
    assert not torch.allclose(y1, y2, atol=1e-3), \
        "Different seeds should produce different results"

    print(f"  Max difference: {(y1 - y2).abs().max():.3f}")
    print("  PASSED")


def test_scale_parameter():
    """Test that scale parameter affects output variance."""
    print("\nTesting scale parameter effect...")

    n_input_dims = 3
    n_features = 64
    batch_size = 1024

    # Low scale -> low frequency -> smoother output
    encoding_low = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 1.0,
            "seed": 42,
        },
        dtype=torch.float32,
    )

    # High scale -> high frequency -> more variation
    encoding_high = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 100.0,
            "seed": 42,
        },
        dtype=torch.float32,
    )

    # Use closely spaced points to see smoothness difference
    x = torch.linspace(0, 0.1, batch_size, device="cuda").unsqueeze(1).expand(-1, n_input_dims)

    y_low = encoding_low(x)
    y_high = encoding_high(x)

    # Compute variance of differences between consecutive outputs
    diff_low = (y_low[1:] - y_low[:-1]).abs().mean()
    diff_high = (y_high[1:] - y_high[:-1]).abs().mean()

    print(f"  Low scale (1.0) avg consecutive diff: {diff_low:.4f}")
    print(f"  High scale (100.0) avg consecutive diff: {diff_high:.4f}")

    # High scale should have more variation
    assert diff_high > diff_low, \
        "Higher scale should produce more variation between nearby points"

    print("  PASSED")


def test_numerical_gradient():
    """Test gradient numerically using finite differences."""
    print("\nTesting numerical gradient (finite differences)...")

    n_input_dims = 2
    n_features = 8
    batch_size = 4
    eps = 1e-4

    encoding = tcnn.Encoding(
        n_input_dims=n_input_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 5.0,
            "seed": 42,
        },
        dtype=torch.float32,
    )

    x = torch.rand(batch_size, n_input_dims, device="cuda", requires_grad=True)

    # Analytical gradient
    y = encoding(x)
    loss = y.sum()
    loss.backward()
    analytical_grad = x.grad.clone()

    # Numerical gradient
    numerical_grad = torch.zeros_like(x)
    for i in range(batch_size):
        for j in range(n_input_dims):
            x_plus = x.detach().clone()
            x_plus[i, j] += eps
            y_plus = encoding(x_plus)

            x_minus = x.detach().clone()
            x_minus[i, j] -= eps
            y_minus = encoding(x_minus)

            numerical_grad[i, j] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

    # Compare
    max_diff = (analytical_grad - numerical_grad).abs().max()
    rel_diff = (analytical_grad - numerical_grad).abs() / (numerical_grad.abs() + 1e-8)
    max_rel_diff = rel_diff.max()

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")

    assert max_diff < 1e-2 or max_rel_diff < 1e-2, \
        f"Gradient check failed: max_diff={max_diff:.2e}, max_rel_diff={max_rel_diff:.2e}"

    print("  PASSED")


def test_with_network():
    """Test encoding combined with a network."""
    print("\nTesting encoding with network...")

    n_input_dims = 3
    n_features = 64
    n_output_dims = 1
    batch_size = 1024

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=n_input_dims,
        n_output_dims=n_output_dims,
        encoding_config={
            "otype": "RandomFourierFeatures",
            "n_features": n_features,
            "scale": 10.0,
            "seed": 42,
        },
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
    )

    x = torch.rand(batch_size, n_input_dims, device="cuda")
    y = model(x)

    assert y.shape == (batch_size, n_output_dims), \
        f"Expected shape {(batch_size, n_output_dims)}, got {y.shape}"

    # Test backward
    loss = y.sum()
    loss.backward()

    print(f"  Output shape: {y.shape}")
    print(f"  Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("  PASSED")


def main():
    print("=" * 60)
    print("RandomFourierFeatures Encoding Tests")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return

    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print()

    test_basic_forward()
    test_backward()
    test_reproducibility()
    test_different_seeds()
    test_scale_parameter()
    test_numerical_gradient()
    test_with_network()

    print()
    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
