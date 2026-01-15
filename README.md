# Random Fourier Features Encoding

This fork adds a new encoding type `RandomFourierFeatures` that implements random Fourier features [Rahimi & Recht (2007)](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). This encoding was proposed in [Tancik et al. (2020)](https://arxiv.org/abs/2006.10739) as a swap-in replacement for the positional encoding used in the original NeRF paper [Mildenhall et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3503250)

## Mathematical Background

Unlike the standard `FrequencyEncoding` which uses axis-aligned frequencies at powers of 2, this encoding uses random Gaussian frequency vectors:

```
γ(x) = [cos(2πBx), sin(2πBx)]
```

Where:
- `x` is the input vector (n_dims_to_encode dimensions)
- `B` is a (n_features × n_dims_to_encode) matrix with entries sampled from N(0, σ²)
- `σ` (scale parameter) controls the bandwidth
- Output dimension: `2 * n_features`

## Configuration

```json
{
    "otype": "RandomFourierFeatures",
    "n_features": 128,
    "scale": 10.0,
    "seed": 1337
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_features` | 128 | Number of random frequency vectors |
| `scale` | 10.0 | Standard deviation of Gaussian frequencies (controls bandwidth) |
| `seed` | 1337 | Random seed for reproducibility |

## Usage

### Standalone Encoding

```python
import tinycudann_rff as tcnn
import torch

encoding = tcnn.Encoding(
    n_input_dims=3,
    encoding_config={
        "otype": "RandomFourierFeatures",
        "n_features": 128,
        "scale": 10.0,
        "seed": 42,
    },
    dtype=torch.float32,  # or torch.float16
)

x = torch.rand(1024, 3, device="cuda")
y = encoding(x)  # shape: (1024, 256)
```

### With Network

```python
model = tcnn.NetworkWithInputEncoding(
    n_input_dims=3,
    n_output_dims=1,
    encoding_config={
        "otype": "RandomFourierFeatures",
        "n_features": 64,
        "scale": 10.0,
    },
    network_config={
        "otype": "FullyFusedMLP",  # or "CutlassMLP" for larger widths
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    },
)

x = torch.rand(1024, 3, device="cuda")
y = model(x)  # shape: (1024, 1)
```

## Installation

### Prerequisites

- CUDA Toolkit 11.0+
- PyTorch with CUDA support
- C++17 compatible compiler

### Install as `tinycudann_rff`

This installs alongside any existing `tinycudann` installation:

```bash
# Clone and enter the repository
git clone <repo-url> tiny-cuda-ffns
cd tiny-cuda-ffns
git submodule update --init --recursive

# Install PyTorch bindings
cd bindings/torch
pip install .

# (optional) Test the installation
cd ../..
python test_random_fourier_features.py
```

### Import

```python
import tinycudann_rff as tcnn  # This fork
import tinycudann              # Original (if installed)
```

## Notes

- **Reproducibility**: Same seed produces identical frequency matrices
- **Scale parameter**: Higher values = higher frequency content = more detail but potentially harder to optimize
- **Large n_features**: No hard limit, but very large values may slow JIT compilation. For >64 features (>128 output dims), consider using `CutlassMLP` instead of `FullyFusedMLP`
- **Precision**: Encoding supports runtime dtype selection (`torch.float32` or `torch.float16`). Networks use compile-time precision (set `TCNN_HALF_PRECISION=0` or `1` before `pip install`)
