# Syntropy

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/syntropy/badge/?version=latest)](https://syntropy.readthedocs.io/en/latest/)

**Syntropy** is a Python library for multivariate information-theoretic analysis of discrete and continuous data. It provides efficient implementations of information measures ranging from basic quantities like entropy and mutual information to modern constructs like the partial information decomposition, O-information, and information rates for time series.

## Features

- **Multiple estimators**: Discrete, Gaussian, KNN (Kraskov), and neural (normalizing flow) estimators
- **Pointwise measures**: Access local/pointwise values, not just expected values
- **Higher-order information**: Total correlation, dual total correlation, O-information, S-information
- **Information decomposition**: Partial information decomposition (PID) and partial entropy decomposition
- **Time series**: Information rates and Lempel-Ziv complexity measures
- **Consistent API**: Same interface across all estimator types

## Installation

```bash
git clone https://github.com/thosvarley/syntropy.git
cd syntropy
pip install .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Discrete Distributions

Discrete estimators work with probability distributions represented as dictionaries:

```python
from syntropy.discrete import mutual_information, o_information

# XOR distribution: pure synergy
xor = {
    (0, 0, 0): 0.25,
    (0, 1, 1): 0.25,
    (1, 0, 1): 0.25,
    (1, 1, 0): 0.25,
}

# Mutual information between inputs (0,1) and output (2)
ptw, mi = mutual_information(idxs_x=(0, 1), idxs_y=(2,), joint_distribution=xor)
print(f"I(X0,X1 ; X2) = {mi:.3f} bits")  # 1.0 bit

# O-information (negative = synergy-dominated)
ptw, omega = o_information(idxs=(0, 1, 2), joint_distribution=xor)
print(f"Omega = {omega:.3f} bits")  # -1.0 bit
```

### Gaussian Estimator

For continuous data with approximately Gaussian distributions:

```python
import numpy as np
from syntropy.gaussian import mutual_information, total_correlation

# Generate correlated Gaussian data
n = 10_000
x = np.random.randn(n)
y = 0.8 * x + 0.6 * np.random.randn(n)
z = 0.5 * x + 0.866 * np.random.randn(n)
data = np.vstack([x, y, z])

cov = np.cov(data)
mi = mutual_information(idxs_x=(0,), idxs_y=(1,), cov=cov)
tc = total_correlation(idxs=(0, 1, 2), cov=cov)

print(f"I(X ; Y) = {mi:.3f} nats")
print(f"TC(X, Y, Z) = {tc:.3f} nats")
```

### KNN Estimator (Kraskov)

Non-parametric estimation for continuous data:

```python
import numpy as np
from syntropy.knn import mutual_information

# Non-linear relationship
n = 5_000
x = np.random.randn(n)
y = x**2 + 0.5 * np.random.randn(n)
data = np.vstack([x, y])

ptw, mi = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, k=5)
print(f"I(X ; Y) = {mi:.3f} nats")
```

### Neural Estimator

For complex, high-dimensional distributions using normalizing flows:

```python
import torch
from syntropy.neural import mutual_information

# Generate data (samples x features format)
n = 10_000
x = torch.randn(n)
y = 0.7 * x + 0.714 * torch.randn(n)
data = torch.stack([x, y], dim=1)

ptw, mi = mutual_information(idxs_x=(0,), idxs_y=(1,), data=data, verbose=True)
print(f"I(X ; Y) = {mi:.3f} nats")
```

### Mixed Discrete-Continuous

For mutual information between discrete and continuous variables:

```python
import numpy as np
from syntropy.mixed import mutual_information

n = 10_000
continuous = np.random.randn(1, n)
discrete = (continuous > 0).astype(int)

ptw, mi = mutual_information(discrete_vars=discrete, continuous_vars=continuous)
print(f"I(discrete ; continuous) = {mi:.3f} nats")
```

## Available Measures

| Measure | Discrete | Gaussian | KNN | Neural | Mixed |
|---------|:--------:|:--------:|:---:|:------:|:-----:|
| Entropy | x | x | x | x | x |
| Conditional Ent. | x | x | x | x | x |
| Mutual Information | x | x | x | x | x |
| Conditional MI | x | x | x | x |
| KL Divergence | x | x | | |
| Total Correlation | x | x | x | x |
| Dual Total Correlation | x | x | x | x |
| O-Information | x | x | x | x |
| S-Information | x | x | x | x |
| Co-Information | x | | | |
| TSE Complexity | x | x | | | 
| Partial Info. Decomp. | x | x | | |
| Partial Entropy Decomp. | x | x | | |
| Generalized Info. Decomp. | x | x | | |
| Information Rates | x | x | | |
| Connected Information | x| | | |

### Optimizations and Utilities

Syntropy also includes a number of optimization algorithms. 

* Finding optimally-synergistic submatrices from a covariance matrix (as done by [Varley, Pope et al., 2023](https://www.nature.com/articles/s42003-023-04843-w)).
* Finding the maximum-entropy discrete distribution consistent with k-order marginals (as done in the [DIT package](https://dit.readthedocs.io/en/latest/optimization.html?highlight=maxentoptimizer)).

In the ```utils.py``` files, you can also find a number of utility functions for interacting with discrete and continuous probability distributions. 

## Documentation

Full documentation is available at [syntropy.readthedocs.io](https://syntropy.readthedocs.io).

- [Quickstart Guide](https://syntropy.readthedocs.io/en/latest/quickstart.html)
- [Theory Primer](https://syntropy.readthedocs.io/en/latest/theory.html)
- [API Reference](https://syntropy.readthedocs.io/en/latest/api/syntropy.html)

## Testing

```bash
pytest tests/
```

## Citation

If you use Syntropy in your research, please cite:

```bibtex
@software{syntropy,
  author = {Varley, Thomas F.},
  title = {Syntropy: Multivariate Information Theory for Python},
  url = {https://github.com/thosvarley/syntropy},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
