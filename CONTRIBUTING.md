# Contributing to Syntropy

Thank you for your interest in contributing to Syntropy! This document provides guidelines and information for contributors.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub:
1. Check existing issues to avoid duplicates
2. Use a clear, descriptive title
3. Provide as much relevant information as possible
4. Include code samples or test cases if applicable

### Submitting Changes

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the code style guidelines below
3. **Add tests** for any new functionality
4. **Run the test suite** to ensure all tests pass
5. **Submit a pull request** with a clear description of your changes

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/syntropy.git
cd syntropy

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=syntropy --cov-report=html
```

## Style Guide

Syntropy follows five core design principles. New contributions should adhere
to these, and reviewers will check pull requests against them explicitly.

### 1. Functional style

Prefer small, modular, composable, pure functions. Each function should
implement a single, well-defined mathematical operation, rather than several
unrelated computations bundled together behind a string flag or an if/elif
dispatch. In particular, avoid:

- Mutating input arguments in place.
- Hidden I/O (e.g. `print` statements) inside otherwise-pure numerical routines.
- One function computing several independently-useful quantities (e.g. don't
  compute total correlation, dual total correlation, and O-information all
  inside a single function just because they share intermediate terms --
  factor the shared work into its own helper instead).
- Classes or other stateful objects where a plain function would do. The one
  accepted exception is `syntropy.neural`, where PyTorch `nn.Module`s are the
  idiomatic way to represent a normalizing flow, and training a flow
  necessarily updates its parameters in place.

### 2. Minimal global state

Module-level globals should be constants only (named reference distributions,
mathematical constants such as `LN_TWO_PI_E`, and the like). Functions must
never read or mutate shared mutable state:

- No module-level mutable containers (`list`, `dict`, `set`) that get written
  to after import.
- No `global` statements.
- Any function that needs randomness must accept an explicit `rng`/`seed`
  argument (e.g. via `np.random.default_rng(seed)`) rather than calling the
  global `np.random`/`random` APIs directly. This keeps results reproducible
  and stops one function's randomness from silently affecting another's.

### 3. Heavy type-hinting

Every function signature -- parameters and return type -- should be fully
type-hinted. Prefer specific types over bare containers: `dict[tuple[Any,
...], float]` instead of `dict`, `NDArray[np.floating]` instead of `NDArray`,
`Callable[[Iterable[int]], int]` instead of bare `Callable`. Arguments that
default to `None` should be typed `X | None`, not just `X`.

### 4. Heavy documentation, with citations

Every non-trivial function needs a docstring with real Parameters/Returns
descriptions, not just a repeated type with no prose. If a function
implements a specific technique from the literature -- an entropy estimator,
an information decomposition, a redundancy function, a network architecture
-- its docstring must include a References section citing the original
paper, in the same format already used throughout the codebase (author,
year, title, venue, DOI/URL).

### 5. Heavy testing for mathematical correctness

Every function exported in a submodule's `__all__` must have a corresponding
test with a real numeric assertion: a hand-derived value, a closed-form
analytic limit, a cross-check against an independent implementation, or a
known invariant (e.g. `I(X;X) = H(X)`). A test that only checks a function
runs without raising is not sufficient. Untested code is exactly where bugs
hide silently -- treat a missing test on an exported function as equivalent
to a missing implementation.

## Use of AI
Contributors are allowed to use AI for help with coding (e.g. Claude Code, Cursor, etc), however, contributors **must** be able to articulate, in their own words, what the added functionality accomplishes, its internal logic, and justify its inclusion in the codebase. 

The Syntropy project is not anti-AI, but it is opposed to mindless cognitive offloading. We reserve the right to revisit this policy in the event that we get deluged with slop-PRs, or we feel that illegible AI-generated code is compromising the maintainability or integrity of the codebase. 

## Project Structure

- `syntropy/discrete/` - Estimators for discrete random variables
- `syntropy/gaussian/` - Estimators for Gaussian random variables
- `syntropy/knn/` - k-nearest neighbor estimators for continuous variables
- `syntropy/neural/` - Neural network-based estimators
- `syntropy/mixed/` - Estimators for mixed discrete/continuous variables
- `syntropy/lattices/` - Partial information lattice structures
- `tests/` - Test suite

## Testing

All contributions should include appropriate tests. Tests are located in the `tests/` directory and use pytest.

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_discrete.py

# Run with coverage report
pytest --cov=syntropy
```

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.
