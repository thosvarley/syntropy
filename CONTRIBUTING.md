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

## Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

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
