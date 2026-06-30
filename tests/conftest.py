"""Shared pytest configuration.

Slow tests (e.g. neural normalizing-flow convergence) are skipped by default so
the everyday suite stays fast. Run them explicitly with::

    pytest --runslow
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow tests (e.g. neural estimator convergence)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
