"""Smoke test: every public submodule imports cleanly."""
import importlib
import pytest

CORE = ["syntropy.discrete", "syntropy.gaussian", "syntropy.knn",
        "syntropy.mixed", "syntropy.lattices"]


@pytest.mark.parametrize("module", CORE)
def test_core_imports(module):
    assert importlib.import_module(module) is not None


def test_neural_imports():
    pytest.importorskip("torch")
    pytest.importorskip("nflows")
    assert importlib.import_module("syntropy.neural") is not None
