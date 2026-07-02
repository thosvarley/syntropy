import numpy as np
import pytest

# The neural estimators are an optional extra; skip the whole module if their
# dependencies are not installed.
torch = pytest.importorskip("torch")
nflows = pytest.importorskip("nflows")

from syntropy.neural.utils import initialize_flow, train_flow, evaluate_flow

# Tiny, fast configuration mirroring tests/test_neural.py: these tests check
# structural properties (types, shapes, that training moves parameters, that
# convergence_threshold stops early), which hold regardless of training
# quality.
FAST_FLOW = {"num_layers": 2, "hidden_features": 8}
FAST_N = 64


def _toy_data(n=FAST_N, d=2, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


def test_initialize_flow():
    flow = initialize_flow(dim=3, **FAST_FLOW)
    assert isinstance(flow, nflows.flows.base.Flow)

    log_probs = flow.log_prob(_toy_data(d=3))
    assert log_probs.shape == (FAST_N,)
    assert torch.isfinite(log_probs).all()

    # Conditioning (dim_context > 0) works the same way.
    conditioned = initialize_flow(dim=2, dim_context=1, **FAST_FLOW)
    context_log_probs = conditioned.log_prob(
        _toy_data(d=2), context=torch.randn(FAST_N, 1)
    )
    assert context_log_probs.shape == (FAST_N,)
    assert torch.isfinite(context_log_probs).all()


def test_train_flow_updates_parameters_in_place():
    flow = initialize_flow(dim=2, **FAST_FLOW)
    before = [p.clone() for p in flow.parameters()]

    trained = train_flow(flow, _toy_data(n=128), num_epochs=3, batch_size=32)

    assert trained is flow
    assert any(not torch.allclose(b, a) for b, a in zip(before, flow.parameters()))


def test_train_flow_with_context():
    flow = initialize_flow(dim=2, dim_context=1, **FAST_FLOW)
    data = _toy_data()
    context = torch.randn(FAST_N, 1)

    trained = train_flow(flow, data, context=context, num_epochs=2, batch_size=32)

    log_probs = trained.log_prob(data, context=context)
    assert torch.isfinite(log_probs).all()


def test_train_flow_convergence_threshold(capsys):
    # A pathologically high threshold is satisfied after the very first
    # epoch (coefficient of variation is always finite and far below 1e10),
    # so training should stop well before num_epochs.
    train_flow(
        initialize_flow(dim=2, **FAST_FLOW),
        _toy_data(),
        num_epochs=20,
        batch_size=32,
        convergence_threshold=1e10,
        verbose=True,
    )
    epochs_with_high_threshold = capsys.readouterr().out.count("Epoch")
    assert 0 < epochs_with_high_threshold < 20

    # The default threshold (0.0) is a strict "<" comparison against a
    # coefficient of variation that starts positive, so it never triggers
    # and every requested epoch runs.
    train_flow(
        initialize_flow(dim=2, **FAST_FLOW),
        _toy_data(),
        num_epochs=4,
        batch_size=32,
        convergence_threshold=0.0,
        verbose=True,
    )
    assert capsys.readouterr().out.count("Epoch") == 4


def test_evaluate_flow_output_matches_log_prob():
    flow = initialize_flow(dim=2, **FAST_FLOW)
    data = _toy_data()

    ptw, h = evaluate_flow(flow, data)

    assert ptw.shape == (FAST_N,)
    assert np.isfinite(h)
    assert float(ptw.mean()) == pytest.approx(h, abs=1e-5)

    with torch.no_grad():
        log_probs = flow.log_prob(data)
    assert torch.allclose(ptw, -log_probs)


def test_evaluate_flow_with_context():
    flow = initialize_flow(dim=2, dim_context=1, **FAST_FLOW)
    ptw, h = evaluate_flow(flow, _toy_data(), context=torch.randn(FAST_N, 1))

    assert ptw.shape == (FAST_N,)
    assert np.isfinite(h)


def test_evaluate_flow_sets_eval_mode():
    flow = initialize_flow(dim=2, **FAST_FLOW)
    flow.train()
    assert flow.training is True

    evaluate_flow(flow, _toy_data())

    assert flow.training is False
