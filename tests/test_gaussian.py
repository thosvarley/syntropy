import pytest
import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats

import syntropy.gaussian.shannon as shannon
import syntropy.gaussian.multivariate_mi as mi
from syntropy.gaussian.decompositions import (
    partial_information_decomposition as pid,
    integrated_information_decomposition as phiid,
    partial_entropy_decomposition as ped,
    generalized_information_decomposition as gid,
    idep_partial_information_decomposition as idep,
)
from syntropy.gaussian.temporal import (
    differential_entropy_rate,
    mutual_information_rate,
    total_correlation_rate,
    o_information_rate,
)

data_path = pathlib.Path(__file__).parent
data = pd.read_csv(data_path / "../examples/bold.csv", header=None).values

cov = np.cov(data, ddof=0.0)

# Due to natural instability in Scipy's matrix algebra, we need a slightly
# more relaxed tolerance for our unit tests. 1 part in 1,000,000 is probably ok.
pytest_abs = 1e-6

def test_differential_entropy():
    h_all = shannon.local_differential_entropy(data).mean()
    assert shannon.differential_entropy(cov) == pytest.approx(h_all, abs=pytest_abs)

    h1 = shannon.local_differential_entropy(
        data=data[(1, 2), :], cov=cov[np.ix_((1, 2), (1, 2))]
    ).mean()
    h2 = shannon.local_differential_entropy(
        data=data[(1, 2), :],
    ).mean()
    assert h1 == pytest.approx(h2, abs=pytest_abs)

    assert h1 == pytest.approx(
        shannon.differential_entropy(cov=cov, idxs=(1, 2)), abs=pytest_abs
    )

    h = shannon.differential_entropy(cov[np.ix_((2,), (2,))])
    c = shannon.conditional_entropy((2,), (1,), cov)

    assert h1 - h == pytest.approx(c, abs=pytest_abs)


def test_mutual_information():
    i = shannon.mutual_information((0,), (1,), cov)
    r, _ = stats.pearsonr(data[0, :], data[1, :])

    assert i == pytest.approx(-np.log(1 - (r**2)) / 2, abs=pytest_abs)

    lmi = shannon.local_mutual_information((0,), (1,), data)
    assert i == pytest.approx(lmi.mean(), abs=pytest_abs)

    tc = mi.total_correlation(cov[(0, 1), :][:, (0, 1)])
    assert i == pytest.approx(tc, abs=pytest_abs)


def test_higher_order_mi():
    idxs = (0, 1, 2, 3)
    cov_ix = cov[np.ix_(idxs, idxs)]

    tc = mi.total_correlation(cov, idxs)
    dtc = mi.dual_total_correlation(cov, idxs)
    o = mi.o_information(cov, idxs)
    s = mi.s_information(cov, idxs)

    assert tc - dtc == pytest.approx(o, abs=pytest_abs)
    assert tc + dtc == pytest.approx(s, abs=pytest_abs)

    ltc = mi.local_total_correlation(data=data, cov=cov, idxs=idxs)
    ldtc = mi.local_dual_total_correlation(data=data, cov=cov, idxs=idxs)
    lo = mi.local_o_information(data=data, cov=cov, idxs=idxs)
    ls = mi.local_s_information(data=data, cov=cov, idxs=idxs)

    assert ltc.mean() == pytest.approx(tc, abs=pytest_abs)
    assert ldtc.mean() == pytest.approx(dtc, abs=pytest_abs)
    assert lo.mean() == pytest.approx(o, abs=pytest_abs)
    assert ls.mean() == pytest.approx(s, abs=pytest_abs)

    c = mi.description_complexity(cov, idxs)
    assert c == pytest.approx(dtc / len(idxs), abs=pytest_abs)

    triad = np.array(
        [
            [1.0, 0.14621677, 0.59480877],
            [0.14621677, 1.0, 0.27645032],
            [0.59480877, 0.27645032, 0.99999999],
        ]
    )

    mi_12_3 = shannon.mutual_information((0, 1), (2,), triad)
    mi_1_3 = shannon.mutual_information((0,), (2,), triad)
    mi_2_3 = shannon.mutual_information((1,), (2,), triad)

    # Equivavlance between O-information and WMS MI for 3 variables.
    wms = mi_12_3 - (mi_1_3 + mi_2_3)

    assert -wms == pytest.approx(mi.o_information(triad), abs=pytest_abs)

    # Testing the definition in terms of entropy
    whole = shannon.differential_entropy(triad)
    sum_diffs = 0.0
    for i in range(3):
        residuals = tuple(j for j in range(3) if j != i)
        sum_diffs += shannon.differential_entropy(
            triad, idxs=(i,)
        ) - shannon.differential_entropy(triad, idxs=residuals)

    assert whole + sum_diffs == pytest.approx(mi.o_information(triad), abs=pytest_abs)


def test_pid():
    idxs = (0, 1)
    target = (3, 4)
    ptw, avg = pid(idxs, target, data, cov)
    mi = shannon.mutual_information(idxs, target, cov)

    assert sum(avg.values()) == pytest.approx(mi, abs=pytest_abs)

    mi = shannon.mutual_information((0,), target, cov)
    assert avg[((0,),)] + avg[((0,), (1,))] == pytest.approx(mi, abs=pytest_abs)


def test_ped():
    idxs = (0, 1, 2, 3)

    ptw, avg = ped(data, idxs, cov)
    h = shannon.differential_entropy(cov, idxs)

    assert sum(avg.values()) == pytest.approx(h, abs=pytest_abs)


def test_gid():
    idxs = (0, 1, 2, 3)
    cov_prior = np.eye(len(idxs))
    cov_posterior = cov[np.ix_(idxs, idxs)]

    ptw, avg = gid(idxs, data[idxs,:], cov_posterior, cov_prior)

    dkl = shannon.kullback_leibler_divergence(cov_posterior, cov_prior)
    assert sum(avg.values()) == pytest.approx(dkl, abs=pytest_abs)

    ldkl = shannon.local_kullback_leibler_divergence(
        cov_posterior, cov_prior, data[idxs, :]
    )

    assert ldkl.mean() == pytest.approx(sum(avg.values()), abs=pytest_abs)


def test_phiid():
    return None

def test_oinfo_rate():
    T = 5_000_000

    noise = np.random.randn(3, T)

    _, mi_joint = mutual_information_rate((0, 1), (2,), noise, nperseg=2**13)
    _, mi_1 = mutual_information_rate((0,), (2,), noise, nperseg=2**13)
    _, mi_2 = mutual_information_rate((1,), (2,), noise, nperseg=2**13)

    _, oir = o_information_rate((0, 1, 2), noise, nperseg=2**13)

    assert oir == pytest.approx(mi_1 + mi_2 - mi_joint, abs=pytest_abs)


def test_idep_univariate():
    """
    Unit test from Kay and Ince (2018), Example 4 (p. 12).
    Lower tolerance used because fewer digits are used.
    Results in Kay and Ince are given in bits rather than nats, so we convert.
    """
    p, q, r = -0.2, 0.7, -0.7

    cov = np.array([[1.0, p, q], [p, 1.0, r], [q, r, 1.0]])

    result = idep(inputs=((0,), (1,)), target=(2,), cov=cov)

    # Convert from nats to bits
    bits_conversion = 1.0 / np.log(2)

    assert result["unq0"] * bits_conversion == pytest.approx(0.2877, abs=1e-3)
    assert result["unq1"] * bits_conversion == pytest.approx(0.2877, abs=1e-3)
    assert result["red"] * bits_conversion == pytest.approx(0.1981, abs=1e-3)
    assert result["syn"] * bits_conversion == pytest.approx(0.4504, abs=1e-3)

    mi_joint = shannon.mutual_information(idxs_x=(0, 1), idxs_y=(2,), cov=cov)

    assert sum(result.values()) == pytest.approx(mi_joint, abs=pytest_abs)


def test_idep_multivariate():
    """
    Test Idep PID on multivariate example from Kay & Ince (2018), p 24.
    Setup: (n0, n1, n2) = (3, 4, 3), (p, q, r) = (-0.15, 0.15, 0.15)
    Uses equi-correlation structure from Equation 63.

    Once again, must convert from nats to bits, more relaxed abs.
    """
    n0, n1, n2 = 3, 4, 3
    p, q, r = -0.15, 0.15, 0.15

    # Build P, Q, R with equi-correlation structure (Equation 63)
    P = p * np.ones((n0, n1))
    Q = q * np.ones((n0, n2))
    R = r * np.ones((n1, n2))

    # Build full covariance matrix
    cov = np.zeros((10, 10))

    # Diagonal blocks are identity
    cov[:n0, :n0] = np.eye(n0)
    cov[n0 : n0 + n1, n0 : n0 + n1] = np.eye(n1)
    cov[n0 + n1 :, n0 + n1 :] = np.eye(n2)

    # Off-diagonal blocks
    cov[:n0, n0 : n0 + n1] = P
    cov[n0 : n0 + n1, :n0] = P.T
    cov[:n0, n0 + n1 :] = Q
    cov[n0 + n1 :, :n0] = Q.T
    cov[n0 : n0 + n1, n0 + n1 :] = R
    cov[n0 + n1 :, n0 : n0 + n1] = R.T

    result = idep(inputs=((0, 1, 2), (3, 4, 5, 6)), target=(7, 8, 9), cov=cov)

    # Convert from nats to bits
    bits_conversion = 1.0 / np.log(2)

    # Assert matches (slightly looser tolerance for multivariate)
    assert result["unq0"] * bits_conversion == pytest.approx(0.1227, abs=1e-3)
    assert result["unq1"] * bits_conversion == pytest.approx(0.1865, abs=1e-3)
    assert result["red"] * bits_conversion == pytest.approx(0.0406, abs=1e-3)
    assert result["syn"] * bits_conversion == pytest.approx(2.4772, abs=1e-3)

    # Verify PID constraint
    total = result["red"] + result["unq0"] + result["unq1"] + result["syn"]
    mi_joint = shannon.mutual_information(idxs_x=(0,1,2,3,4,5,6), idxs_y=(7,8,9), cov=cov)

    assert total == pytest.approx(mi_joint, abs=pytest_abs)

# Absolute tolerance for all analytic comparisons.
# Tighter than before because these tests have known ground truth.
ABS_TOL = 1e-3
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_ar1(T: int, a: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Scalar AR(1): X_t = a * X_{t-1} + sigma * eps_t."""
    X = np.zeros(T)
    eps = rng.standard_normal(T) * sigma
    for t in range(1, T):
        X[t] = a * X[t - 1] + eps[t]
    return X[np.newaxis, :]  # shape (1, T)


def generate_lead_lag(
    T: int, c: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Bivariate lead-lag model:
        X_t = eps1_t                  (unit-variance white noise)
        Y_t = c * X_{t-1} + eps2_t   (unit-variance independent noise)

    Analytic MIR:
        I(X; Y) = (1/2) * log(1 + c^2)

    Derivation sketch:
        S_XX(w) = 1
        S_YY(w) = 1 + c^2          (variance of Y = c^2 * var(X) + var(eps2))
        S_XY(w) = c * e^{+iw}      (X leads Y by one step)
        |S(w)| = S_XX * S_YY - |S_XY|^2 = (1 + c^2) - c^2 = 1

        f_{X;Y}(w) = log( S_XX * S_YY / |S(w)| ) = log(1 + c^2)  [flat in w]

        MIR = (1/4pi) * integral_{-pi}^{pi} log(1 + c^2) dw
            = (1/2) * log(1 + c^2)
    """
    eps1 = rng.standard_normal(T)
    eps2 = rng.standard_normal(T)
    X = eps1
    Y = np.zeros(T)
    Y[1:] = c * X[:-1] + eps2[1:]
    Y[0] = eps2[0]
    return np.stack([X, Y], axis=0)  # shape (2, T)


# ---------------------------------------------------------------------------
# Test: AR(1) entropy rate
# ---------------------------------------------------------------------------

class TestAR1EntropyRate:
    """
    For X_t = a * X_{t-1} + sigma * eps_t with eps_t ~ N(0,1):

    The entropy RATE is the conditional entropy given the full past:
        h(X) = H(X_t | X_{t-1}, X_{t-2}, ...) = H(sigma * eps_t)
             = (1/2) * log(2 * pi * e * sigma^2)
    """

    @pytest.mark.parametrize("a, sigma", [
        (0.0, 1.0),    # white noise: rate == marginal entropy
        (0.5, 1.0),    # moderate autocorrelation
        (0.9, 1.0),    # strong autocorrelation
        (0.5, 0.1),    # small innovation noise
        (0.5, 2.0),    # large innovation noise
    ])
    def test_ar1_entropy_rate(self, a, sigma):
        rng = np.random.default_rng(42)
        T = 5_000_000
        X = generate_ar1(T, a=a, sigma=sigma, rng=rng)

        _, h = differential_entropy_rate((0,), X, nperseg=2**14)

        analytic = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
        assert h == pytest.approx(analytic, abs=ABS_TOL), (
            f"AR(1) entropy rate failed for a={a}, sigma={sigma}: "
            f"got {h:.6f}, expected {analytic:.6f}"
        )

    def test_white_noise_entropy_rate(self):
        """Special case a=0: entropy rate == marginal entropy."""
        rng = np.random.default_rng(0)
        T = 5_000_000
        noise = rng.standard_normal((1, T))

        _, h = differential_entropy_rate((0,), noise, nperseg=2**14)

        analytic = 0.5 * np.log(2 * np.pi * np.e)
        assert h == pytest.approx(analytic, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Test: bivariate lead-lag MIR
# ---------------------------------------------------------------------------

class TestLeadLagMIR:
    """
    For the lead-lag model X_t = eps1_t, Y_t = c*X_{t-1} + eps2_t:

        I(X; Y) = (1/2) * log(1 + c^2)

    This test is particularly valuable because:
    1. The spectral density matrix S(w) has non-trivial complex off-diagonal
       entries (S_XY(w) = c*e^{iw}), directly testing the complex CSD fix.
    2. The theoretical MI is flat in frequency, so the test is not sensitive
       to spectral resolution.
    3. The formula has a clean parametric form to test across multiple c values.
    """

    @pytest.mark.parametrize("c", [0.1, 0.5, 1.0, 2.0])
    def test_mir_lead_lag(self, c):
        rng = np.random.default_rng(42)
        T = 5_000_000
        data = generate_lead_lag(T, c=c, rng=rng)

        _, mir = mutual_information_rate((0,), (1,), data, nperseg=2**13)

        analytic = 0.5 * np.log(1 + c**2)
        assert mir == pytest.approx(analytic, abs=ABS_TOL), (
            f"Lead-lag MIR failed for c={c}: "
            f"got {mir:.6f}, expected {analytic:.6f}"
        )

    def test_mir_zero_coupling(self):
        """c=0: X and Y are independent, MIR should be 0."""
        rng = np.random.default_rng(0)
        T = 5_000_000
        data = generate_lead_lag(T, c=0.0, rng=rng)

        _, mir = mutual_information_rate((0,), (1,), data, nperseg=2**13)

        assert mir == pytest.approx(0.0, abs=ABS_TOL)

    def test_mir_symmetry(self):
        """MIR should be symmetric: I(X;Y) == I(Y;X)."""
        rng = np.random.default_rng(0)
        T = 5_000_000
        c = 0.7
        data = generate_lead_lag(T, c=c, rng=rng)

        _, mir_xy = mutual_information_rate((0,), (1,), data, nperseg=2**13)
        _, mir_yx = mutual_information_rate((1,), (0,), data, nperseg=2**13)

        assert mir_xy == pytest.approx(mir_yx, abs=1e-4)

    def test_mir_matches_entropy_decomposition(self):
        """I(X;Y) == h(X) + h(Y) - h(X,Y) for lead-lag model."""
        rng = np.random.default_rng(0)
        T = 5_000_000
        c = 0.7
        data = generate_lead_lag(T, c=c, rng=rng)

        _, mir = mutual_information_rate((0,), (1,), data, nperseg=2**13)
        _, hx = differential_entropy_rate((0,), data, nperseg=2**13)
        _, hy = differential_entropy_rate((1,), data, nperseg=2**13)
        _, hxy = differential_entropy_rate((0, 1), data, nperseg=2**13)

        assert mir == pytest.approx(hx + hy - hxy, abs=1e-4)

    def test_complex_csd_matters(self):
        """
        Regression test for the real-vs-complex CSD bug.

        If the imaginary parts of the CSD are discarded (the old bug),
        the MIR for the lead-lag model is overestimated because
        det(Re(S)) > Re(det(S)) when Im(S_XY) != 0.

        Specifically, for this model:
            |S(w)| = 1  (exact, for all w)
        but
            det(Re(S(w))) = S_XX * S_YY - Re(S_XY)^2
                          = (1+c^2) - c^2*cos^2(w)
                          > 1  for w != 0, pi
        """
        rng = np.random.default_rng(42)
        T = 5_000_000
        c = 1.0
        data = generate_lead_lag(T, c=c, rng=rng)

        _, mir = mutual_information_rate((0,), (1,), data, nperseg=2**13)

        analytic = 0.5 * np.log(1 + c**2)  # = 0.5 * log(2) ≈ 0.347
        assert mir == pytest.approx(analytic, abs=ABS_TOL)
