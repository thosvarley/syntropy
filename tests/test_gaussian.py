import pytest
import pathlib

import numpy as np
import pandas as pd
import scipy.stats as stats

import syntropy.gaussian.shannon as shannon
import syntropy.gaussian.multivariate_mi as mi
from syntropy.gaussian.decompositions import (
    partial_information_decomposition as pid,
    partial_entropy_decomposition as ped,
    generalized_information_decomposition as gid,
    idep_partial_information_decomposition as idep,
    integrated_information_decomposition as phiid,
    local_precompute_sources,
)
from syntropy.gaussian.temporal import (
    differential_entropy_rate,
    mutual_information_rate,
)
from helpers import equicorr_matrix

data_path = pathlib.Path(__file__).parent
data = pd.read_csv(data_path / "../examples/bold.csv", header=None).values

cov = np.cov(data, ddof=0.0)

# Due to natural instability in Scipy's matrix algebra, we need a slightly
# more relaxed tolerance for our unit tests. 1 part in 1,000,000 is probably ok.
pytest_abs = 1e-6


def analytic_tc(N: int, rho: float) -> float:
    """
    TC(N, rho) = -(N-1)/2 * log(1-rho) - 1/2 * log(1 + (N-1)*rho)

    Derived from TC = (1/2)*log(prod_i sigma_i^2 / |Sigma|)
    with sigma_i^2 = 1 and |Sigma| = (1-rho)^{N-1} * (1+(N-1)*rho).
    """
    return -((N - 1) / 2) * np.log(1 - rho) - 0.5 * np.log(1 + (N - 1) * rho)


def analytic_o_information(N: int, rho: float) -> float:
    """
    Omega(N, rho) = (N-2)/2 * log((1+(N-1)*rho) / (1-rho))
                   - N/2 * log(1 + (N-2)*rho)

    Derived from Omega = (2-N)*TC(N,rho) + N*TC(N-1,rho).

    Sign convention: positive = redundancy dominated,
                     negative = synergy dominated.
    """
    if N == 2:
        return 0.0
    term1 = ((N - 2) / 2) * np.log((1 + (N - 1) * rho) / (1 - rho))
    term2 = (N / 2) * np.log(1 + (N - 2) * rho)
    return term1 - term2


def analytic_dtc(N: int, rho: float) -> float:
    """DTC = TC - Omega."""
    return analytic_tc(N, rho) - analytic_o_information(N, rho)


def analytic_s_information(N: int, rho: float) -> float:
    """S = TC + DTC."""
    return analytic_tc(N, rho) + analytic_dtc(N, rho)


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
    for m in range(10):
        for n in range(m):
            i = shannon.mutual_information(idxs_x=(m,), idxs_y=(n,), cov=cov)
            r, _ = stats.pearsonr(data[m, :], data[n, :])

            assert i == pytest.approx(-np.log(1 - (r**2)) / 2, abs=pytest_abs)

            lmi = shannon.local_mutual_information((m,), (n,), data)
            assert i == pytest.approx(lmi.mean(), abs=pytest_abs)

            tc = mi.total_correlation(cov[np.ix_((m, n), (m, n))])
            assert i == pytest.approx(tc, abs=pytest_abs)


class TestTotalCorrelation:
    @pytest.mark.parametrize(
        "N, rho",
        [
            (2, 0.1),
            (2, 0.5),
            (2, 0.9),
            (3, 0.1),
            (3, 0.5),
            (3, 0.9),
            (4, 0.3),
            (4, 0.7),
            (5, 0.5),
        ],
    )
    def test_tc_equicorrelation(self, N, rho):
        cov = equicorr_matrix(N, rho)
        tc_numeric = mi.total_correlation(cov)
        tc_analytic = analytic_tc(N, rho)
        assert tc_numeric == pytest.approx(tc_analytic, abs=ABS_TOL), (
            f"TC failed for N={N}, rho={rho}: "
            f"got {tc_numeric:.8f}, expected {tc_analytic:.8f}"
        )

    def test_tc_independent(self):
        """TC = 0 for identity covariance."""
        for N in (2, 3, 5):
            cov = np.eye(N)
            assert mi.total_correlation(cov) == pytest.approx(0.0, abs=ABS_TOL)

    def test_tc_nonnegative(self):
        """TC is always non-negative."""
        for N in (2, 3, 4):
            for rho in np.linspace(0.0, 0.95, 10):
                cov = equicorr_matrix(N, rho)
                assert mi.total_correlation(cov) >= -ABS_TOL


# ---------------------------------------------------------------------------
# Numerical tests: O-information against analytic formula
# ---------------------------------------------------------------------------


class TestOInformation:
    @pytest.mark.parametrize(
        "N, rho",
        [
            (3, 0.1),
            (3, 0.5),
            (3, 0.9),
            (4, 0.1),
            (4, 0.5),
            (4, 0.9),
            (5, 0.3),
        ],
    )
    def test_o_info_equicorrelation(self, N, rho):
        cov = equicorr_matrix(N, rho)
        o_numeric = mi.o_information(cov)
        o_analytic = analytic_o_information(N, rho)
        assert o_numeric == pytest.approx(o_analytic, abs=ABS_TOL), (
            f"O-info failed for N={N}, rho={rho}: "
            f"got {o_numeric:.8f}, expected {o_analytic:.8f}"
        )

    def test_o_info_independent(self):
        """O-information = 0 for independent variables."""
        for N in (3, 4, 5):
            cov = np.eye(N)
            assert mi.o_information(cov) == pytest.approx(0.0, abs=ABS_TOL)

    def test_o_info_redundant_for_positive_rho(self):
        """Positive equicorrelation gives redundancy-dominated system."""
        for N in (3, 4, 5):
            cov = equicorr_matrix(N, rho=0.5)
            assert mi.o_information(cov) > 0


class TestHigherOrderIdentities:
    @pytest.mark.parametrize(
        "N, rho",
        [
            (3, 0.3),
            (4, 0.5),
            (5, 0.7),
        ],
    )
    def test_all_measures_analytic(self, N, rho):
        """All four measures match their analytic values simultaneously."""
        cov = equicorr_matrix(N, rho)

        tc = mi.total_correlation(cov)
        dtc = mi.dual_total_correlation(cov)
        o = mi.o_information(cov)
        s = mi.s_information(cov)

        assert tc == pytest.approx(analytic_tc(N, rho), abs=ABS_TOL)
        assert dtc == pytest.approx(analytic_dtc(N, rho), abs=ABS_TOL)
        assert o == pytest.approx(analytic_o_information(N, rho), abs=ABS_TOL)
        assert s == pytest.approx(analytic_s_information(N, rho), abs=ABS_TOL)

        # Identities should hold exactly (not just approximately)
        assert tc - dtc == pytest.approx(o, abs=1e-12)
        assert tc + dtc == pytest.approx(s, abs=1e-12)

    def test_mi_equals_tc_for_bivariate(self):
        """For N=2, TC == MI (they are the same quantity)."""
        for rho in (0.1, 0.5, 0.9):
            cov = equicorr_matrix(2, rho)
            tc = mi.total_correlation(cov)
            mi_val = shannon.mutual_information((0,), (1,), cov)
            assert tc == pytest.approx(mi_val, abs=ABS_TOL)

    def test_pearson_mi_identity(self):
        """I(X;Y) = -1/2 * log(1 - rho^2) for bivariate Gaussian."""
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9):
            cov = equicorr_matrix(2, rho)
            mi_numeric = shannon.mutual_information((0,), (1,), cov)
            mi_analytic = -0.5 * np.log(1 - rho**2)
            assert mi_numeric == pytest.approx(mi_analytic, abs=ABS_TOL)


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

    ptw, avg = gid(idxs, data[idxs, :], cov_posterior, cov_prior)

    dkl = shannon.kullback_leibler_divergence(cov_posterior, cov_prior)
    assert sum(avg.values()) == pytest.approx(dkl, abs=pytest_abs)

    ldkl = shannon.local_kullback_leibler_divergence(
        cov_posterior, cov_prior, data[idxs, :]
    )

    assert ldkl.mean() == pytest.approx(sum(avg.values()), abs=pytest_abs)


def test_phiid():
    mi = shannon.mutual_information(idxs_x=(0, 1), idxs_y=(2, 3), cov=cov)
    mi_y2 = shannon.mutual_information(idxs_x=(0, 1), idxs_y=(2,), cov=cov)
    mi_y3 = shannon.mutual_information(idxs_x=(0, 1), idxs_y=(3,), cov=cov)
    avg = phiid(
        inputs=(0, 1), target=(2, 3), data=data, cov=cov, redundancy_function="mmi"
    )
    assert sum(avg.values()) == pytest.approx(mi, abs=pytest_abs)
    assert sum([avg[key] for key in avg.keys() if (0,) in key[1]]) == pytest.approx(
        mi_y2, abs=pytest_abs
    )
    assert sum([avg[key] for key in avg.keys() if (1,) in key[1]]) == pytest.approx(
        mi_y3, abs=pytest_abs
    )

    ptw, avg = phiid(
        inputs=(0, 1), target=(2, 3), data=data, cov=cov, redundancy_function="ipm"
    )
    assert sum(avg.values()) == pytest.approx(mi, abs=pytest_abs)
    assert sum([avg[key] for key in avg.keys() if (0,) in key[1]]) == pytest.approx(
        mi_y2, abs=pytest_abs
    )
    assert sum([avg[key] for key in avg.keys() if (1,) in key[1]]) == pytest.approx(
        mi_y3, abs=pytest_abs
    )

    assert ptw[
        (
            (
                (
                    0,
                    1,
                ),
            ),
            ((0, 1),),
        )
    ].mean() == pytest.approx(
        avg[
            (
                (
                    (
                        0,
                        1,
                    ),
                ),
                ((0, 1),),
            )
        ],
        abs=pytest_abs,
    )

    return None


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
    mi_joint = shannon.mutual_information(
        idxs_x=(0, 1, 2, 3, 4, 5, 6), idxs_y=(7, 8, 9), cov=cov
    )

    assert total == pytest.approx(mi_joint, abs=pytest_abs)


# Absolute tolerance for all analytic comparisons.
# Tighter than before because these tests have known ground truth.
ABS_TOL = 1e-3
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_ar1(
    T: int, a: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Scalar AR(1): X_t = a * X_{t-1} + sigma * eps_t."""
    X = np.zeros(T)
    eps = rng.standard_normal(T) * sigma
    for t in range(1, T):
        X[t] = a * X[t - 1] + eps[t]
    return X[np.newaxis, :]  # shape (1, T)


def generate_lead_lag(T: int, c: float, rng: np.random.Generator) -> np.ndarray:
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

    @pytest.mark.parametrize(
        "a, sigma",
        [
            (0.0, 1.0),  # white noise: rate == marginal entropy
            (0.5, 1.0),  # moderate autocorrelation
            (0.9, 1.0),  # strong autocorrelation
            (0.5, 0.1),  # small innovation noise
            (0.5, 2.0),  # large innovation noise
        ],
    )
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
            f"Lead-lag MIR failed for c={c}: got {mir:.6f}, expected {analytic:.6f}"
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

# ---------------------------------------------------------------------------
# Test: local (pointwise) multivariate measures
#
# cov = np.cov(data, ddof=0.0) is the *exact* empirical covariance of data
# (not an independently-specified ground truth being approximated by a
# finite sample), so local_X(data, cov).mean() should match the static
# X(cov) to floating-point precision -- the same reasoning already used by
# test_differential_entropy/test_mutual_information above -- rather than
# needing a loose Monte Carlo tolerance.
#
# local_delta_k and local_description_complexity are not tested directly:
# local_delta_k is exercised through local_s_information/
# local_dual_total_correlation/local_o_information below (all three are
# thin k=0/1/2 wrappers around it, same reasoning as the static delta_k
# above), and local_description_complexity is just
# local_dual_total_correlation / N.
# ---------------------------------------------------------------------------


class TestLocalMultivariateMeasures:
    idxs = (0, 1, 2, 3)

    def test_local_total_correlation(self):
        ltc = mi.local_total_correlation(data, cov=cov, idxs=self.idxs)
        assert ltc.mean() == pytest.approx(
            mi.total_correlation(cov, self.idxs), abs=pytest_abs
        )

        # tc(x) = sum_i h(x_i) - h(x), reconstructed independently from
        # local_differential_entropy rather than calling the function itself.
        whole = shannon.local_differential_entropy(data, cov, idxs=self.idxs)
        sum_parts = sum(
            shannon.local_differential_entropy(data[i, :]) for i in self.idxs
        )
        assert np.allclose(ltc, sum_parts - whole, atol=1e-9)

    def test_local_s_dtc_o_information(self):
        ltc = mi.local_total_correlation(data, cov=cov, idxs=self.idxs)
        ldtc = mi.local_dual_total_correlation(data, cov=cov, idxs=self.idxs)
        ls = mi.local_s_information(data, cov=cov, idxs=self.idxs)
        lo = mi.local_o_information(data, cov=cov, idxs=self.idxs)

        assert ls.mean() == pytest.approx(mi.s_information(cov, self.idxs), abs=pytest_abs)
        assert ldtc.mean() == pytest.approx(
            mi.dual_total_correlation(cov, self.idxs), abs=pytest_abs
        )
        assert lo.mean() == pytest.approx(mi.o_information(cov, self.idxs), abs=pytest_abs)

        # Pointwise identities, exact given the same deterministic data.
        assert np.allclose(ls, ltc + ldtc, atol=1e-9)
        assert np.allclose(lo, ltc - ldtc, atol=1e-9)

    def test_local_co_information(self):
        lco = mi.local_co_information(data, cov=cov, idxs=self.idxs)
        assert lco.mean() == pytest.approx(
            mi.co_information(cov, self.idxs), abs=pytest_abs
        )

        # co(x) = sum over nonempty subsets xi of (-1)^|xi| * h(xi),
        # reconstructed independently from local_precompute_sources.
        sources = local_precompute_sources(
            data=data[list(self.idxs), :], cov=cov[np.ix_(self.idxs, self.idxs)]
        )
        manual = np.zeros((1, data.shape[1]))
        for source, values in sources.items():
            sign = (-1) ** len(source)
            manual -= sign * values

        assert np.allclose(lco, manual, atol=1e-9)


class TestLocalConditionalMeasures:
    idxs_x, idxs_y, idxs_z = (0,), (1,), (2, 3)

    def test_local_conditional_entropy(self):
        joint = self.idxs_x + self.idxs_y
        lce = shannon.local_conditional_entropy(self.idxs_x, self.idxs_y, data, cov=cov)

        assert lce.mean() == pytest.approx(
            shannon.conditional_entropy(self.idxs_x, self.idxs_y, cov), abs=pytest_abs
        )

        manual = shannon.local_differential_entropy(
            data[joint, :], cov[np.ix_(joint, joint)]
        ) - shannon.local_differential_entropy(
            data[self.idxs_y, :], cov[np.ix_(self.idxs_y, self.idxs_y)]
        )
        assert np.allclose(lce, manual, atol=1e-9)

    def test_local_conditional_mutual_information(self):
        lcmi = shannon.local_conditional_mutual_information(
            self.idxs_x, self.idxs_y, self.idxs_z, data, cov=cov
        )
        assert lcmi.mean() == pytest.approx(
            shannon.conditional_mutual_information(
                self.idxs_x, self.idxs_y, self.idxs_z, cov
            ),
            abs=pytest_abs,
        )

        # i(x;y|z) = i(x;y,z) - i(x;z), cross-checked against
        # local_mutual_information -- a different, independently-tested code
        # path than the one local_conditional_mutual_information itself uses
        # (local_conditional_entropy).
        lmi_x_yz = shannon.local_mutual_information(
            self.idxs_x, self.idxs_y + self.idxs_z, data, cov=cov
        )
        lmi_x_z = shannon.local_mutual_information(self.idxs_x, self.idxs_z, data, cov=cov)
        assert np.allclose(lcmi, lmi_x_yz - lmi_x_z, atol=1e-9)
