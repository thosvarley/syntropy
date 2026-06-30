import pytest
import numpy as np

from syntropy.mixed import (
    mutual_information,
    conditional_entropy,
    shannon_entropy,
)

# Differential entropy of a unit-variance 1-D Gaussian: 0.5 * ln(2*pi*e).
LN_2PI_E = np.log(2 * np.pi * np.e)
# Discrete (Shannon) entropy of a balanced binary variable, in nats.
H_BINARY = np.log(2)

# The mixed estimator is a Gaussian plug-in:
#   I(D; C) = H_Gauss(pooled C) - sum_d p(d) * H_Gauss(C | D=d)
# This gives exact closed forms for the constructions below. The comparisons
# are against finite-sample estimates, so we use a large N and a loose
# tolerance.
SEED = 0
N = 500_000
ABS_TOL = 5e-3


# ---------------------------------------------------------------------------
# Helpers: binary discrete D (p = 0.5), one-dimensional Gaussian C.
# ---------------------------------------------------------------------------
def make_mean_shift(mu, sigma, rng, n=N):
    """C|D=0 ~ N(-mu, sigma^2), C|D=1 ~ N(+mu, sigma^2)."""
    d = rng.integers(0, 2, size=n)
    c = np.where(d == 1, mu, -mu) + sigma * rng.standard_normal(n)
    return d[None, :], c[None, :]


def make_var_diff(s0, s1, rng, n=N):
    """C|D=0 ~ N(0, s0^2), C|D=1 ~ N(0, s1^2)."""
    d = rng.integers(0, 2, size=n)
    c = np.where(d == 1, s1, s0) * rng.standard_normal(n)
    return d[None, :], c[None, :]


# ---------------------------------------------------------------------------
# Analytic ground truths
# ---------------------------------------------------------------------------
class TestMixedMutualInformationAnalytic:
    @pytest.mark.parametrize(
        "mu, sigma",
        [(1.0, 1.0), (2.0, 1.0), (0.5, 2.0), (1.5, 0.8)],
    )
    def test_mean_shift(self, mu, sigma):
        """
        Equal-variance, mean-shifted classes:
            I = 1/2 * ln(1 + mu^2 / sigma^2)
        """
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(mu, sigma, rng)
        _, mi = mutual_information(discrete_vars=d, continuous_vars=c)

        analytic = 0.5 * np.log(1 + mu**2 / sigma**2)
        assert mi == pytest.approx(analytic, abs=ABS_TOL), (
            f"mean-shift failed for mu={mu}, sigma={sigma}: "
            f"got {mi:.6f}, expected {analytic:.6f}"
        )

    @pytest.mark.parametrize(
        "s0, s1",
        [(1.0, 2.0), (1.0, 3.0), (0.5, 2.0)],
    )
    def test_variance_difference(self, s0, s1):
        """
        Zero-mean, variance-differing classes:
            I = 1/2 * ln((s0^2 + s1^2) / (2 * s0 * s1))
        """
        rng = np.random.default_rng(SEED)
        d, c = make_var_diff(s0, s1, rng)
        _, mi = mutual_information(discrete_vars=d, continuous_vars=c)

        analytic = 0.5 * np.log((s0**2 + s1**2) / (2 * s0 * s1))
        assert mi == pytest.approx(analytic, abs=ABS_TOL), (
            f"var-diff failed for s0={s0}, s1={s1}: "
            f"got {mi:.6f}, expected {analytic:.6f}"
        )

    def test_independence(self):
        """C drawn independently of D gives I = 0."""
        rng = np.random.default_rng(SEED)
        d = rng.integers(0, 3, size=N)
        c = rng.standard_normal(N)
        _, mi = mutual_information(discrete_vars=d[None, :], continuous_vars=c[None, :])
        assert mi == pytest.approx(0.0, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Property / invariant tests (no analytic value required)
# ---------------------------------------------------------------------------
def test_local_average_identity():
    """The mean of the pointwise values equals the expected value."""
    rng = np.random.default_rng(SEED)
    d, c = make_mean_shift(1.0, 1.0, rng)
    ptw, mi = mutual_information(discrete_vars=d, continuous_vars=c)
    assert ptw.mean() == pytest.approx(mi, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Joint entropy H(D, C) = H(D) + H(C | D)
# ---------------------------------------------------------------------------
class TestMixedJointEntropy:
    @pytest.mark.parametrize("mu, sigma", [(1.0, 1.0), (2.0, 1.0), (0.5, 2.0)])
    def test_mean_shift(self, mu, sigma):
        """H(D, C) = ln 2 + 1/2 * ln(2*pi*e * sigma^2)."""
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(mu, sigma, rng)
        _, h = shannon_entropy(discrete_vars=d, continuous_vars=c)
        analytic = H_BINARY + 0.5 * (LN_2PI_E + np.log(sigma**2))
        assert h == pytest.approx(analytic, abs=ABS_TOL)

    @pytest.mark.parametrize("s0, s1", [(1.0, 2.0), (0.5, 2.0)])
    def test_variance_difference(self, s0, s1):
        """H(D, C) = ln 2 + 1/2 * ln(2*pi*e * s0 * s1)."""
        rng = np.random.default_rng(SEED)
        d, c = make_var_diff(s0, s1, rng)
        _, h = shannon_entropy(discrete_vars=d, continuous_vars=c)
        analytic = H_BINARY + 0.5 * LN_2PI_E + 0.5 * np.log(s0 * s1)
        assert h == pytest.approx(analytic, abs=ABS_TOL)

    def test_local_average_identity(self):
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.0, 1.0, rng)
        ptw, h = shannon_entropy(discrete_vars=d, continuous_vars=c)
        assert ptw.mean() == pytest.approx(h, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Conditional entropy of the continuous variable given the discrete: H(C | D)
# ---------------------------------------------------------------------------
class TestContinuousGivenDiscrete:
    @pytest.mark.parametrize("mu", [0.0, 1.0, 3.0])
    def test_mean_shift_is_mu_invariant(self, mu):
        """
        H(C | D) = 1/2 * ln(2*pi*e * sigma^2), independent of the mean shift mu
        (shifting class means does not change the within-class spread).
        """
        rng = np.random.default_rng(SEED)
        sigma = 1.0
        d, c = make_mean_shift(mu, sigma, rng)
        _, h = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="discrete"
        )
        analytic = 0.5 * (LN_2PI_E + np.log(sigma**2))
        assert h == pytest.approx(analytic, abs=ABS_TOL)

    @pytest.mark.parametrize("s0, s1", [(1.0, 2.0), (0.5, 2.0)])
    def test_variance_difference(self, s0, s1):
        """H(C | D) = 1/2 * ln(2*pi*e) + 1/2 * ln(s0 * s1) (geometric mean)."""
        rng = np.random.default_rng(SEED)
        d, c = make_var_diff(s0, s1, rng)
        _, h = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="discrete"
        )
        analytic = 0.5 * LN_2PI_E + 0.5 * np.log(s0 * s1)
        assert h == pytest.approx(analytic, abs=ABS_TOL)

    def test_local_average_identity(self):
        rng = np.random.default_rng(SEED)
        d, c = make_var_diff(1.0, 2.0, rng)
        ptw, h = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="discrete"
        )
        assert ptw.mean() == pytest.approx(h, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Conditional entropy of the discrete variable given the continuous: H(D | C)
# ---------------------------------------------------------------------------
class TestDiscreteGivenContinuous:
    @pytest.mark.parametrize("mu, sigma", [(1.0, 1.0), (2.0, 1.0)])
    def test_mean_shift(self, mu, sigma):
        """H(D | C) = H(D) - I(D; C) = ln 2 - 1/2 * ln(1 + mu^2 / sigma^2)."""
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(mu, sigma, rng)
        _, h = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="continuous"
        )
        analytic = H_BINARY - 0.5 * np.log(1 + mu**2 / sigma**2)
        assert h == pytest.approx(analytic, abs=ABS_TOL)

    def test_local_average_identity(self):
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.0, 1.0, rng)
        ptw, h = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="continuous"
        )
        assert ptw.mean() == pytest.approx(h, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Cross-consistency identities tying the four estimators together
# ---------------------------------------------------------------------------
class TestConsistencyIdentities:
    def test_chain_rule(self):
        """H(D, C) == H(D) + H(C | D)."""
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.2, 0.9, rng)
        _, joint = shannon_entropy(discrete_vars=d, continuous_vars=c)
        _, h_c_given_d = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="discrete"
        )
        assert joint == pytest.approx(H_BINARY + h_c_given_d, abs=ABS_TOL)

    def test_mutual_information_identity(self):
        """I(D; C) == H(D) - H(D | C)."""
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.2, 0.9, rng)
        _, mi = mutual_information(discrete_vars=d, continuous_vars=c)
        _, h_d_given_c = conditional_entropy(
            discrete_vars=d, continuous_vars=c, conditional="continuous"
        )
        assert mi == pytest.approx(H_BINARY - h_d_given_c, abs=ABS_TOL)


# ---------------------------------------------------------------------------
# Cross-estimator agreement: gaussian vs KNN (continuous_estimator flag)
# ---------------------------------------------------------------------------
class TestGaussianKNNAgreement:
    """
    For Gaussian continuous data, the Gaussian plug-in and the KNN
    (Kozachenko-Leonenko) estimator estimate the *same* per-class differential
    entropy, so the two ``continuous_estimator`` options should agree.

    We test only the within-class quantities -- the joint entropy H(D, C) and
    H(C | D). H(D | C) is deliberately excluded: it depends on the pooled
    marginal H(C), where the pooled data is a Gaussian *mixture*. There the
    Gaussian plug-in (which fits a single Gaussian) and KNN (which sees the true
    mixture) estimate genuinely different quantities and need not agree.
    """

    KNN_N = 50_000
    KNN_TOL = 2e-2  # KNN entropy estimates are noisier; worst observed gap ~7e-3.

    def test_joint_entropy_agreement(self):
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.0, 1.0, rng, n=self.KNN_N)
        _, h_gauss = shannon_entropy(
            discrete_vars=d, continuous_vars=c, continuous_estimator="gaussian"
        )
        _, h_knn = shannon_entropy(
            discrete_vars=d, continuous_vars=c, continuous_estimator="knn", k=5
        )
        assert h_gauss == pytest.approx(h_knn, abs=self.KNN_TOL)

    @pytest.mark.parametrize(
        "builder, args",
        [
            (make_mean_shift, (1.0, 1.0)),
            (make_var_diff, (1.0, 2.0)),
        ],
    )
    def test_conditional_entropy_agreement(self, builder, args):
        """H(C | D) is a within-class quantity, so gaussian and KNN agree."""
        rng = np.random.default_rng(SEED)
        d, c = builder(*args, rng, n=self.KNN_N)
        _, h_gauss = conditional_entropy(
            discrete_vars=d,
            continuous_vars=c,
            conditional="discrete",
            continuous_estimator="gaussian",
        )
        _, h_knn = conditional_entropy(
            discrete_vars=d,
            continuous_vars=c,
            conditional="discrete",
            continuous_estimator="knn",
            k=5,
        )
        assert h_gauss == pytest.approx(h_knn, abs=self.KNN_TOL)


# ---------------------------------------------------------------------------
# KNN-based mutual information (continuous_estimator="knn")
# ---------------------------------------------------------------------------
class TestKNNMutualInformation:
    """
    Unlike the Gaussian plug-in, the KNN estimator recovers the *true* mutual
    information rather than a single-Gaussian approximation of the pooled
    marginal. These tests therefore use true-MI ground truths.
    """

    KNN_N = 50_000
    KNN_TOL = 2e-2

    def test_saturation_at_h_of_d(self):
        """
        For well-separated classes, C determines D perfectly, so the true
        mutual information saturates at H(D) = ln 2 (a balanced binary D).
        (The Gaussian plug-in over-shoots this, even exceeding ln 2.)
        """
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(5.0, 1.0, rng, n=self.KNN_N)
        _, mi = mutual_information(
            discrete_vars=d, continuous_vars=c, continuous_estimator="knn", k=5
        )
        assert mi == pytest.approx(H_BINARY, abs=self.KNN_TOL)

    def test_independence(self):
        """C drawn independently of D gives I = 0."""
        rng = np.random.default_rng(SEED)
        d = rng.integers(0, 2, size=self.KNN_N)
        c = rng.standard_normal(self.KNN_N)
        _, mi = mutual_information(
            discrete_vars=d[None, :],
            continuous_vars=c[None, :],
            continuous_estimator="knn",
            k=5,
        )
        assert mi == pytest.approx(0.0, abs=self.KNN_TOL)

    def test_local_average_identity(self):
        """The mean of the pointwise values equals the expected value."""
        rng = np.random.default_rng(SEED)
        d, c = make_mean_shift(1.0, 1.0, rng, n=self.KNN_N)
        ptw, mi = mutual_information(
            discrete_vars=d, continuous_vars=c, continuous_estimator="knn", k=5
        )
        assert ptw.mean() == pytest.approx(mi, abs=self.KNN_TOL)


# ---------------------------------------------------------------------------
# Shape smoke tests (regression guards for univariate vs multivariate C)
# ---------------------------------------------------------------------------
def test_univariate_continuous_runs():
    """A single continuous variable must not crash (regression guard)."""
    rng = np.random.default_rng(SEED)
    n = 10_000
    d = rng.integers(0, 2, size=n)[None, :]
    c = rng.standard_normal(n)[None, :]
    ptw, mi = mutual_information(discrete_vars=d, continuous_vars=c)
    assert np.isfinite(mi)
    assert ptw.shape == (1, n)


def test_multivariate_continuous_runs():
    """Multiple continuous variables must run and give finite output."""
    rng = np.random.default_rng(SEED)
    n = 10_000
    x = rng.standard_normal(n)
    d = (x > 0).astype(int)[None, :]
    c = np.vstack([x, 0.3 * x + rng.standard_normal(n)])
    ptw, mi = mutual_information(discrete_vars=d, continuous_vars=c)
    assert np.isfinite(mi)
    assert ptw.shape[1] == n
