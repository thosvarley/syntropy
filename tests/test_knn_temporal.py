import pytest
import numpy as np
from scipy.special import erf
from scipy.stats import zscore
from numpy.lib.stride_tricks import sliding_window_view

from syntropy.knn.temporal import sample_entropy, cross_sample_entropy

# For an i.i.d. process, sample entropy converges to -ln(p1), where
# p1 = P(|X1 - X2| <= r) is the probability that two independent samples fall
# within the tolerance r. This limit is independent of the embedding dimension m.
# The estimate has a finite-N positive bias, so we use a large N and a loose
# tolerance; the worst error observed over several seeds at this N is ~6e-3.
SEED = 0
N = 50_000
ABS_TOL = 2e-3
PYTEST_ABS = 1e-6


class TestSampleEntropyClosedForm:

    @pytest.mark.parametrize("m", [2, 3])
    def test_iid_gaussian(self, m):
        """
        IID Gaussian: X1 - X2 ~ N(0, 2*sigma^2), so
            p1 = erf(r / (2*sigma))  and  SampEn -> -ln(erf(r / (2*sigma))).
        With r normalized as r = 0.2 * sigma, the sigma cancels and the target
        is -ln(erf(r / 2)), independent of both sigma and m.
        """
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal(N)
        r = 0.2

        est = sample_entropy((0,), x[None, :], m=m, r=r, normalize_r=True)
        analytic = -np.log(erf(r / 2))

        assert est == pytest.approx(analytic, abs=ABS_TOL)

    def test_iid_uniform(self):
        """
        IID Uniform: the difference of two uniforms is triangular, so
            p1 = 2*r_abs - r_abs^2  and  SampEn -> -ln(2*r_abs - r_abs^2),
        where r_abs = r * std(x) is the absolute tolerance the estimator uses.
        """
        rng = np.random.default_rng(SEED)
        x = rng.random(N)
        r = 0.2
        r_abs = r * np.std(x)

        est = sample_entropy((0,), x[None, :], m=2, r=r, normalize_r=True)
        analytic = -np.log(2 * r_abs - r_abs**2)

        assert est == pytest.approx(analytic, abs=ABS_TOL)

def test_antropy_matches():
    # This one compares Syntropy to Antropy 
    # https://raphaelvallat.com/antropy/ 
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((5,10_000))
    
    sampens = [sample_entropy((i,), noise) for i in range(5)]
    assert sampens[0] == pytest.approx(2.1960718154968815, abs=PYTEST_ABS)
    assert sampens[1] == pytest.approx(2.183291739784684, abs=PYTEST_ABS)
    assert sampens[2] == pytest.approx(2.183114546350932, abs=PYTEST_ABS)
    assert sampens[3] == pytest.approx(2.1764155747815264, abs=PYTEST_ABS)
    assert sampens[4] == pytest.approx(2.1901149239999116, abs=PYTEST_ABS)
    

    brownian = np.cumsum(noise, axis=-1)
    sampens = [sample_entropy((i,), brownian) for i in range(5)]

    assert sampens[0] == pytest.approx(0.04959120707043105, abs=PYTEST_ABS)
    assert sampens[1] == pytest.approx(0.061431721321519785, abs=PYTEST_ABS)
    assert sampens[2] == pytest.approx(0.07189084355966345, abs=PYTEST_ABS)
    assert sampens[3] == pytest.approx(0.08464506385060633, abs=PYTEST_ABS)
    assert sampens[4] == pytest.approx(0.0437523057152039, abs=PYTEST_ABS)


class TestCrossSampleEntropy:
    """Cross-sample entropy measures dynamical similarity between two series.
    Both series are z-scored internally, so the measure is symmetric."""

    def test_matches_brute_force(self):
        """Cross-counts match an explicit O(N^2) double loop over the two series'
        templates. Unlike SampEn, there is no self-match exclusion -- the
        templates come from different series."""
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((2, 250))
        m, r = 2, 0.2

        est = cross_sample_entropy((0,), (1,), data, m=m, r=r)

        n = data.shape[1] - m
        sx, sy = zscore(data[0]), zscore(data[1])
        exs, eys = sliding_window_view(sx, m)[:n], sliding_window_view(sy, m)[:n]
        exb, eyb = sliding_window_view(sx, m + 1), sliding_window_view(sy, m + 1)

        def cross_count(EX, EY):
            return sum(
                1
                for i in range(len(EX))
                for j in range(len(EY))
                if np.max(np.abs(EX[i] - EY[j])) <= r
            )

        B, A = cross_count(exs, eys), cross_count(exb, eyb)
        assert est == pytest.approx(-np.log(A / B), abs=PYTEST_ABS)

    def test_symmetric(self):
        """Because both series are z-scored and count_neighbors is symmetric for a
        fixed r, cross(X, Y) == cross(Y, X) exactly."""
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((2, 5_000))
        assert cross_sample_entropy((0,), (1,), data) == pytest.approx(
            cross_sample_entropy((1,), (0,), data), abs=1e-12
        )

    def test_reduces_to_sample_entropy(self):
        """cross_sample_entropy(X, X) recovers ordinary sample entropy, sitting
        slightly below it: the i==j self-matches that SampEn excludes are retained
        in the cross count, nudging A/B toward 1 (observed offset ~0.012 at this N)."""
        rng = np.random.default_rng(SEED)
        x = rng.standard_normal((1, N))

        cross_xx = cross_sample_entropy((0,), (0,), x)
        sampen = sample_entropy((0,), x)

        assert cross_xx < sampen  # strict, by self-match inclusion
        assert cross_xx == pytest.approx(sampen, abs=2e-2)

    def test_similar_dynamics_lower_than_dissimilar(self):
        """Low for signals with similar dynamics, high for dissimilar ones (it
        measures dynamical similarity, not coupling). Two independent random walks
        (both smooth) score far lower than a random walk vs. white noise."""
        rng = np.random.default_rng(SEED)
        n = 20_000
        smooth_1 = np.cumsum(rng.standard_normal(n))
        smooth_2 = np.cumsum(rng.standard_normal(n))
        noise = rng.standard_normal(n)

        similar = np.vstack([smooth_1, smooth_2])
        dissimilar = np.vstack([smooth_1, noise])

        assert cross_sample_entropy((0,), (1,), similar) < cross_sample_entropy(
            (0,), (1,), dissimilar
        )
