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

    ptw, avg = ped(idxs, data, cov)
    h = shannon.differential_entropy(cov, idxs)

    assert sum(avg.values()) == pytest.approx(h, abs=pytest_abs)


def test_gid():
    idxs = (0, 1, 2, 3)
    cov_prior = np.eye(len(idxs))
    cov_posterior = cov[np.ix_(idxs, idxs)]

    ptw, avg = gid(idxs, data, cov_posterior, cov_prior)

    dkl = shannon.kullback_leibler_divergence(cov_posterior, cov_prior)
    assert sum(avg.values()) == pytest.approx(dkl, abs=pytest_abs)

    ldkl = shannon.local_kullback_leibler_divergence(
        cov_posterior, cov_prior, data[idxs, :]
    )

    assert ldkl.mean() == pytest.approx(sum(avg.values()), abs=pytest_abs)


def test_gaussian_rate():
    T = 5_000_000

    noise = np.random.randn(2, T)

    _, mir = mutual_information_rate((0,), (1,), noise, nperseg=2**13)

    _, h1 = differential_entropy_rate((0,), noise, nperseg=2**13)
    _, h2 = differential_entropy_rate((1,), noise, nperseg=2**13)
    _, h12 = differential_entropy_rate((0, 1), noise, nperseg=2**13)

    assert mir == pytest.approx(h1 + h2 - h12, abs=pytest_abs)

    _, tcr = total_correlation_rate((0, 1), noise, nperseg=2**13)

    assert mir == pytest.approx(tcr, abs=pytest_abs)

    # for white noise with 0 mean and unit variance,
    # the analytic entropy rate should be (1/2)*log(2*pi*e)
    analytic = (1 / 2) * np.log(2.0 * np.pi * np.e)

    assert analytic == pytest.approx(h1, abs=1e-3)

    # %% A VAR(1) model with fixed noise and coefficient.

    X = np.zeros((1, T))
    eps = 0.01
    noise = np.random.randn(T) * eps
    a = 0.2

    for t in range(1, T):
        X[0, t] = X[0, t - 1] * a + noise[t - 1]

    _, hX = differential_entropy_rate((0,), X, nperseg=2**14)

    analytic = 0.5 * (np.log(2 * np.pi * np.e * (eps**2))) - (0.5 * np.log(1 - (a**2)))

    assert np.isclose(hX, analytic, rtol=1e-2)


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
