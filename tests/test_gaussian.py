#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:16:01 2025

@author: thosvarley
"""

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
)
from syntropy.gaussian.temporal import (
    entropy_rate,
    mutual_information_rate,
    total_correlation_rate,
)

data_path = pathlib.Path(__file__).parent
data = pd.read_csv(data_path / "../examples/bold.csv", header=None).values

cov = np.cov(data, ddof=0.0)

# Due to natural instability in Scipy's matrix algebra, we need a slightly
# more relaxed tolerance for our unit tests. 1 part in 1,000,000 is probably ok.
pytest_abs = 1e-6


def test_differential_entropy():

    h = shannon.local_differential_entropy(data[1, :]).mean()
    assert shannon.differential_entropy(cov[1, 1], (1,)) == pytest.approx(
        h, abs=pytest_abs
    )

    h1 = shannon.local_differential_entropy(
        data[(1, 2), :], cov[(1, 2), :][:, (1, 2)]
    ).mean()
    h2 = shannon.local_differential_entropy(
        data[(1, 2), :],
    ).mean()
    assert h1 == pytest.approx(h2, abs=pytest_abs)

    assert h1 == pytest.approx(
        shannon.differential_entropy(cov, (1, 2)), abs=pytest_abs
    )

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

    inputs = (0, 1, 2, 3)

    tc = mi.total_correlation(cov, inputs)
    dtc = mi.dual_total_correlation(cov, inputs)
    o = mi.o_information(cov, inputs)
    s = mi.s_information(cov, inputs)

    assert tc - dtc == pytest.approx(o, abs=pytest_abs)
    assert tc + dtc == pytest.approx(s, abs=pytest_abs)

    ltc = mi.local_total_correlation(data, cov, inputs)
    ldtc = mi.local_dual_total_correlation(data, cov, inputs)
    lo = mi.local_o_information(data, cov, inputs)
    ls = mi.local_s_information(data, cov, inputs)

    assert ltc.mean() == pytest.approx(tc, abs=pytest_abs)
    assert ldtc.mean() == pytest.approx(dtc, abs=pytest_abs)
    assert lo.mean() == pytest.approx(o, abs=pytest_abs)
    assert ls.mean() == pytest.approx(s, abs=pytest_abs)

    c = mi.description_complexity(cov, inputs)
    assert c == pytest.approx(dtc / len(inputs), abs=pytest_abs)

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
            triad, inputs=(i,)
        ) - shannon.differential_entropy(triad, inputs=residuals)

    assert whole + sum_diffs == pytest.approx(mi.o_information(triad), abs=pytest_abs)


def test_pid():

    inputs = (0, 1)
    target = (3, 4)
    ptw, avg = pid(inputs, target, data, cov)
    mi = shannon.mutual_information(inputs, target, cov)

    assert sum(avg.values()) == pytest.approx(mi, abs=pytest_abs)

    mi = shannon.mutual_information((0,), target, cov)
    assert avg[((0,),)] + avg[((0,), (1,))] == pytest.approx(mi, abs=pytest_abs)


def test_ped():

    inputs = (0, 1, 2, 3)

    ptw, avg = ped(inputs, data, cov)
    h = shannon.differential_entropy(cov, inputs)

    assert sum(avg.values()) == pytest.approx(h, abs=pytest_abs)


def test_gid():

    inputs = (0, 1, 2, 3)
    cov_prior = np.eye(len(inputs))
    cov_posterior = cov[inputs, :][:, inputs]

    ptw, avg = gid(inputs, data, cov_posterior, cov_prior)

    dkl = shannon.kullback_leibler_divergence(cov_posterior, cov_prior)
    assert sum(avg.values()) == pytest.approx(dkl, abs=pytest_abs)

    ldkl = shannon.local_kullback_leibler_divergence(
        cov_posterior, cov_prior, data[inputs, :]
    )

    assert ldkl.mean() == pytest.approx(sum(avg.values()), abs=pytest_abs)


def test_gaussian_rate():

    T = 5_000_000

    noise = np.random.randn(2, T)

    _, mir = mutual_information_rate((0,), (1,), noise, nperseg=2**13, overlap=2**12)

    _, h1 = entropy_rate((0,), noise, nperseg=2**13, overlap=2**12)
    _, h2 = entropy_rate((1,), noise, nperseg=2**13, overlap=2**12)
    _, h12 = entropy_rate((0, 1), noise, nperseg=2**13, overlap=2**12)

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

    _, hX = entropy_rate((0,), X, nperseg=2**14, overlap=2**13)

    analytic = 0.5 * (np.log(2 * np.pi * np.e * (eps**2))) - (0.5 * np.log(1 - (a**2)))

    assert np.isclose(hX, analytic, rtol=1e-2)
