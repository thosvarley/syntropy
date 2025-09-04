#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:11:57 2025

@author: thosvarley
"""

# %% Monster list of imports
import numpy as np
from syntropy.discrete.shannon import (
    shannon_entropy,
    mutual_information as discrete_mutual_information,
    conditional_entropy as discrete_conditional_entropy,
    conditional_mutual_information as discrete_conditional_mutual_information,
)
from syntropy.discrete.multivariate_mi import (
    total_correlation as discrete_total_correlation,
    dual_total_correlation as discrete_dual_total_correlation,
    o_information as discrete_o_information,
    s_information as discrete_s_information,
    description_complexity as discrete_description_complexity,
    tse_complexity as discrete_tse_complexity,
)

from syntropy.gaussian.shannon import (
    differential_entropy,
    conditional_entropy as differential_conditional_entropy,
    mutual_information as differential_mutual_information,
    conditional_mutual_information as differential_conditional_mutual_information,
    #
    local_differential_entropy,
    local_conditional_entropy as local_differential_conditional_entropy,
    local_mutual_information as local_differential_mutual_information,
    local_conditional_mutual_information as local_differential_conditional_mutual_information,
)

from syntropy.gaussian.multivariate_mi import (
    total_correlation as differential_total_correlation,
    dual_total_correlation as differential_dual_total_correlation,
    o_information as differential_o_information,
    s_information as differential_s_information,
    description_complexity as differential_description_complexity,
    tse_complexity as differential_tse_complexity,
    #
    local_total_correlation as local_differential_total_correlation,
    local_dual_total_correlation as local_differential_dual_total_correlation,
    local_o_information as local_differential_o_information,
    local_s_information as local_differential_s_information,
    local_description_complexity as local_differential_description_complexity,
)

from syntropy.gaussian.shannon import H_SINGLE

# %%


class Distribution:
    """
    A class that defines a discrete or Gaussian probability distribution.
    Inspired by the Networkx Graph() class and the DIT Distribution() class.

    Meant to make doing high-level analyses easier.
    """

    def __init__(self, data):

        if type(data) == np.ndarray:

            assert False not in np.isclose(
                data.mean(axis=-1), 0.0, atol=1e-15
            ), "The data must have mean of 0."
            assert False not in np.isclose(
                data.std(axis=-1), 1.0, atol=1e-15
            ), "The data must have standard deviation of 1"

            self.cov: np.ndaray = np.cov(data, ddof=0.0)
            self.estimator: str = "gaussian"
            self.unit: str = "nat"
            self.N: int = self.cov.shape[0]

        elif type(data) == dict:
            self.estimator: str = "discrete"
            self.unit: str = "bit"
            self.N: int = len(list(data.keys())[0])

        self.data = data

        self.entropy: float = np.nan
        self.local_entropy: dict[tuple, float] | np.ndarray = None

        self.total_correlation: float = np.nan
        self.local_total_correlation: dict[tuple, float] = None

        self.dual_total_correlation: float = np.nan
        self.local_dual_total_correlation: dict[tuple, float] = None

        self.o_information: float = np.nan
        self.local_o_information: dict[tuple, float] = None

        self.s_information: float = np.nan
        self.local_s_information: dict[tuple, float] = None

        self.description_complexity = np.nan
        self.local_description_complexity: dict[tuple, float] = None

        self.tse_complexity = np.nan

        self.mutual_information_matrix: np.ndarray = np.array([-1])

    def estimate_entropy(self, inputs: tuple = (-1,), return_locals: bool = False):

        if self.estimator == "discrete":
            ptw, avg = shannon_entropy(self.data)
            self.entropy = avg
            self.local_entropy = ptw
            if return_locals is True:
                return ptw, avg
            else:
                return avg

        elif self.estimator == "gaussian":
            if inputs[0] == -1:
                avg = differential_entropy(self.cov)
                self.entropy = avg
                if return_locals is True:
                    ptw = local_differential_entropy(self.data, self.cov)
                    self.local_entropy = ptw
                    return ptw, avg
                return avg
            else:
                avg = differential_entropy(self.cov[inputs, :][:, inputs])
                if return_locals is True:
                    ptw = local_differential_entropy(
                        self.data[inputs, :], self.cov[inputs, :][:, inputs]
                    )
                    return ptw, avg
                return avg
    
    def estimate_total_correlation(self, inputs: tuple = (-1,), return_locals: bool = False):

        if self.estimator == "discrete":
            ptw, avg = discrete_total_correlation(self.data)
            self.total_correlation = avg
            self.local_total_correlation = ptw
            if return_locals is True:
                return ptw, avg
            else:
                return avg

        elif self.estimator == "gaussian":
            if inputs[0] == -1:
                avg = differential_total_correlation(self.cov)
                self.total_correlation = avg
                if return_locals is True:
                    ptw = local_differential_total_correlation(self.data, self.cov)
                    self.local_total_correlation = ptw
                    return ptw, avg
                return avg
            else:
                avg = differential_total_correlation(self.cov, inputs=inputs)
                if return_locals is True:
                    ptw = local_differential_total_correlation(self.data, inputs=inputs)
                    return ptw, avg
                return avg
