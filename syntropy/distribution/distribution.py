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
            self.estimator: str = ("discrete",)
            self.unit: str = "bit"
            self.N: int = len(list(data.keys())[0])

        self.data = data
        self.entropy: float = np.nan
        self.total_correlation: float = np.nan
        self.dual_total_correlation: float = np.nan
        self.o_information: float = np.nan
        self.s_information: float = np.nan
        self.description_complexity = np.nan
        self.tse_complexity = np.nan

        self.mutual_information_matrix: np.ndarray = np.array([-1])

    def estimate_entropy(
        self, inputs: tuple = (-1,), return_locals: bool = False
    ) -> float:

        if self.estimator == "gaussian":
            h = differential_entropy(self.cov, inputs=inputs)
            if return_locals == True:
                if inputs == (-1,):
                    h_ptw = local_differential_entropy(self.data, self.cov)
                else:
                    h_ptw = local_differential_entropy(
                        self.data[inputs, :], self.cov[inputs, :][:, inputs]
                    )

        elif self.estimator == "discrete":
            h_ptw, h = shannon_entropy(self.data)

        if inputs == (-1,):
            self.entropy = h

        if return_locals == True:
            return h, h_ptw
        else:
            return h

    def estimate_conditional_entropy(
        self, inputs_x: tuple, inputs_y: tuple, return_locals=False
    ):

        if self.estimator == "gaussian":
            h = differential_conditional_entropy(inputs_x, inputs_y, self.cov)
            if return_locals == True:
                h_ptw = local_differential_conditional_entropy(
                    inputs_x, inputs_y, self.data, self.cov
                )
        elif self.estimator == "discrete":
            h_ptw, h = discrete_conditional_entropy(inputs_x, inputs_y, self.data)

        if return_locals == True:
            return h, h_ptw
        else:
            return h

    def estimate_mutual_information(
        self, inputs: tuple, target: tuple, return_locals: bool = False
    ) -> float:

        if self.estimator == "gaussian":
            mi = differential_mutual_information(inputs, target, self.cov)
            if return_locals == True:
                mi_ptw = local_differential_mutual_information(
                    inputs, target, self.data
                )
        elif self.estimator == "discrete":
            mi_ptw, mi = discrete_mutual_information(inputs, target, self.data)

        if return_locals == True:
            return mi, mi_ptw
        else:
            return mi

    def estimate_mutual_information_matrix(self):

        if self.mutual_information_matrix[0] == -1:
            if self.estimator == "gaussian":
                self.mutual_information_matrix = 1 * self.cov
                np.fill_diagonal(self.mutual_information_matrix, np.nan)

                self.mutual_information_matrix = (
                    -np.log(1 - (self.mutual_information_matrix**2)) / 2.0
                )
                np.fill_diagonal(self.mutual_information_matrix, H_SINGLE)

            elif self.estimator == "discrete":
                self.mutual_information_matrix = np.zeros((self.N, self.N))
                for i in range(self.N):
                    for j in range(i + 1):
                        self.mutual_information_matrix[i, j] = (
                            discrete_mutual_information((i,), (j,), self.data)[1]
                        )

                diag = 1 * np.diag(self.mutual_information_matrix)
                self.mutual_information_matrix += self.mutual_information_matrix.T
                np.fill_diagonal(self.mutual_information_matrix, diag)

        return self.mutual_information_matrix

    def estimate_conditional_mutual_information(
        self, inputs_x: tuple, inputs_y: tuple, inputs_z: tuple, return_locals=True
    ):

        if self.estimator == "gaussian":
            mi = differential_conditional_mutual_information(
                inputs_x, inputs_y, inputs_z, self.cov
            )
            if return_locals == True:
                mi_ptw = local_differential_conditional_mutual_information(
                    inputs_x, inputs_y, inputs_z, self.data, self.cov
                )
        elif self.estimator == "discrete":
            mi_ptw, mi = discrete_conditional_mutual_information(
                inputs_x, inputs_y, inputs_z, self.data
            )

        if return_locals == True:
            return mi, mi_ptw
        else:
            return mi

    def estimate_total_correlation(self, inputs: tuple = (-1,), return_locals=False):

        if self.estimator == "gaussian":
            tc = differential_total_correlation(self.cov, inputs=inputs)
            if return_locals == True:
                tc_ptw = local_differential_total_correlation(
                    self.data, self.cov, inputs
                )
        elif self.estimator == "discrete":
            tc_ptw, tc = discrete_total_correlation(self.data)

        if inputs == (-1,):
            self.total_correlation = tc

        if return_locals == True:
            return tc, tc_ptw
        else:
            return tc

    def estimate_dual_total_correlation(
        self, inputs: tuple = (-1,), return_locals=False
    ):

        if self.estimator == "gaussian":
            dtc = differential_dual_total_correlation(self.cov, inputs=inputs)
            if return_locals == True:
                dtc_ptw = local_differential_dual_total_correlation(
                    self.data, self.cov, inputs
                )
        elif self.estimator == "discrete":
            dtc_ptw, dtc = discrete_dual_total_correlation(self.data)

        if inputs == (-1,):
            self.dual_total_correlation = dtc

        if return_locals == True:
            return dtc, dtc_ptw
        else:
            return dtc

    def estimate_o_information(self, inputs: tuple = (-1,), return_locals=False):

        if self.estimator == "gaussian":
            o = differential_o_information(self.cov, inputs=inputs)
            if return_locals == True:
                o_ptw = local_differential_o_information(self.data, self.cov, inputs)
        elif self.estimator == "discrete":
            o_ptw, o = discrete_o_information(self.data)

        if inputs == (-1,):
            self.o_information = o

        if return_locals == True:
            return o, o_ptw
        else:
            return o

    def estimate_s_information(self, inputs: tuple = (-1,), return_locals=False):

        if self.estimator == "gaussian":
            s = differential_s_information(self.cov, inputs=inputs)
            if return_locals == True:
                s_ptw = local_differential_s_information(self.data, self.cov, inputs)
        elif self.estimator == "discrete":
            s_ptw, s = discrete_s_information(self.data)

        if inputs == (-1,):
            self.s_information = s

        if return_locals == True:
            return s, s_ptw
        else:
            return s

    def estimate_description_complexity(
        self, inputs: tuple = (-1,), return_locals=False
    ):

        if self.estimator == "gaussian":
            c = differential_description_complexity(self.cov, inputs=inputs)
            if return_locals == True:
                c_ptw = local_differential_description_complexity(
                    self.data, self.cov, inputs
                )
        elif self.estimator == "discrete":
            c_ptw, c = discrete_description_complexity(self.data)

        if inputs == (-1,):
            self.description_complexity = c

        if return_locals == True:
            return c, c_ptw
        else:
            return c

    def estimate_tse_complexity(self, num_samples: int):

        if self.estimator == "gaussian":
            tse = differential_tse_complexity(self.cov, num_samples)
        elif self.estimator == "discrete":
            tse = discrete_tse_complexity(self.data, num_samples)

        self.tse_complexity = tse

        return tse
