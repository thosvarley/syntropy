#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:01:12 2025

@author: thosvarley
"""

import numpy as np
import random
import math

from syntropy.gaussian.multivariate_mi import o_information


def neg_o_information(x):
    """
    A utility function for computing the negative of the O-information
    for a hill-climbing optimizer.
    """
    return -1 * o_information(*x)


def simulated_annealing(
    cov: np.ndarray,
    function,
    size: int,
    temperature: float = 1.0,
    cooling_rate: float = 0.999,
    min_temperature: float = 1e-5,
    iters_per_temperature: int = 10,
    convergence_window: int = 100,
    convergence_threshold: float = 1e-6,
) -> (set, float, list):
    """
    Implements a simulated annealing algorithm for optimizing
    multivariate information measures (O-info, DTC, etc) from
    a covariance matrix. Modified from;

        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems
        of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w


    Parameters
    ----------
    cov : np.ndarray
        The covariance matrix that defines the full distribution.
    function : function
        The objective function. Should be taken from the
        syntropy.gaussian.multivariate_mi module.
    size : int
        The size of the ensemble of elements.
    temperature : float, optional
        The initial temperature to initialize the optimization with.
        The default is 1.0.
    cooling_rate : float, optional
        The rate at which the annealing cools.
        Temperature, T, decreases with T(t+1) = T(t)*cooling_rate
        The default is 0.999.
    min_temperature : float, optional
        The stopping temperature. The default is 1e-3.
    iters_per_temperature : int, optional
        How many times to iterate the annealing algorithim before cooling one step.
        The default is 10.
    convergence_window : int, optional
        How many steps back in time to consider for assessing convergence.
        The default is 100.
    convergence_threshold : float, optional
        The standard deviation of the window below which convergence is said
        to have been achieved.. The default is 1e-6.

    Returns
    -------
    set:
        The indicies of the optimal set.
    float:
        The value of the optimal set.
    list:
        The time series of values.
    """
    assert min_temperature > 0, "The minimum temperature must be greater than zero."

    print("Initializing annealing...")

    chosen_set: set = set(np.random.choice(cov.shape[0], size=size, replace=False))

    available_set: set = {i for i in range(cov.shape[0]) if i not in chosen_set}

    value: float = function((cov, tuple(chosen_set)))

    num_steps = (
        math.ceil(np.log(min_temperature / temperature) / np.log(cooling_rate)) + 1
    )
    values: np.ndaray = np.zeros(num_steps)

    best_value: float = value
    best_set: set = chosen_set.copy()

    convergence = False

    counter: int = 1
    print("Annealing...")
    while temperature > min_temperature:

        for _ in range(iters_per_temperature):

            # Randomly pick an available element and an chosen element
            swap = (
                random.choice(tuple(chosen_set)),
                random.choice(tuple(available_set)),
            )

            # Swap the elements, respectively
            chosen_set.remove(swap[0])
            available_set.remove(swap[1])

            chosen_set.add(swap[1])
            available_set.add(swap[0])

            new_value = function((cov, tuple(chosen_set)))

            diff = (new_value - value) / abs(
                value
            )  # I assume this will never *actually* be 0 exactly.

            if new_value > best_value:
                best_value = new_value
                best_set = chosen_set.copy()

            # If the new sets are more optimal than the old ones
            # OR the temperature is high enough to accept a sub-optimal configuraiton
            # update the value.
            if (diff > 0) or random.random() < math.exp(diff / temperature):
                value = new_value
            else:
                chosen_set.remove(swap[1])
                available_set.remove(swap[0])

                chosen_set.add(swap[0])
                available_set.add(swap[1])

        # The saved value is the best value of all iters at a given temperature
        values[counter - 1] = value

        if counter > convergence_window:
            std: float = np.std(values[counter - convergence_window : counter])
            if std < convergence_threshold:
                if value < best_value:
                    print("...testing convergence...")
                    chosen_set = best_set.copy()
                    available_set = {
                        i for i in range(cov.shape[0]) if i not in chosen_set
                    }
                    value = best_value
                else:
                    print("Convergence achieved!")
                    convergence = True
                    break

        temperature *= cooling_rate
        counter += 1

    if convergence == False:
        print("Annealing schedule finished")
        print("No convergence achieved")

    return best_set, best_value, values[:counter]


def irreducible_synergy(cov: np.ndarray, inputs: tuple):
    """
    Computes whether it is possible to remove an element from a
    synergistic sysem and increase the synergy.

    Parameters
    ----------
    cov : np.ndarray
        The covariance matrix that defines the distribution.
    inputs : tuple
        The specific elements to test.

    Returns
    -------
    bool
        Returns True if the synergy is irreducible (i.e. there is no element that, when removed, increaes the synergy).
        Returns False otherwise.

    """

    value = o_information(cov, inputs)
    assert value < 0, "This only works for initially synergy-dominated subsets."
    N = len(inputs)

    for i in range(N):
        reduced_set = tuple(inputs[j] for j in range(N) if j != i)
        if o_information(cov, reduced_set) < value:
            return False
    return True
