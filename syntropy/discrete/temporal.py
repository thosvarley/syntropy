import numpy as np
from typing import Any

def lempel_ziv_complexity(
    X: list[Any], return_dictionary: bool = False
) -> float | tuple[float, set]:
    """
    Uses the classic Lempel-Ziv compression algorithm to estimate the entropy rate of a one-dimensional array with :math:`N` samples. 
    If each element in X is a multi-dimensional tuple, then the result is equivalent to the joint entropy rate.

    The extension to multivariate Lempel-Ziv is straightforward and involves representing the joint state of each element at time t as a tuple (X(t), Y(t)) and treating the two sources as a single joint source.

    Here, the dictionary length :math:`|D|` is normalized:

    .. math::
        \\textnormal{Complexity}(X) = \\frac{|D|\\log|D|}{N}

    Parameters
    ----------
    X : list
        A discrete array. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.)..
    return_substrings : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    float
        The estimated Lempel-Ziv complexity.
    set
        The dictionary (only returned if return_dictionary == True)

    References
    ----------

    Schartner, M. M., Carhart-Harris, R. L., Barrett, A. B., Seth, A. K., & Muthukumaraswamy, S. D. (2017). 
    Increased spontaneous MEG signal diversity for psychoactive doses of ketamine, LSD and psilocybin. 
    Scientific Reports, 7, 46421. 
    https://doi.org/10.1038/srep46421

    Blanc, J.-L., Schmidt, N., Bonnier, L., Pezard, L., & Lesne, A. (2008).
    Quantifying Neural Correlations Using Lempel-Ziv Complexity.
    Deuxième conférence française de Neurosciences Computationnelles,
    Marseille, France.
    https://hal.science/hal-00331599/document
    
    Zozor, S., Ravier, P., & Buttelli, O. (2005).
    On Lempel–Ziv complexity for multidimensional data analysis.
    Physica A: Statistical Mechanics and Its Applications, 345(1), 285–302.
    https://doi.org/10.1016/j.physa.2004.07.025
    """

    tuples: tuple = tuple((i,) for i in X)
    N: int = len(tuples)
    i = 0

    dictionary = {tuples[0]}  # Store unique substrings

    while i < N - 1:  # Scanning the string
        j = i + 1
        while j < N and tuples[i : j + 1] in dictionary:
            j += 1
        dictionary.add(tuples[i : j + 1])
        i = j  # Move index to next position

    c = len(dictionary)
    complexity = (c * np.log2(c)) / N  # The +1 is removed for consistency w/ Coutinho

    if return_dictionary is True:
        return complexity, dictionary
    else:
        return complexity


def lempel_ziv_mutual_information(X: list, Y: list) -> float:
    """
    Estimates the discrete mutual information rate for two channels X and Y with the Lempel-Ziv compression algorithm.
    This measure can be transiently negative, although in the limit it approximates the discrete information rate.

    .. math::
        I_{LZ}(X;Y) = LZ(X) + LZ(Y) - LZ(X,Y)

    Parameters
    ----------
    X : list
        A discrete list. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.)..
    Y : list
        A discrete list. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.)..

    Returns
    -------
    float
        The estimated mutual information rate.

    References
    ----------
    Zozor, S., Ravier, P., & Buttelli, O. (2005).
    On Lempel–Ziv complexity for multidimensional data analysis.
    Physica A: Statistical Mechanics and Its Applications, 345(1), 285–302.
    https://doi.org/10.1016/j.physa.2004.07.025

    Blanc, J.-L., Schmidt, N., Bonnier, L., Pezard, L., & Lesne, A. (2008).
    Quantifying Neural Correlations Using Lempel-Ziv Complexity.
    Deuxième conférence française de Neurosciences Computationnelles,
    Marseille, France.
    https://hal.science/hal-00331599/document

    """

    joint = list(zip(X, Y))

    return (
        lempel_ziv_complexity(X)
        + lempel_ziv_complexity(Y)
        - lempel_ziv_complexity(joint)
    )


def conditional_lempel_ziv_complexity(X: list, Y: list) -> float:
    """
    The conditional entropy rate estimated with the Lempel-Ziv algorithm.

    .. math::
        LZ(X|Y) = LZ(X,Y) - LZ(Y)

    Parameters
    ----------
    X : list
        A discrete list. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.).
    Y : list
        A discrete list. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.).

    Returns
    -------
    float
        The estimated conditional entropy rate.

    References
    ----------
    Zozor, S., Ravier, P., & Buttelli, O. (2005).
    On Lempel–Ziv complexity for multidimensional data analysis.
    Physica A: Statistical Mechanics and Its Applications, 345(1), 285–302.
    https://doi.org/10.1016/j.physa.2004.07.025

    """

    joint: list = list(zip(X, Y))

    return lempel_ziv_complexity(joint) - lempel_ziv_complexity(Y)


def lempel_ziv_total_correlation(data: np.ndarray) -> float:
    """
    A straightforward generalization of the mutual information rate given by
    Zozor et al., and Blanc et al.,

    .. math:: 
        TC_{LZ}(X) = \\sum_{i=1}^{N} LZ(X_i) - LZ(X)


    Parameters
    ----------
    data : np.ndarray
        A multi-dimensional discrete array, assumed to be in
        channels x time format.

    Returns
    -------
    float
        The estimated total correlation rate.

    """

    N0: int
    N1: int

    N0, N1 = data.shape

    sum_marginals: float = 0.0
    for i in range(N0):
        sum_marginals += lempel_ziv_complexity(data[i, :])

    joint = [tuple(data[:, i]) for i in range(N1)]

    return sum_marginals - lempel_ziv_complexity(joint)


def cross_lempel_ziv_complexity(X: list, Y: list) -> int:
    """
    Computes the Lempel-Ziv complexity of a string X
    using a pre-constructed dictionary optimized on Y. The relative
    Lempel-Ziv complexity is the extra patterns that appear in X but
    not in the compressed Y.

    This code uses the optimized dictionary for Y, rather than searching all
    substrings of Y. Using all possible substrings is very impractical for
    long time series. As a result, however, it is very sensitive to the
    particular temporal ordering of fluctuations in X and Y.

    Should only be used when X and Y were recorded at the same time, such
    as channels in an EEG/fMRI/MEG recording.

    MATHEMATICAL GARUNTEES GIVEN IN ZIV & MERHAV 1993 ARE NOT PRESERVED
    BY THIS METHOD. THIS FUNCTION REMAINS EXPERIMENTAL

    Parameters
    ----------
    X : np.ndarray
        A discrete array. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.).
    Y : np.ndarray
        A discrete array. Can contain digits 0-9 and/or
        single-character strings ("A", "a", "B", "b", etc.).

    Returns
    -------
    complexity : int
        The Lempel-ziv relative complexity.

    References
    ----------
    Ziv, J., & Merhav, N. (1993).
    A Measure of Relative Entropy between Individual Sequences with
    Application to Universal Classification.
    Proceedings. IEEE International Symposium on Information Theory, 352–352.
    https://doi.org/10.1109/ISIT.1993.748668

    """
    X_tuples = tuple((i,) for i in X)
    N_X = len(X_tuples)

    # Step 1: Build LZ dictionary from Y (sequential parsing, no full substring storage)
    _, dictionary_Y = lempel_ziv_complexity(Y, return_dictionary=True)

    # Step 2: Parse X with dictionary_Y
    i, c = 0, 0
    while i < N_X - 1:
        j = i + 1
        while j <= N_X and X_tuples[i : j + 1] in dictionary_Y:
            j += 1
        dictionary_Y.add(X_tuples[i : j + 1])  # Store new substrings
        c += 1  # A new additions
        i = j  # Move forward

    complexity = c * np.log2(c) / N_X

    return complexity
