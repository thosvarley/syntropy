import numpy as np
import scipy.signal as signal
import scipy.integrate as integrate
from numpy.typing import NDArray


def construct_csd_tensor(
    idxs: tuple[int, ...], data: NDArray[np.floating], fs: int = 1, nperseg: int = 1024
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """

    Parameters
    ----------
    idxs : tuple[int, ...]

    data : NDArray[np.floating]

    fs : int

    nperseg : int


    Returns
    -------
    NDArray[np.floating]


    """
    N: int = len(idxs)
    S: NDArray[np.floating] = np.zeros((nperseg, N, N))

    for i in range(N):
        for j in range(i + 1):
            f, Pij = signal.csd(
                data[idxs[i], :],
                data[idxs[j], :],
                return_onesided=False,
                fs=fs,
                nperseg=nperseg,
            )
            Pij = np.fft.fftshift(Pij)
            S[:i, j] = Pij.real
            S[:, j, i] = np.conj(Pij).real

    omega = 2 * np.pi * f / fs
    omega = np.fft.fftshift(omega)

    return S, omega


def differential_entropy_rate(
    idxs: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the differential entropy rate of a potentially multivariate stochastic process.

    :math:`H(X)=\\frac{1}{4\\pi} \int_{-\\pi}^{\\pi} \\log \\left( (2\\pi e)^N |S_X(\\omega)| \\right) \\, d\\omega`

    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the analysis.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local entropies for each frequency band.
    float
        The average entropy across the whole spectrum.
    """
    # %%
    N = len(idxs)

    S, omega = construct_csd_tensor(idxs=idxs, data=data, fs=fs, nperseg=nperseg)

    ptw = np.array(
        [(1 / 2) * np.log(((2 * np.pi * np.e) ** N) * np.linalg.det(a)) for a in S]
    )
    avg = (1 / (2 * np.pi)) * integrate.simpson(ptw, omega)
    # %%
    return ptw, avg


def mutual_information_rate(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
) -> tuple[NDArray[np.floating], float]:
    """
    Computes the mutual information rate between two (potentially multivariate) Gaussian processes.

    :math:`I(X; Y) = \\frac{1}{4\\pi} \\int_{-\\pi}^{\\pi} \\log \\left( \\frac{  |S_X(\\omega)||S_Y(\\omega)| }{ |S_{XY}(\\omega)| } \\right) d\\omega`

    See:

    Faes, L., Sparacino, L., Mijatovic, G., Antonacci, Y., Ricci, L., Marinazzo, D., & Stramaglia, S. (2025).
    Partial Information Rate Decomposition (No. arXiv:2502.04550). arXiv.
    https://doi.org/10.48550/arXiv.2502.04550

    Faes, L., Pernice, R., Mijatovic, G., Antonacci, Y., Krohova, J. C., Javorka, M., & Porta, A. (2021).
    Information decomposition in the frequency domain: A new framework to study cardiovascular and cardiorespiratory oscillations.
    Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 379(2212), 20200250.
    https://doi.org/10.1098/rsta.2020.0250

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        The indices of the channels to include in the inputs.
    idxs_y : tuple[int, ...]
        The indices of the channels to include in the target.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local mutual informations for each frequency band.
    float
        The average mutual information across the whole spectrum.
    """
    Nx: int = len(idxs_x)
    Ny: int = len(idxs_y)

    idxs_: tuple[int, ...] = idxs_x + idxs_y

    reidxs_x: tuple[int, ...] = tuple(i for i in range(Nx))
    reidxs_y: tuple[int, ...] = tuple(i + Nx for i in range(Ny))

    S, omega = construct_csd_tensor(idxs=idxs_, data=data, fs=fs, nperseg=nperseg)

    log_x: NDArray[np.floating] = np.array(
        [np.linalg.slogdet(a[np.ix_(reidxs_x, reidxs_x)])[1] for a in S]
    )
    log_y: NDArray[np.floating] = np.array(
        [np.linalg.slogdet(a[np.ix_(reidxs_y, reidxs_y)])[1] for a in S]
    )
    log_xy: NDArray[np.floating] = np.array([np.linalg.slogdet(a)[1] for a in S])

    ptw = (1 / 2) * (log_x + log_y - log_xy)
    avg = (1 / (2 * np.pi)) * integrate.simpson(ptw, omega)

    return ptw, avg


def total_correlation_rate(
    idxs: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
) -> tuple[NDArray[np.floating], float]:
    """
    A straightforward extension of the mutual information rate to the total correlation.

    :math:`TC(X,Y,\\ldots,Z) = \\frac{1}{4\\pi} \\int_{-\\pi}^{\\pi} \\log \\left( \\frac{  |S_X(\\omega)||S_Y(\\omega)|\\ldots|S_Z(\\omega)| }{ |S_{XY\\ldots Z}(\\omega)| } \\right) d\\omega`

    WARNING: As far as I know this TC rate idea has never been formally explored before. It should work fine as a natural generalization of the MI, but it hasn't ever been published or peer reviewed.

    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the total correlation calculation.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local total correlation  for each frequency band.
    float
        The average total correlation across the whole spectrum.
    """

    S, omega = construct_csd_tensor(idxs=idxs, data=data, fs=fs, nperseg=nperseg)

    sum_parts = np.array([np.log(np.diag(a)) for a in S]).sum(axis=-1)

    whole = np.array([np.linalg.slogdet(a)[1] for a in S])

    ptw = (1 / 2) * (sum_parts - whole)
    avg = (1 / (2 * np.pi)) * integrate.simpson(ptw, omega)

    return ptw, avg


def k_wms_rate(
    idxs: tuple[int, ...],
    k: int,
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
    verbose: bool = False,
) -> tuple[NDArray[np.floating], float]:
    """
    A straightforward extension of the total correlation rate to the K whole-minus-sum rate.

    Recall that S-information, DTC, and negative O-information can all be written in a general form:

    :math:`WMS^{k}(X) = (N-k)TC(X) - \\sum_{i=1}^{N}TC(X^{-i})`

    See:
        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w


    WARNING: As far as I know this rate idea has never been formally explored before. It should work fine as a natural generalization of the MI, but it hasn't even been published or peer reviewed.

    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the calculation.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local k_wms rate for each frequency band.
    float
        The average k_wms across the whole spectrum.
    """
    N0: int = len(idxs)
    ptw_whole: NDArray[np.floating]
    avg_whole: float 
    
    ptw_whole, avg_whole = total_correlation_rate(
        idxs=idxs, data=data, fs=fs, nperseg=nperseg
    )

    ptw_whole *= (N0 - k)
    avg_whole *= (N0 - k)

    ptw_sum_parts: NDArray[np.floating] = np.zeros_like(ptw_whole)
    avg_sum_parts: float = 0.0

    for i in range(N0):
        idxs_residual: tuple[idxs, ...] = tuple(idxs[j] for j in range(N0) if j != i)

        ptw_residuals, avg_residuals = total_correlation_rate(
            idxs=idxs_residual, data=data, fs=fs, nperseg=nperseg
        )

        ptw_sum_parts += ptw_residuals
        avg_sum_parts += avg_residuals

        if verbose is True:
            print((i + 1) / N0)

    return ptw_whole - ptw_sum_parts, avg_whole - avg_sum_parts


def s_information_rate(
    idxs: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
    verbose: bool = False,
) -> tuple[NDArray[np.floating], float]:
    """
    A straightforward extension of the S-information rate from the total correlation rate.

    The S-information is equivalant to:

    :math:`\\Sigma(X) = \\sum_{i=1}^{N}I(X_i;X^{-i})`

    :math:`\\Sigma(X) = TC(X) + DTC(X)`

    WARNING: As far as I know this rate idea has never been formally explored before. It should work fine as a natural generalization of the MI, but it hasn't even been published or peer reviewed.

    See:
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305

        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the calculation.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local S-information rate for each frequency band.
    float
        The average S-information across the whole spectrum.
    """

    ptw_s, avg_s = k_wms_rate(
        idxs=idxs, k=0, data=data, fs=fs, nperseg=nperseg, verbose=verbose
    )
    return ptw_s, avg_s


def dual_total_correlation_rate(
    idxs: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
    verbose: bool = False,
) -> tuple[NDArray[np.floating], float]:
    """
    A straightforward extension of the dual total correlation rate from the total correlation rate.

    The dual total correlation is given alternately by:

    :math:`DTC(X) = H(X) - \\sum_{i=1}^{N}H(X_i|X^{-i})`

    :math:`DTC(X) = (N-1)TC(X) - \\sum_{i=1}^{N}TC(X^{-i})`

    WARNING: As far as I know this rate idea has never been formally explored before. It should work fine as a natural generalization of the MI, but it hasn't even been published or peer reviewed.

    See:
        Abdallah, S. A., & Plumbley, M. D. (2012).
        A measure of statistical complexity based on predictive information with application to finite spin systems.
        Physics Letters A, 376(4), 275–281.
        https://doi.org/10.1016/j.physleta.2011.10.066

        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305


    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the calculation.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local dual total correlation rate for each frequency band.
    float
        The average dual total correlation across the whole spectrum.
    """

    ptw_dtc, avg_dtc = k_wms_rate(
        idxs=idxs, k=1, data=data, fs=fs, nperseg=nperseg, verbose=verbose
    )
    return ptw_dtc, avg_dtc


def o_information_rate(
    idxs: tuple[int, ...],
    data: NDArray[np.floating],
    fs: int = 1,
    nperseg: int = 1024,
    verbose: bool = False,
) -> tuple[NDArray[np.floating], float]:
    """
    A straightforward extension of the O-information rate from the total correlation rate.

    :math:`\\Omega(X) = (2-N)TC(X) + \\sum_{i=1}^{N}TC(X^{-i})`

    :math:`\\Omega(X) = TC(X) - DTC(X)`

    WARNING: As far as I know this rate idea has never been formally explored before. It should work fine as a natural generalization of the MI, but it hasn't even been published or peer reviewed.

    See:
        Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
        Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information.
        Physical Review E, 100(3), Article 3.
        https://doi.org/10.1103/PhysRevE.100.032305

        Varley, T. F., Pope, M., Faskowitz, J., & Sporns, O. (2023).
        Multivariate information theory uncovers synergistic subsystems of the human cerebral cortex.
        Communications Biology, 6(1), Article 1.
        https://doi.org/10.1038/s42003-023-04843-w

    Parameters
    ----------
    idxs : tuple[int, ...]
        The indices of the channels to include in the calculation.
    data : NDArray[np.floating]
        The data in channels x samples format.
    fs: int
        The sampling rate of the time series data in Hz.
        The default is 1.
    nperseg: int
        The number of samples to include in each segment.
        Passed to the various scipy.signal functions that compute the spectral analyses.
        The default is 1024

    Returns
    -------
    NDArray[np.floating]
        The local O-information rate for each frequency band.
    float
        The average O-information across the whole spectrum.
    """

    ptw_o, avg_o = k_wms_rate(
        idxs=idxs, k=2, data=data, fs=fs, nperseg=nperseg, verbose=verbose
    )
    return -ptw_o, -avg_o
