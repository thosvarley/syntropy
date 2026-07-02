from __future__ import annotations

import torch
from .shannon import differential_entropy


# %%
def total_correlation(
    idxs: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, float]:
    """
    Computes the total correlation of the data using normalizing flow estimators.


    Parameters
    ----------
    idxs : tuple[int, ...]
        The tuple of indices the differential entropy is computed for.
    data : torch.Tensor
        The training data, in samples x features format.
    data_test : None | torch.Tensor
        If not None, the testing data in samples x features format.
    flow_kwargs : dict
        Arguments for the utils.initalize_flow function.
    train_kwargs : dict
        Arguments for the utils.train_flow function.
    verbose : bool
        Whether to print the training progress.


    Returns
    -------
    torch.Tensor
        The local total correlation for each sample.
    float
        The expected total correlation over all samples.

    References
    ----------
    Watanabe, S. (1960).
    Information theoretical analysis of multivariate correlation.
    IBM Journal of Research and Development, 4(1), 66-82.
    https://doi.org/10.1147/rd.41.0066

    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    lookup: dict[tuple[int, ...], torch.Tensor] = {}

    h: torch.Tensor
    h, _, _ = differential_entropy(
        idxs=idxs,
        data=data,
        data_test=data_test,
        verbose=verbose,
        train_kwargs=train_kwargs,
        flow_kwargs=flow_kwargs,
    )
    lookup[idxs] = h

    h_i: torch.Tensor
    for i in idxs:
        h_i, _, _ = differential_entropy(
            idxs=(i,),
            data=data,
            data_test=data_test,
            verbose=verbose,
            flow_kwargs=flow_kwargs,
            train_kwargs=train_kwargs,
        )

        lookup[(i,)] = h_i

    tc: torch.Tensor = torch.zeros(data.shape[0])

    for key in lookup.keys():
        if len(key) == 1:
            tc += lookup[key]
        else:
            tc -= lookup[key]

    return tc, tc.mean()


def higher_order_information(
    idxs: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> dict[str, dict[str, float | torch.Tensor]]:
    """
    Computes the O-information, S-information, total correlation, and dual total correlation for the data.
    Computing them all as a set is more efficient than computing each one independently.

    Parameters
    ----------
    idxs : tuple[int, ...]
        The tuple of indices the differential entropy is computed for.
    data : torch.Tensor
        The training data, in samples x features format.
    data_test : None | torch.Tensor
        If not None, the testing data in samples x features format.
    flow_kwargs : dict
        Arguments for the utils.initalize_flow function.
    train_kwargs : dict
        Arguments for the utils.train_flow function.
    verbose : bool
        Whether to print the training progress.



    Returns
    -------
    dict[str, dict[str, float | torch.Tensor]]
        A dictionary keyed by "sinfo", "dtc", "oinfo", and "tc", each
        mapping to a dict with keys "ptw" (the local/pointwise values,
        torch.Tensor) and "avg" (the expected value, float).

    References
    ----------
    Watanabe, S. (1960).
    Information theoretical analysis of multivariate correlation.
    IBM Journal of Research and Development, 4(1), 66-82.
    https://doi.org/10.1147/rd.41.0066

    Abdallah, S. A., & Plumbley, M. D. (2012).
    A measure of statistical complexity based on predictive information
    with application to finite spin systems.
    Physics Letters A, 376(4), 275-281.
    https://doi.org/10.1016/j.physleta.2011.10.066

    Rosas, F., Mediano, P. A. M., Gastpar, M., & Jensen, H. J. (2019).
    Quantifying High-order Interdependencies via Multivariate
    Extensions of the Mutual Information.
    Physical Review E, 100(3), Article 3.
    https://doi.org/10.1103/PhysRevE.100.032305

    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    lookup_marginals: dict[tuple[int, ...], torch.Tensor] = {}
    lookup_residuals: dict[tuple[int, ...], torch.Tensor] = {}

    N: int = len(idxs)

    h, _, _ = differential_entropy(
        idxs=idxs,
        data=data,
        data_test=data_test,
        verbose=verbose,
        train_kwargs=train_kwargs,
        flow_kwargs=flow_kwargs,
    )

    for i in idxs:
        h_i, _, _ = differential_entropy(
            idxs=(i,),
            data=data,
            data_test=data_test,
            verbose=verbose,
            flow_kwargs=flow_kwargs,
            train_kwargs=train_kwargs,
        )
        lookup_marginals[(i,)] = h_i

        idxs_r = tuple(j for j in idxs if j != i)

        h_r, _, _ = differential_entropy(
            idxs=idxs_r,
            data=data,
            data_test=data_test,
            verbose=verbose,
            flow_kwargs=flow_kwargs,
            train_kwargs=train_kwargs,
        )
        lookup_residuals[idxs_r] = h_r

    tc: torch.Tensor = -h
    for key in lookup_marginals.keys():
        tc += lookup_marginals[key]

    residual_tcs: torch.Tensor = torch.zeros(data.shape[0])
    for key in lookup_residuals.keys():
        tc_r = -lookup_residuals[key]
        for i in key:
            tc_r += lookup_marginals[(i,)]

        residual_tcs += tc_r

    oinfo: torch.Tensor = ((2 - N) * tc) + residual_tcs
    dtc: torch.Tensor = ((N - 1) * tc) - residual_tcs
    sinfo: torch.Tensor = (N * tc) - residual_tcs

    results: dict[str, dict[str, float | torch.Tensor]] = {
        "sinfo": {"ptw": sinfo, "avg": sinfo.mean()},
        "dtc": {"ptw": dtc, "avg": dtc.mean()},
        "oinfo": {"ptw": oinfo, "avg": oinfo.mean()},
        "tc": {"ptw": tc, "avg": tc.mean()},
    }

    return results
