import torch
from .shannon import differential_entropy
from __future__ import annotations

# %%
def total_correlation(
    idxs: tuple[int],
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

    float


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
    idxs: tuple[int],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Computes the O-information, S-information, total correlation, and dual total correlation for the data.
    Computing them all as a set is more efficient than computing each one independently.

    Parameters
    ----------
    idxs : tuple[int, ...]
        The tuple of indices the differential entropy is computed for.
    data : torch.Tensor
        The training data, in samples x features format.
    context : None | tuple[int]
        If not None, the indices of the conditioning variables.
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
