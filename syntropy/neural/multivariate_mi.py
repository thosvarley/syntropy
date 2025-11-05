import scipy.stats as stats
import torch
from shannon import differential_entropy


def total_correlation(
    idxs: tuple[int],
    data: torch.Tensor,
    context: None | tuple[int] = None,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Computes the total correlation of the data using normalizing flow estimators.  


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
    float


    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    if context is None:
        context_arg = None
    else:
        context_arg = context

    lookup: dict[tuple[int,...], float] = {}

    h_idxs, _ = differential_entropy(
        idxs=idxs,
        data=data,
        context=context_arg,
        data_test=data_test,
        verbose=verbose,
        train_kwargs=train_kwargs,
        flow_kwargs=flow_kwargs,
    )
    lookup[idxs] = h_idxs

    for i in idxs:
        h_i, _ = differential_entropy(
            idxs=(i,),
            data=data,
            context=context_arg,
            data_test=data_test,
            verbose=verbose,
            flow_kwargs=flow_kwargs,
            train_kwargs=train_kwargs,
        )

        lookup[(i,)] = h_i

    tc: float = 0.0

    for key in lookup.keys():
        if len(key) == 1:
            tc += lookup[key][0]
        else:
            tc -= lookup[key][0]

    return tc


def higher_order_information(
    idxs: tuple[int],
    data: torch.Tensor,
    context: None | tuple[int] = None,
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
    dict[str, float]
    
        

    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    if context is None:
        context_arg = None
    else:
        context_arg = context

    lookup_marginals: dict[tuple[int, ...], float] = {(i,): 0.0 for i in idxs}
    lookup_residuals: dict[tuple[int, ...], float] = {tuple(j for j in idxs if j != i): 0.0 for i in idxs}
    
    N: int = len(idxs)
    
    h_idxs, _ = differential_entropy(
        idxs=idxs,
        data=data,
        context=context_arg,
        data_test=data_test,
        verbose=verbose,
        train_kwargs=train_kwargs,
        flow_kwargs=flow_kwargs,
    )
    
    for i in idxs:
        h_i, _ = differential_entropy(
            idxs=(i,),
            data=data,
            context=context_arg,
            data_test=data_test,
            verbose=verbose,
            flow_kwargs=flow_kwargs,
            train_kwargs=train_kwargs,
        )
        lookup_marginals[(i,)] = h_i

        idxs_minus_i = tuple(j for j in idxs if j != i)
        h_minus_i, _ = differential_entropy(
            idxs = idxs_minus_i,
            data = data, 
            context = context_arg,
            data_test = data_test,
            verbose = verbose,
            flow_kwargs = flow_kwargs,
            train_kwargs = train_kwargs
        )
        lookup_residuals[idxs_minus_i] = h_minus_i

    tc_idxs: float =  sum(lookup_marginals[key] for key in lookup_marginals.keys()) - h_idxs
    
    residual_tcs: dict = {}
    for key in lookup_residuals:
        residual_tc = -lookup_residuals[key]
        for i in key:
            residual_tc += lookup_marginals[(i,)]

        residual_tcs[key] = residual_tc

    results = {
        "o_information": -(((N-2)*tc_idxs) - sum(residual_tcs[key] for key in residual_tcs.keys())),
        "s_information": (N*tc_idxs) - sum(residual_tcs[key] for key in residual_tcs.keys()),
        "dual_total_correlation": ((N-1)*tc_idxs) - sum(residual_tcs[key] for key in residual_tcs.keys()),
        "total_correlation": tc_idxs 
    }

    return results 
