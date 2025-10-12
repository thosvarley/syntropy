import numpy as np
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

    Parameters
    ----------
    idxs : tuple[int]

    data : torch.Tensor

    context : None | tuple[int]

    data_test : None | torch.Tensor

    flow_kwargs : dict

    train_kwargs : dict

    verbose : bool


    Returns
    -------



    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    if context is None:
        context_arg = None
    else:
        context_arg = context

    lookup: dict = {}

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
):
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    if context is None:
        context_arg = None
    else:
        context_arg = context

    lookup_marginals: dict = {(i,): None for i in idxs}
    lookup_residuals: dict = {tuple(j for j in idxs if j != i): None for i in idxs}
    
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

# %%
in_dir = "/home/thosvarley/Documents/UVM/research/normalizing_flow/data/"
from syntropy.gaussian.shannon import mutual_information as MI

cov: np.ndarray = np.load(in_dir + "six_cov.npz")["arr_0"]

gaussian = stats.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
data_train = torch.Tensor(gaussian.rvs(10_000))
data_test = torch.Tensor(gaussian.rvs(10_000))


idxs = (0, 1, 3)
train_kwargs = {"num_epochs": 50}
results = higher_order_information(
    idxs, data_train, data_test=data_test, train_kwargs=train_kwargs, verbose=True
)

# %%
from syntropy.gaussian.multivariate_mi import o_information, total_correlation, s_information, dual_total_correlation 
cov_ = cov[idxs,:][:,idxs]
print(o_information(cov_))
print(s_information(cov_))
print(dual_total_correlation(cov_))
print(total_correlation(cov_))
