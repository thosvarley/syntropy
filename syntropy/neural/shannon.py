import numpy as np
import scipy.stats as stats
import torch
from utils import initialize_flow, train_flow, evaluate_flow


def differential_entropy(
    idxs: tuple[int],
    data: torch.Tensor,
    context: None | tuple[int] = None,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> float:

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

    # select context tensor if provided
    context_arg: None | torch.Tensor = None
    if context is not None:
        context_arg = data[:, context]
        flow_kwargs["dim_context"] = len(context)

    # initialize the flow
    dim: int = len(idxs)
    flow = initialize_flow(dim=dim, **flow_kwargs)

    # train the flow
    flow = train_flow(
        flow=flow,
        data=data[:, idxs],
        context=context_arg,
        verbose=verbose,
        **train_kwargs,
    )
    # evaluation
    eval_data = data if data_test is None else data_test
    eval_context = (
        None
        if context is None
        else (data_test[:, context] if data_test is not None else context_arg)
    )

    h = evaluate_flow(flow, eval_data[:, idxs], context=eval_context)

    return h, flow


def mutual_information(
    idxs_x: tuple[int],
    idxs_y: tuple[int],
    data: torch.Tensor,
    context: None | tuple[int] = None,
    data_test: None | torch.Tensor = None,
    flow_kwargs: dict = None,
    train_kwargs: dict = None,
    verbose: bool = False,
) -> float:
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    if context is None:
        context_arg = None
        conditional_context_arg = idxs_y
    else:
        context_arg = context
        conditional_context_arg = idxs_y + context

    h_x,  _, = differential_entropy(
        idxs=idxs_x,
        data=data,
        context=context_arg,
        data_test=data_test,
        verbose=verbose,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
    )
    h_x_given_y, _ = differential_entropy(
        idxs=idxs_x,
        data=data,
        context=conditional_context_arg,
        data_test=data_test,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
        verbose=verbose,
    )

    mi = h_x - h_x_given_y

    return mi 


"""
in_dir = "/home/thosvarley/Documents/UVM/research/normalizing_flow/data/"
from syntropy.gaussian.shannon import conditional_mutual_information as cMI

cov: np.ndarray = np.load(in_dir + "six_cov.npz")["arr_0"]

gaussian = stats.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov)
data_train = torch.Tensor(gaussian.rvs(20_000))
data_test = torch.Tensor(gaussian.rvs(20_000))


idxs_x = (0, 1)
idxs_y = (2, 3)
idxs_z = (5, 4)

flow_kwargs = {"num_layers": 10, "hidden_features": 64}

train_kwargs = {"num_epochs": 50, "lr": 1e-4,
                "convergence_threshold": 0.01}

ests = []
for i in range(10):
    mi = mutual_information(
        idxs_x=idxs_x,
        idxs_y=idxs_y,
        data=data_train,
        data_test=data_test,
        context=idxs_z,
        verbose=True,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
    )
    ests.append(mi)
    print(i)

#h, _ = differential_entropy(
#    idxs=idxs_x, data=data_train, data_test=data_test, context=idxs_y + idxs_z, verbose=True
#)



# print(f"MI: {cMI(idxs_x, idxs_y, idxs_z, cov)} nat.")
# print(f"Estimated MI: {mi} +/- {stderr}")
"""
