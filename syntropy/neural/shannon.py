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


