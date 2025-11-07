import torch
import nflows
from .utils import initialize_flow, train_flow, evaluate_flow
from typing import Any


def differential_entropy(
    idxs: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: None | dict[str, Any] = None,
    train_kwargs: None | dict[str, Any] = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, float, nflows.flows.base.Flow]:
    """
    Computes the differential entropy of the data.

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
    float

    nflows.flows.base.Flow


    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    # initialize the flow
    dim: int = len(idxs)
    flow = initialize_flow(dim=dim, **flow_kwargs)

    # train the flow
    flow = train_flow(
        flow=flow,
        data=data[:, idxs],
        verbose=verbose,
        **train_kwargs,
    )
    # evaluation
    eval_data = data if data_test is None else data_test
    ptw, avg = evaluate_flow(flow, eval_data[:, idxs])

    return ptw, avg, flow


def conditional_entropy(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: None | dict[str, Any] = None,
    train_kwargs: None | dict[str, Any] = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, float, nflows.flows.base.Flow]:
    """
    Computes the differential entropy of the data.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        The tuple of indices the differential entropy is computed for.
    idxs_y : tuple[int, ...]
        The tuple of indices to condition on.
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

    nflows.flows.base.Flow


    """
    flow_kwargs = flow_kwargs or {}
    train_kwargs = train_kwargs or {}

    flow_kwargs["dim_context"] = len(idxs_y)

    # initialize the flow
    dim: int = len(idxs_x)
    flow = initialize_flow(dim=dim, **flow_kwargs)

    # train the flow
    flow = train_flow(
        flow=flow,
        data=data[:, idxs_x],
        context=data[:, idxs_y],
        verbose=verbose,
        **train_kwargs,
    )
    # evaluation
    eval_data = data if data_test is None else data_test
    
    ptw: torch.Tensor
    avg: float

    ptw, avg = evaluate_flow(flow, eval_data[:, idxs_x], context=eval_data[:, idxs_y])

    return ptw, avg, flow


def mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: None | dict[str, Any] = None,
    train_kwargs: None | dict[str, Any] = None,
    verbose: bool = False,
) -> float:
    """
    Computes the mutual information between the elements given by idxs_x and idxs_y.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        The tuple of indices the x variable.
    idxs_y : tuple[int, ...]
        The tuple of indices for the y variable.
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

    ptw_x, avg_x, _ = differential_entropy(
        idxs=idxs_x,
        data=data,
        data_test=data_test,
        verbose=verbose,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
    )
    ptw_cond, avg_cond, _ = conditional_entropy(
        idxs_x=idxs_x,
        idxs_y=idxs_y,
        data=data,
        data_test=data_test,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
        verbose=verbose,
    )

    ptw: torch.Tensor = ptw_x - ptw_cond
    mi: float = avg_x - avg_cond

    return ptw, mi


def conditional_mutual_information(
    idxs_x: tuple[int, ...],
    idxs_y: tuple[int, ...],
    idxs_z: tuple[int, ...],
    data: torch.Tensor,
    data_test: None | torch.Tensor = None,
    flow_kwargs: None | dict[str, Any] = None,
    train_kwargs: None | dict[str, Any] = None,
    verbose: bool = False,
) -> float:
    """
    Computes the mutual information between the elements given by idxs_x and idxs_y.

    Parameters
    ----------
    idxs_x : tuple[int, ...]
        The tuple of indices the x variable.
    idxs_y : tuple[int, ...]
        The tuple of indices for the y variable.
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

    ptw_x_given_z, avg_x_given_z, _ = conditional_entropy(
        idxs_x=idxs_x,
        idxs_y=idxs_z,
        data=data,
        data_test=data_test,
        verbose=verbose,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
    )
    ptw_x_given_yz, avg_x_given_yz, _ = conditional_entropy(
        idxs_x=idxs_x,
        idxs_y=idxs_y + idxs_z,
        data=data,
        data_test=data_test,
        flow_kwargs=flow_kwargs,
        train_kwargs=train_kwargs,
        verbose=verbose,
    )

    ptw: torch.Tensor = ptw_x_given_z - ptw_x_given_yz
    avg: float = avg_x_given_z - avg_x_given_yz

    return ptw, avg
