import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import nflows
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation


def initialize_flow(
    dim: int,
    dim_context: int = 0,
    num_layers: int = 5,
    hidden_features: int = 64,
    dropout_probability: float = 0.1,
) -> nflows.flows.base.Flow:
    """
    Initializes a new normalizing flow network.

    Parameters
    ----------
    dim : int
        The number of input dimensions.
    dim_context : int
        The number of conditioning dimension.
        The default is zero.
    num_layers : int
        The number of hidden layers.
        The default is 5 layers.
    hidden_features : int
        The number of neurons in each hidden layer.
        The default is 64 neurons.
    dropout_probability : int
        The probability of a neuron dropping out.

    Returns
    -------
        nflows.flows.base.Flow
    """
    transforms: list = []

    for _ in range(num_layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=dim_context,
                dropout_probability=dropout_probability,
            )
        )
        transforms.append(RandomPermutation(features=dim))

    transform = CompositeTransform(transforms)
    base_dist = StandardNormal(shape=[dim])

    return Flow(transform, base_dist)


def train_flow(
    flow: nflows.flows.base.Flow,
    data: torch.Tensor,
    context: None | torch.Tensor = None,
    batch_size: int = 256,
    lr: float = 1e-4,
    num_epochs: int = 100,
    weight_decay: float = 1e-5,
    convergence_threshold: float = 0.0,
    alpha: float = 0.1,
    verbose: bool = False,
) -> nflows.flows.base.Flow:
    """
    Trains a normalizing flow network to approximate the maximally likely distribution to have generated the given data.

    Parameters
    ----------
    flow : nflows.flows.base.Flow
        An untrained normalizing flow network.
    data : torch.Tensor
        The training data, in samples x features format.
    context : None | torch.Tensor
        Conditioning random variables.
        Default is None.
    batch_size : int
        The size of each batch.
        The default value is 256.
    lr : float
        The learning rate.
        The default value is 1e-4
    num_epochs : int
        The number of training epochs.
        The default value is 100,
    weight_decay: float
        The rate at which parameter weights decay. A regularizer to reduce over-fitting.
        The default is 1e-5.
    convergence_threshold : float
        The value of the coefficient of variation below which the training terminates.
        The default value is 0.0, which means the training will not stop before num_epochs is hit.
    alpha : float
        How quickly the exponentially weighted moving standard deviation downweights older data.
        The default is 0.2.
    verbose : bool
        Whether to print the loss for each epoch throughout training.
        The default is False.


    Returns
    -------

    nflows.flows.base.Flow
    """

    if context is None:
        dataset = TensorDataset(data)
    elif context is not None:
        dataset = TensorDataset(data, context)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    flow.train()

    running_mean: float = 0.0
    running_var: float = 0.0

    for epoch in range(num_epochs):
        epoch_nll_sum: float = 0

        for batch in loader:
            optimizer.zero_grad()

            if context is None:
                x = batch[0]
                batch_nll = -flow.log_prob(x).mean()
            else:
                x, c = batch
                batch_nll = -flow.log_prob(x, context=c).mean()
            batch_nll.backward()

            optimizer.step()

            epoch_nll_sum += batch_nll.item() * x.shape[0]

        epoch_avg_nll: float = epoch_nll_sum / data.shape[0]

        delta = epoch_avg_nll - running_mean
        running_mean += alpha * delta
        running_var = (1 - alpha) * (running_var + alpha * delta**2)
        running_std = running_var ** (1 / 2)
        coef_var = running_std / (running_mean + 1e-8)

        if verbose is True:
            print(
                f"Epoch {epoch + 1}, NLL: {epoch_avg_nll:.3f}, Coef. var: {coef_var:.3f}"
            )

        if coef_var < convergence_threshold:
            return flow

    return flow


def evaluate_flow(
    flow: nflows.flows.base.Flow,
    data: torch.Tensor,
    context: None | torch.Tensor = None,
) -> tuple[float, float]:
    """
    Evaluates a trainied normalizing flow network on the given data.

    Parameters
    ----------
    flow : nflows.flows.base.Flow
        A trained normalilzing flow network.
    data : torch.Tensor
        The testing data in samples x features format.
    context : None | torch.Tensor


    Returns
    -------
    tuple[float, float]
        The average entropy (in nat)
        The standard error of the estimate.

    """

    flow.eval()

    with torch.no_grad():
        if context is None:
            log_probs = flow.log_prob(data)
        else:
            log_probs = flow.log_prob(data, context=context)
        h = -log_probs.mean().item()

    return h
