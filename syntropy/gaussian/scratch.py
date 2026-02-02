def idep_partial_information_decomposition(
    inputs: tuple[int, ...],
    target: tuple[int, ...],
    data: NDArray[np.floating] | None = None,
    cov: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """
    Computes the I_dep partial information decomposition for Gaussian systems
    using the dependency constraint method from Kay & Ince (2018).
    
    Currently only supports 2 predictors (univariate or multivariate).
    
    Parameters
    ----------
    inputs : tuple[int, ...]
        The indices of the two predictor variables/sets.
        Must have length 2.
    target : tuple[int, ...]
        The indices of the target variable(s).
    data : NDArray[np.floating] | None
        The data in channels x samples format. Optional if cov provided.
    cov : NDArray[np.floating] | None
        The covariance matrix. If None, computed from data.
        
    Returns
    -------
    dict[str, float]
        Dictionary with keys: 'unq0', 'unq1', 'red', 'syn'
        
    References
    ----------
    Kay, J. W., & Ince, R. A. A. (2018). 
    Exact Partial Information Decompositions for Gaussian Systems 
    Based on Dependency Constraints. Entropy, 20(4), 240.
    https://doi.org/10.3390/e20040240
    """
    
    assert len(inputs) == 2, "I_dep currently only supports 2 predictors"
    assert cov is not None or data is not None, "Must provide either data or covariance matrix"
    
    # Get covariance matrix
    if cov is None:
        cov = np.cov(data, ddof=0)
    
    # Extract indices
    X0_idx = inputs[0] if isinstance(inputs[0], (list, tuple)) else (inputs[0],)
    X1_idx = inputs[1] if isinstance(inputs[1], (list, tuple)) else (inputs[1],)
    Y_idx = target if isinstance(target, (list, tuple)) else (target,)
    
    # Standardize to correlation matrix (Kay & Ince assume identity marginals)
    joint_idx = tuple(X0_idx) + tuple(X1_idx) + tuple(Y_idx)
    joint_cov = cov[np.ix_(joint_idx, joint_idx)]
    corr = covariance_to_correlation(joint_cov)
    
    # Extract P, Q, R matrices (cross-correlations)
    n0, n1, n2 = len(X0_idx), len(X1_idx), len(Y_idx)
    
    P = corr[:n0, n0:n0+n1]          # X0-X1 correlations
    Q = corr[:n0, n0+n1:]            # X0-Y correlations  
    R = corr[n0:n0+n1, n0+n1:]       # X1-Y correlations
    
    # Compute basic mutual informations using your functions
    # (on standardized correlation matrix)
    I_X0_Y = mutual_information(tuple(range(n0)), tuple(range(n0, n0+n2)), 
                                 corr[:n0+n2, :n0+n2])
    I_X1_Y = mutual_information(tuple(range(n0, n0+n1)), tuple(range(n0+n1, n0+n1+n2)),
                                 corr[n0:, n0:])
    I_X0X1_Y = mutual_information(tuple(range(n0+n1)), tuple(range(n0+n1, n0+n1+n2)), 
                                   corr)
    
    # Compute edge values from Table 9 (multivariate case)
    edge_b = I_X0_Y  # Adding X0Y to base model
    edge_d = I_X0_Y  # Adding X0Y to X0X1,Y model
    
    # Edge i: Adding X0Y to X1Y,X0 model (equation from Table 9)
    # i = (1/2)log(|I - RQ^TQR^T| / (|I-Q^TQ||I-R^TR|)) - I(X1;Y)
    edge_i = _compute_edge_i(Q, R, I_X1_Y)
    
    # Edge k: Adding X0Y to X0X1,X1Y model
    # k = I(X0,X1;Y) - I(X1;Y)
    edge_k = I_X0X1_Y - I_X1_Y
    
    # Unique information from X0 is minimum across all edges adding X0Y
    unq0 = min(edge_b, edge_d, edge_i, edge_k)
    
    # Derive other components (equations 61-62)
    red = I_X0_Y - unq0
    unq1 = I_X1_Y - red
    syn = I_X0X1_Y - I_X1_Y - unq0
    
    return {
        'unq0': unq0,
        'unq1': unq1, 
        'red': red,
        'syn': syn,
        # For debugging/analysis
        'I_X0_Y': I_X0_Y,
        'I_X1_Y': I_X1_Y,
        'I_X0X1_Y': I_X0X1_Y,
        'edges': {'b': edge_b, 'd': edge_d, 'i': edge_i, 'k': edge_k}
    }


def _compute_edge_i(
    Q: NDArray[np.floating],
    R: NDArray[np.floating],
    I_X1_Y: float
) -> float:
    """
    Compute edge value i from Table 9 in Kay & Ince (2018).
    
    This is the increase in mutual information when adding X0Y constraint
    to a model that already has X1Y and X0 marginals.
    
    Formula: i = (1/2)log(|I - RQ^TQR^T| / (|I-Q^TQ||I-R^TR|)) - I(X1;Y)
    
    Parameters
    ----------
    Q : NDArray[np.floating]
        Cross-correlation matrix between X0 and Y (n0 x n2)
    R : NDArray[np.floating]
        Cross-correlation matrix between X1 and Y (n1 x n2)
    I_X1_Y : float
        Mutual information between X1 and Y
        
    Returns
    -------
    float
        The edge value i
    """
    n1, n2 = R.shape
    I1 = np.eye(n1)
    I2 = np.eye(n2)
    
    # Compute |I - RQ^TQR^T|
    numerator_det = np.linalg.det(I1 - R @ Q.T @ Q @ R.T)
    
    # Compute |I - Q^TQ| 
    denom1_det = np.linalg.det(I2 - Q.T @ Q)
    
    # Compute |I - R^TR|
    denom2_det = np.linalg.det(I2 - R.T @ R)
    
    term = 0.5 * np.log(numerator_det / (denom1_det * denom2_det))
    
    return term - I_X1_Y


def covariance_to_correlation(cov: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Converts a non-standardized covariance matrix to a correlation matrix. 

    Parameters
    ----------
    cov : NDArray[np.floating]
        The covariance matrix.     

    Returns
    -------
    NDArray[np.floating]
        The correlation matrix. 
    """
    diag: NDArray[np.floating] = np.sqrt(np.diag(cov))
    d_inv: NDArray[np.floating] = np.diag(1.0 / diag)

    return d_inv @ cov @ d_inv
