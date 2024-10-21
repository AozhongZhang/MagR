import torch

# <script src="https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55.js"></script>

def project_onto_l1_ball(x, eps=1.0):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

      min ||x - u||_2 s.t. ||u||_1 <= eps

    Inspired by the corresponding numpy version by Adrien Gaidon.

    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU

    eps: float
      radius of l-1 ball to project onto

    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original

    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.

    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def linfty_proximal(x, scale):
    '''
    the proximal operator of l_infinity norm:

    Prox_{scale * |.|_\infty}(x) = x - scale * project_onto_l1_ball(x/scale)

    parameters
    ------------
    x: (batch_size, *) torch array
    batch of arbitrary-size tensors to project, possibly on GPU

    scale: float 
    the scale for the proximal operator:

    returns
    -------------
    the proximal operator on x: (batch_size, *) torch array
    batch of proximal operator applied tensors, reshaped to match the original
    '''
    assert scale != 0
    return x - scale * project_onto_l1_ball(x / scale)

# this is one sample
def W_proximal_preprocess(W, X, device, alpha=0.001, n_iter=200):
    
    W_hat = W.clone().T

    m, n = X.shape

    U, s, Vt = torch.linalg.svd(X)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    for _ in range(n_iter):
        W_hat = linfty_proximal(
            (W_hat - torch.matmul(XtX, W_hat-W.T)).T, alpha).T

    del XtX
    return W_hat.T




#-------------------Proximal_groupwise---------------------------  

def project_onto_l1_ball_groupwise(x, eps=1.0):
    """
    Compute Euclidean projection onto the L1 ball for a batch.

    Parameters:
    x: (batch_size, num_groups, group_size) torch array
      batch of grouped tensors to project, possibly on GPU

    eps: float
      radius of the L-1 ball to project onto

    Returns:
    u: (batch_size, num_groups, group_size) torch array
      batch of projected tensors, reshaped to match the original
    """
    # Flattening within each group but keeping batch and group separations
    batch_size, num_groups, group_size = x.shape
    x = x.view(batch_size * num_groups, group_size)
    
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, group_size + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(batch_size * num_groups), rho - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    
    # Reshape back to the original grouped shape
    return x.view(batch_size, num_groups, group_size)


def linfty_proximal_groupwise(x, scale, group_size=128):
    """
    The proximal operator of L-infinity norm applied groupwise.

    Parameters:
    x: (batch_size, num_features) torch array
      Batch of arbitrary-size tensors to project, possibly on GPU

    scale: float
      The scale for the proximal operator.

    group_size: int
      The size of each group to apply the proximal operation.

    Returns:
    The proximal operator on x: (batch_size, num_features) torch array
      Batch of proximal operator applied tensors, reshaped to match the original
    """
    assert scale != 0

    # Reshape x to have groups of `group_size`
    num_features = x.shape[1]
    
    if num_features % group_size != 0:
        raise ValueError("The number of features must be divisible by the group size.")
    
    num_groups = num_features // group_size
    
    x = x.view(-1, num_groups, group_size)

    # Apply the projection for each group
    proximal_result = x - scale * project_onto_l1_ball_groupwise(x / scale)
    
    # Reshape back to the original shape
    return proximal_result.view(-1, num_features)


def W_proximal_preprocess_groupwise(W, X, device, alpha=0.0001, n_iter=200, group_size=128):

    W_hat = W.clone().T

    m, n = X.shape

    U, s, Vt = torch.linalg.svd(X, full_matrices=False)
    del X
    s /= torch.max(s)
    S = torch.diag(s)
    if m > n:
        U = U[:, :n]
    elif m < n:
        Vt = Vt[:m, :]

    X = torch.mm(torch.mm(U, S), Vt)
    XtX = torch.matmul(X.T, X).to(device)

    for _ in range(n_iter):
        W_hat = linfty_proximal_groupwise(
            (W_hat - torch.matmul(XtX, W_hat-W.T)).T, scale=alpha, group_size=group_size).T

    del XtX
    return W_hat.T


