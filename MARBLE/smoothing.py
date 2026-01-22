import torch


def scalar_diffusion(x, t, method="spectral", par=None):
    """
    Scalar diffusion: smooths scalar signals over the graph.
    Uses spectral formulation: exp(-t*Lambda) * U^T x.
    
    x : (n x 1) or (n x C)
    par: (eigenvalues, eigenvectors)
    """

    if len(x.shape) == 1:
        x = x.unsqueeze(1)

    # Only spectral method is supported in simplified version
    assert method == "spectral", "Only spectral method is supported in simplified version."
    assert isinstance(par, (list, tuple)) and len(par) == 2, "par must be (eigenvalues, eigenvectors)."

    evals, evecs = par

    # project into spectral domain
    x_spec = evecs.T @ x

    # apply diffusion coefficients exp(-t*lambda)
    diff_coef = torch.exp(-evals.unsqueeze(-1) * t)

    # diffuse in spectral domain
    x_diff = diff_coef * x_spec

    # project back to spatial domain
    return evecs @ x_diff


def vector_diffusion(x, t, Lc, L=None, method="spectral", normalise=False):
    """
    Vector diffusion simplified:
    - No parallel transport
    - No normalisation
    - Simply reshapes and applies scalar diffusion

    x: (n x d)
    """
    n, d = x.shape
    assert method == "spectral", "Only spectral method is supported."

    out = x.view(-1, 1)
    out = scalar_diffusion(out, t, method, Lc)
    return out.view(n, d)
