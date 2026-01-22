import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing

import MARBLE.smoothing as s


class Diffusion(nn.Module):
    """
    Diffusion layer with learnable time parameter t.
    Only scalar spectral diffusion is supported.
    """

    def __init__(self, tau0=0.0):
        super().__init__()
        self.diffusion_time = nn.Parameter(torch.tensor(float(tau0)))

    def forward(self, x, L, method="spectral"):
        """
        x: node features
        L: (eigenvalues, eigenvectors)
        """
        # ensure positive diffusion time
        with torch.no_grad():
            self.diffusion_time.data.clamp_(min=1e-8)

        t = self.diffusion_time

        # diffuse each channel independently
        out = [s.scalar_diffusion(x_, t, method, L) for x_ in x.T]
        return torch.cat(out, dim=1)


class AnisoConv(MessagePassing):
    """
    Computes anisotropic convolution (directional derivatives).
    This layer extracts geometric gradient information from data.
    """

    def __init__(self, **kwargs):
        super().__init__(aggr="add", **kwargs)

    def forward(self, x, kernels):
        """
        x: node features
        kernels: directional kernels from preprocessing
        """
        outs = [self.propagate(K, x=x) for K in kernels]
        outs = torch.stack(outs, dim=2)  # (n, F, dirs)
        outs = outs.view(outs.size(0), -1)
        return outs

    def message_and_aggregate(self, K_t, x):
        """
        Performs K_t @ x (matrix multiplication on graph)
        """
        return K_t.matmul(x, reduce=self.aggr)


class InnerProductFeatures(nn.Module):
    """
    Computes inner products between channels to produce scalar features.
    MLP consumes scalar features more easily.
    """

    def __init__(self, C, D):
        super().__init__()
        self.C, self.D = C, D
        self.O_mat = nn.ModuleList([nn.Linear(D, D, bias=False) for _ in range(C)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.O_mat:
            layer.weight.data = torch.eye(self.D)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        x = [x_.view(x_.shape[0], -1, self.D) for x_ in x]

        if self.D == 1:
            # for scalar signals output magnitudes
            x = [x_.norm(dim=2) for x_ in x]
            return torch.cat(x, dim=1)

        x = [x_.swapaxes(1, 2) for x_ in x]
        x = torch.cat(x, dim=2)

        assert x.shape[2] == self.C, "Invalid number of channels"

        Ox = torch.stack([self.O_mat[j](x[..., j]) for j in range(self.C)], dim=2)
        xOx = torch.einsum("bki,bkj->bi", x, Ox)

        return torch.tanh(xOx).view(x.shape[0], -1)
