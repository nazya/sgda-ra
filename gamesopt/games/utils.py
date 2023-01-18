import torch
import math
import sys


def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x


def generate_matrix(num_samples: int, dim: int,
                    mu: float, ell: float) -> torch.Tensor:
    A1 = torch.randn(dim, dim, requires_grad=False)
    e, V = torch.linalg.eigh(A1.T@A1)
    e -= e.min()
    e /= e.max()
    e *= ell - mu
    e += mu
    A1 = V @ torch.diag_embed(e) @ V.T
    A2 = torch.randn(dim, dim, requires_grad=False)
    A2 = A2.T@A2
    A3 = torch.randn(dim, dim, requires_grad=False)
    e, V = torch.linalg.eigh(A3.T@A3)
    e -= e.min()
    e /= e.max()
    e *= ell - mu
    e += mu
    A3 = V @ torch.diag_embed(e) @ V.T
    A12 = torch.cat((A1, A2), dim=0)
    A23 = torch.cat((-A2.T, A3), dim=0)
    A = torch.cat((A12, A23), dim=1)
    return A


def create_matrix(config):
    A1 = torch.randn(config.num_samples, config.dim, config.dim,
                     requires_grad=False)
    e, V = torch.linalg.eigh(A1@A1.transpose(1, 2))
    e -= torch.min(e, 1)[0][:, None]
    e /= torch.max(e, 1)[0][:, None]
    e *= config.ell - config.mu
    e += config.mu
    A1 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    A2 = torch.randn(config.num_samples, config.dim, config.dim,
                     requires_grad=False)
    A2 = A2@A2.transpose(1, 2)
    A3 = torch.randn(config.num_samples, config.dim, config.dim,
                     requires_grad=False)
    e, V = torch.linalg.eigh(A3@A3.transpose(1, 2))
    e -= torch.min(e, 1)[0][:, None]
    e /= torch.max(e, 1)[0][:, None]
    e *= config.ell - config.mu
    e += config.mu
    A3 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    A12 = torch.cat((A1, A2), dim=1)
    A23 = torch.cat((-A2.transpose(1, 2), A3), dim=1)
    A = torch.cat((A12, A23), dim=2)
    # s, _ = torch.linalg.eigh(A[0])
    # print('Matrix generated', s.min(), s.max())
    return A
