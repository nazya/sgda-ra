import torch
import math


def random_vector(dim: int) -> torch.Tensor:
    x = torch.zeros(dim).normal_() / math.sqrt(dim)
    x.requires_grad_()
    return x


def create_matrix(dim, num_samples, mu, ell, with_bias):
    A1 = torch.randn(num_samples, dim, dim,
                     requires_grad=False)
    e, V = torch.linalg.eigh(A1@A1.transpose(1, 2))
    e -= torch.min(e, 1)[0][:, None]
    e /= torch.max(e, 1)[0][:, None]
    e *= ell - mu
    e += mu
    A1 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    A2 = torch.randn(num_samples, dim, dim,
                     requires_grad=False)
    e, V = torch.linalg.eigh(A2@A2.transpose(1, 2))
    e /= torch.max(e, 1)[0][:, None]
    e *= ell
    A2 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    A3 = torch.randn(num_samples, dim, dim,
                     requires_grad=False)
    e, V = torch.linalg.eigh(A3@A3.transpose(1, 2))
    e -= torch.min(e, 1)[0][:, None]
    e /= torch.max(e, 1)[0][:, None]
    e *= ell - mu
    e += mu
    A3 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
    A12 = torch.cat((A1, A2), dim=1)
    A23 = torch.cat((-A2.transpose(1, 2), A3), dim=1)
    A = torch.cat((A12, A23), dim=2)
    return A.clone()


def create_bias(dim, num_samples, with_bias):
    bias = torch.zeros(num_samples, 2*dim)
    if with_bias:
        bias = 10*bias.normal_() / math.sqrt(2*dim)
    # s, _ = torch.linalg.eigh(A[0])
    # print('Matrix generated', s.min(), s.max())
    return bias.clone()
