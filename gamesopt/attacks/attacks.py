import torch
# import torch.distributed as dist
import numpy as np
from scipy.stats import norm
from .base import _BaseAttack, AttackOptions


class BitFlipping(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options
        pass

    def __call__(self, grads, rank):
        if rank in self.options.byzan_index:
            grads[rank] = -grads[rank]

    def __str__(self):
        return "BitFlipping"


class ALIE(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options
        if options.n_total/2 < options.n_byzan:
            raise Exception("Byzantines ratio must be less than 0.5")
        if options.alie_z is not None:
            self.z_max = options.alie_z
        else:
            s = np.floor(options.n_total / 2 + 1) - options.n_byzan
            cdf_value = (options.n_total - options.n_byzan - s) \
                / (options.n_total - options.n_byzan)
            self.z_max = norm.ppf(cdf_value)
        self.n_good = options.n_total - options.n_byzan

    def __call__(self, grads, rank):
        if rank in self.options.byzan_index:
            good_grads = []
            for i in range(len(grads)):
                if i not in self.options.byzan_index:
                    good_grads.append(grads[i])
            stacked_gradients = torch.stack(good_grads, 1)
            mu = torch.mean(stacked_gradients, 1)
            std = torch.std(stacked_gradients, 1)
            grads[rank] = mu - std * self.z_max

    def __str__(self):
        return "ALittleIsEnough"


class IPM(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options

    def __call__(self, grads, rank):
        if rank in self.options.byzan_index:
            good_grads = []
            for i in range(len(grads)):
                if i not in self.options.byzan_index:
                    good_grads.append(grads[i])
            grads[rank] = -self.options.ipm_epsilon * (sum(good_grads)) \
                / len(good_grads)

    def __str__(self):
        return "InnerProductManipulation"


class RandomNoise(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options

    def __call__(self, grads, rank):
        if rank in self.options.byzan_index:
            grads[rank] = torch.randn_like(grads[rank]) * self.options.rn_sigma

    def __str__(self):
        return "RandomNoise"