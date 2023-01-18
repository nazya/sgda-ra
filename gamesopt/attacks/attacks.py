import torch
# import torch.distributed as dist
import numpy as np
from scipy.stats import norm
from .base import _BaseAttack, AttackOptions


class BitFlipping(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        self.options = options
        pass

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.options.byzantines:
                if len(attacking_peers) == self.options.n_attacking:
                    break
                grads[i] = -grads[i]
                attacking_peers.append(peers_to_aggregate[i])

        return attacking_peers

    def __str__(self):
        return "BitFlipping"


class IPM(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.options.byzantines:
                attacking_peers.append(peers_to_aggregate[i])
                if len(attacking_peers) == self.options.n_attacking:
                    break

        good_grads = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] not in attacking_peers:
                good_grads.append(grads[i])

        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in attacking_peers:
                grads[i] = -self.options.ipm_epsilon * (sum(good_grads)) \
                    / len(good_grads)

        return attacking_peers

    def __str__(self):
        return "InnerProductManipulation"


class RandomNoise(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.options.byzantines:
                if len(attacking_peers) == self.options.n_attacking:
                    break
                attacking_peers.append(peers_to_aggregate[i])
                grads[i] = torch.randn_like(grads[i]) * self.options.rn_sigma
        return attacking_peers

    def __str__(self):
        return "RandomNoise"


class ALIE(_BaseAttack):
    def __init__(self, options: AttackOptions) -> None:
        # self.rank = dist.get_rank()
        self.options = options
        if options.n_total/2 < options.n_byzan:
            raise Exception("Byzantines ratio must be less than 0.5")

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.options.byzantines:
                attacking_peers.append(peers_to_aggregate[i])
                if len(attacking_peers) == self.options.n_attacking:
                    break

        n_byzan = len(attacking_peers)
        if n_byzan == 0:
            return []
        n_total = len(peers_to_aggregate)

        if self.options.alie_z is None:
            s = np.floor(n_total / 2 + 1) - n_byzan
            cdf_value = (n_total - n_byzan - s) / (n_total - n_byzan)
            z_max = norm.ppf(cdf_value)
            # print(z_max, n_total, n_byzan, s, (n_total - n_byzan - s) / (n_total - n_byzan))
        else:
            z_max = self.options.alie_z

        good_grads = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] not in attacking_peers:
                good_grads.append(grads[i])
        stacked_gradients = torch.stack(good_grads, 1)
        mu = torch.mean(stacked_gradients, 1)
        std = torch.std(stacked_gradients, 1)

        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in attacking_peers:
                grads[i] = mu - std * z_max

        return attacking_peers

    def __str__(self):
        return "ALittleIsEnough"