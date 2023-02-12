import torch
import numpy as np
from scipy.stats import norm
from .base import _BaseAttack


class BitFlipping(_BaseAttack):
    def __init__(self, config) -> None:
        self.n_peers = config.n_peers
        self.n_byzan = config.n_byzan
        self.n_attacking = config.n_attacking
        self.byzantines = np.arange(self.n_peers - self.n_byzan, self.n_peers)

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.byzantines:
                if len(attacking_peers) == self.n_attacking:
                    break
                grads[i] = -grads[i]
                attacking_peers.append(peers_to_aggregate[i])

        return attacking_peers

    def __str__(self):
        return "BitFlipping"


class IPM(_BaseAttack):
    def __init__(self, config) -> None:
        self.n_peers = config.n_peers
        self.n_byzan = config.n_byzan
        self.n_attacking = config.n_attacking
        self.byzantines = np.arange(self.n_peers - self.n_byzan, self.n_peers)
        self.ipm_epsilon = config.ipm_epsilon

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.byzantines:
                attacking_peers.append(peers_to_aggregate[i])
                if len(attacking_peers) == self.n_attacking:
                    break

        good_grads = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] not in attacking_peers:
                good_grads.append(grads[i])

        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in attacking_peers:
                grads[i] = -self.ipm_epsilon * (sum(good_grads)) \
                    / len(good_grads)

        return attacking_peers

    def __str__(self):
        return "InnerProductManipulation"


class RandomNoise(_BaseAttack):
    def __init__(self, config) -> None:
        self.n_attacking = config.n_attacking
        self.byzantines = np.arange(self.n_peers - self.n_byzan, self.n_peers)
        self.rn_sigma = config.rn_sigma

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        noise = None
        if len(grads) > 0:
            noise = torch.randn_like(grads[0])
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.byzantines:
                if len(attacking_peers) == self.n_attacking:
                    break
                attacking_peers.append(peers_to_aggregate[i])
                grads[i] = noise * self.rn_sigma
        return attacking_peers

    def __str__(self):
        return "RandomNoise"


class ALIE(_BaseAttack):
    def __init__(self, config) -> None:
        self.n_attacking = config.n_attacking
        self.byzantines = np.arange(config.n_peers - config.n_byzan, config.n_peers)
        self.rn_sigma = config.rn_sigma
        self.alie_z = config.alie_z

        if config.n_peers/2 < config.n_byzan:
            raise Exception("Byzantines ratio must be less than 0.5")

    def __call__(self, grads, peers_to_aggregate, not_banned_peers=None):
        attacking_peers = []
        for i in range(len(peers_to_aggregate)):
            if peers_to_aggregate[i] in self.byzantines:
                attacking_peers.append(peers_to_aggregate[i])
                if len(attacking_peers) == self.n_attacking:
                    break

        n_byzan = len(attacking_peers)
        if n_byzan == 0:
            return []
        n_peers = len(peers_to_aggregate)

        if self.alie_z is None:
            s = np.floor(n_peers / 2 + 1) - n_byzan
            cdf_value = (n_peers - n_byzan - s) / (n_peers - n_byzan)
            z_max = norm.ppf(cdf_value)
            # print(z_max, n_peers, n_byzan, s, (n_peers - n_byzan - s) / (n_peers - n_byzan))
        else:
            z_max = self.alie_z

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