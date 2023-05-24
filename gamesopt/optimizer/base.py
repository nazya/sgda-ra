import torch
from abc import ABC, abstractmethod
from typing import Optional
from enum import auto
from gamesopt import DictEnum
from gamesopt.attacks import load_attack
from gamesopt.games import load_game


class Optimizer(DictEnum):
    SGDARA = auto()
    MSGDARA = auto()
    SEGRA = auto()
    SGDACC = auto()
    SEGCC = auto()
    RDEG = auto()
    SGDA = auto()


class _OptimizerBase(ABC):
    def __init__(self, config, data, rank):
        self.rank = rank
        self.k = 0
        self.num_grad = 0
        self.lr = config.lr
        self.alpha = config.alpha
        self.n_peers = config.n_peers
        self.lr_inner = config.lr
        self.lr_outer = config.lr
        self.batch_size = config.batch_size
        self.attack = load_attack(config)
        self.game = load_game(config, data, rank, config.n_peers)

        self.peers_to_aggregate = [i for i in range(config.n_peers)]

    def sample(self) -> Optional[torch.Tensor]:
        return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass