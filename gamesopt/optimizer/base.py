import torch
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
from gamesopt.attacks import load_attack
from gamesopt.games import load_game


class OptimizerType(Enum):
    SGDARA = '>'
    MSGDARA = '<'
    SEGRA = 'v'
    SGDACC = 's'
    SEGCC = 'D'
    RDEG = '*'


class Optimizer(ABC):
    def __init__(self, config, data, rank):
        self.rank = rank
        self.k = 0
        self.num_grad = 0
        self.lr = config.lr
        self.alpha = config.alpha
        self.n_peers = config.n_peers
        self.lr_inner = config.lr_inner
        self.lr_outer = config.lr_outer
        self.batch_size = config.batch_size
        self.attack = load_attack(config)
        self.game = load_game(config, data, rank, config.n_peers)

        self.peers_to_aggregate = [i for i in range(config.n_peers)]

    def sample(self) -> Optional[torch.Tensor]:
        return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass