from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from gamesopt.games import Game
from enum import Enum
from dataclasses import dataclass

# from .prox import Prox
from gamesopt.aggregator import AggregationOptions
from gamesopt.attacks import AttackOptions, load_attack
# from .lr import LRScheduler, FixedLR
import torch
import torch.distributed as dist


class OptimizerType(Enum):
    # PROX_SGDA = "Prox-SGDA"
    # PROX_LSVRGDA = "Prox-L-SVRGDAs"
    # SVRG = "SVRG"
    # VRFORB = "VR-FoRB"
    # VRAGDA = "VR-AGDA"
    # SVRE = "SVRE"
    # EG_VR = "EG-VR"
    # QSGDA = "QSGDA"
    # DIANA_SGDA = "DIANA-SGDA"
    # VR_DIANA_SGDA = "VR-DIANA-SGDA"
    SGDARA = "SGDARA"
    SGDACC = "SGDACC"


@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType
    lr: float
    batch_size: int
    aggregation_options: AggregationOptions
    attack_options: AttackOptions
    sigmaC: float

class Optimizer(ABC):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        self.game: Game = game
        self.options = options
        self.k = 0
        self.num_grad = 0
        self.lr = options.lr
        self.batch_size = options.batch_size
        self.attack_options = options.attack_options
        self.attack = load_attack(options.attack_options)

    def sample(self) -> Optional[torch.Tensor]:
        return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass

    # def fixed_point_check(self, precision: float = 1.) -> float:
    #     grad = self.game.full_operator()
    #     dist = 0
    #     for i in range(self.game.num_players):
    #         g = self.game.unflatten(i, grad)
    #         dist += ((self.game.players[i] - self.prox(self.game.players[i] \
    #                                     - precision*g, precision))**2).sum()
    #     return float(dist)


class DistributedOptimizer(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        super().__init__(game, options)
        self.size = int(dist.get_world_size())
        self.peers_to_aggregate = [i for i in range(self.size)]
        # self.n_bits = 0

    # def get_num_grad(self) -> int:
    #     num_grad = torch.tensor([self.num_grad])
    #     dist.all_reduce(num_grad)
    #     return int(num_grad)

    # def fixed_point_check(self, precision: float = 1., rank: int = 0) -> float:
    #     grad = self.game.full_operator()
    #     dist.reduce(grad, rank)
    #     grad /= self.size
    #     d = 0
    #     for i in range(self.game.num_players):
    #         g = self.game.unflatten(i, grad)
    #         d += ((self.game.players[i] - self.prox(self.game.players[i] - precision*g, precision))**2).sum()
    #     return float(d)