from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from gamesopt.attacks import AttackOptions, load_attack
import torch
import torch.distributed as dist
from gamesopt.aggregator import AggregationOptions, load_bucketing
from gamesopt.games import Game


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
    MSGDARA = "MSGDARA"
    SEGDARA = 'SEGDARA'
    SGDACC = "SGDACC"


@dataclass
class OptimizerOptions:
    optimizer_type: OptimizerType
    lr: float
    alpha: float
    batch_size: int
    aggregation_options: AggregationOptions
    attack_options: AttackOptions
    sigmaC: float
    lr_inner: float
    lr_outer: float

class Optimizer(ABC):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        self.game: Game = game
        self.options = options
        self.k = 0
        self.num_grad = 0
        self.lr = options.lr
        self.alpha = options.alpha
        self.lr_inner = options.lr_inner
        self.lr_outer = options.lr_outer
        self.batch_size = options.batch_size
        self.attack_options = options.attack_options
        self.attack = load_attack(options.attack_options)

    def sample(self) -> Optional[torch.Tensor]:
        return self.game.sample(self.batch_size)

    @abstractmethod
    def step(self) -> None:
        pass


class DistributedOptimizer(Optimizer):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        super().__init__(game, options)
        self.size = int(dist.get_world_size())
        self.peers_to_aggregate = [i for i in range(self.size)]