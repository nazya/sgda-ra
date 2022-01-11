from abc import ABC, abstractmethod
from typing import Optional
from gamesopt.games import Game


class Optimizer(ABC):
    def __init__(self, game: Game) -> None:
        self.game = game

    @abstractmethod
    def step(self, index: Optional[int] = None) -> None:
        pass