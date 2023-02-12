import torch
from abc import ABC
from pathlib import Path
from typing import Optional


class Game(ABC):
    def __init__(self, rank, config) -> None:
        self.master_node = 0
        self.num_samples = config.num_samples
        self.rank = rank
        self.size = config.n_peers

        self.p = torch.ones(self.num_samples) / self.num_samples

    def sample(self, n: int) -> torch.Tensor:
        if n > self.num_samples:
            raise Exception("Batch size should be not greater than the total number of samples")
        return torch.multinomial(self.p, n, replacement=False)

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def save(self, filename: Path) -> None:
        pass

    @staticmethod
    def load(filename: Path):
        pass