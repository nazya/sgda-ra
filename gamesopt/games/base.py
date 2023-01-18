from abc import ABC
from pathlib import Path
import torch
import torch.distributed as dist


class Game(ABC):
    def __init__(self, num_samples: int) -> None:
        self.num_players = 2
        self.num_samples = num_samples
        self.master_node = 0
        self.rank = dist.get_rank()
        self.size = int(dist.get_world_size())

        self.p = torch.ones(self.num_samples) / self.num_samples

    def sample(self, n: int) -> torch.Tensor:
        if n > self.config.num_samples:
            raise Exception("Batch size should be not greater than the total number of samples")
        return torch.multinomial(self.p, n, replacement=False)

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def save(self, filename: Path) -> None:
        pass

    @staticmethod
    def load(filename: Path):
        pass