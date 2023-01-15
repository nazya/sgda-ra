from pathlib import Path
from .base import Game
from .utils import random_vector
import torch
import math
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from torch import linalg
import torch.distributed as dist


@dataclass
class QuadraticGameConfig:
    num_samples: int
    dim: int
    num_players: int
    bias: bool
    sigma: float
    mu: float
    ell: Optional[float]
    matrix: Optional[torch.Tensor] = None



class QuadraticGame(Game):
    def __init__(self, config: QuadraticGameConfig) -> None:
        self.config = config
        self._dim = config.dim
        players = [torch.zeros(self._dim, requires_grad=True),
                   torch.zeros(self._dim, requires_grad=True)]
        super().__init__(players, config.num_samples)
        torch.manual_seed(dist.get_rank())
        self.matrix = torch.randn(self.num_samples,
                                  self.num_players*config.dim,
                                  self.num_players*config.dim,
                                  requires_grad=False)

        for i in range(self.num_samples):
            self.matrix[i] = self.generate_matrix(self.num_samples,
                                                  self._dim, config.mu,
                                                  config.ell)

        mean = self.matrix.mean(dim=0)
        for i in range(self.num_samples):
            self.matrix[i] = (1-config.sigma)*mean \
                + config.sigma*self.matrix[i]

        # print('Matrix generated')
        # print(dist.get_rank(), self.matrix)
        self.bias = torch.zeros(2, config.num_samples, config.dim)
        if config.bias:
            if self.rank == self.master_node:
                self.bias = 10*self.bias.normal_() / math.sqrt(self._dim)
            dist.broadcast(self.bias, src=self.master_node)

        mean = self.bias.mean(dim=1)
        for i in range(self.num_samples):
            self.bias[:, i, :] = (1 - config.sigma) * mean \
                + config.sigma*self.bias[:, i, :]

        for i in range(self.num_players):
            dist.broadcast(self.players[i].data, src=self.master_node)

    def generate_matrix(self, num_samples: int, dim: int, mu: float, ell: float) -> torch.Tensor:
        A1 = torch.randn(dim, dim, requires_grad=False)
        dist.broadcast(A1, src=self.master_node)
        e, V = torch.linalg.eig(A1.T@A1)
        e -= e.real.min()
        e /= e.real.max()
        e *= ell - mu
        e += mu
        A1 = V @ torch.diag_embed(e) @ torch.linalg.inv(V)
        A2 = torch.randn(dim, dim, requires_grad=False)
        dist.broadcast(A2, src=self.master_node)
        A2 = A2.T@A2
        A3 = torch.randn(dim, dim, requires_grad=False)
        dist.broadcast(A3, src=self.master_node)
        e, V = torch.linalg.eig(A3.T@A3)
        e -= e.real.min()
        e /= e.real.max()
        e *= ell - mu
        e += mu
        A3 = V @ torch.diag_embed(e) @ torch.linalg.inv(V)
        A12 = torch.cat((A1, A2), dim=0)
        A23 = torch.cat((-A2, A3), dim=0)
        A = torch.cat((A12, A23), dim=1)
        return A.real
        # print('a', A)
        # s = torch.linalg.eigvals(A)
        # print('Matrix generated', s)

    def sample(self, n: int) -> torch.Tensor:
        if n > self.config.num_samples:
            raise Exception("Batch size should be not greater than the total number of samples")
        return torch.multinomial(self.p, n, replacement=False)

    def sample_batch(self) -> torch.Tensor:
        return torch.arange(self.num_samples).long()

    def loss(self, index: int) -> torch.Tensor:
        loss = []
        for i in range(self.num_players):
            _loss = self.bias[i, index]
            for j in range(self.num_players):
                _loss += (self.matrix[index, i*self._dim:(i+1)*self._dim,
                                      j*self._dim:(j+1)*self._dim]
                          * self.players[j].view(1, 1, -1)).sum(-1)
            _loss = (_loss*self.players[i].view(1, -1)).sum(-1).mean()

            loss.append(_loss)
        return loss


    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        filename = "model"
        if self.rank is not None:
            filename += "_%i"%self.rank
        filename = path / ("%s.pth" % filename)

        torch.save({"config": self.config, "players": self.players, "matrix": self.matrix, "bias": self.bias}, filename)

    def load(self, path: Path, copy: bool = False) -> Game:
        filename = "model"
        if self.rank is not None:
            filename += "_%i"%self.rank
        filename = path / ("%s.pth" % filename)

        checkpoint = torch.load(filename)
        self.matrix = checkpoint["matrix"]
        self.bias = checkpoint["bias"]

        if copy:
            game = self.copy()
            game.players = checkpoint["players"]
            return game
        else:
            self.players = checkpoint["players"]
            return self