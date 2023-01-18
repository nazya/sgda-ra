from pathlib import Path
from .base import Game
import torch
import math
from dataclasses import dataclass
from typing import Optional
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
    seed: int = 0
    matrix: Optional[torch.Tensor] = None
    # bias: Optional[torch.Tensor] = None


class QuadraticGame(Game):
    def __init__(self, config: QuadraticGameConfig) -> None:
        self.config = config
        # self._dim = config.dim
        super().__init__(config.num_samples)
        self.players = torch.ones(self.num_players*config.dim, requires_grad=False)
        self.dim = sum(p.numel() for p in self.players)
        torch.manual_seed(config.seed)
        # self.true = torch.zeros(1, self.num_players*config.dim, requires_grad=True)
        self.bias = torch.zeros(config.num_samples,
                                self.num_players*config.dim)
        self.matrix = torch.zeros(self.num_samples,
                                  self.num_players*config.dim,
                                  self.num_players*config.dim,
                                  requires_grad=False)

        if self.rank == self.master_node:
            A1 = torch.randn(config.num_samples, config.dim, config.dim,
                             requires_grad=False)
            e, V = torch.linalg.eigh(A1@A1.transpose(1, 2))
            e -= torch.min(e, 1)[0][:, None]
            e /= torch.max(e, 1)[0][:, None]
            e *= config.ell - config.mu
            e += config.mu
            A1 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
            A2 = torch.randn(config.num_samples, config.dim, config.dim,
                             requires_grad=False)
            # A2 = A2@A2.transpose(1, 2)
            # s = torch.linalg.eigvals(A2)
            # print(s.real.max(1))
            e, V = torch.linalg.eigh(A2@A2.transpose(1, 2))
            e /= torch.max(e, 1)[0][:, None]
            e *= config.ell
            A2 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
            A3 = torch.randn(config.num_samples, config.dim, config.dim,
                             requires_grad=False)
            e, V = torch.linalg.eigh(A3@A3.transpose(1, 2))
            e -= torch.min(e, 1)[0][:, None]
            e /= torch.max(e, 1)[0][:, None]
            e *= config.ell - config.mu
            e += config.mu
            A3 = V @ torch.diag_embed(e) @ V.transpose(1, 2)
            A12 = torch.cat((A1, A2), dim=1)
            A23 = torch.cat((-A2.transpose(1, 2), A3), dim=1)
            self.matrix = torch.cat((A12, A23), dim=2)
            mean_m = self.matrix.mean(dim=0)
            self.matrix = (1-config.sigma)*mean_m \
                + config.sigma*self.matrix
            print('Matrix generated')

            if config.bias:
                if self.rank == self.master_node:
                    self.bias = 10*self.bias.normal_() \
                        / math.sqrt(self.num_players*config.dim)
            mean_b = self.bias.mean(dim=0)
            self.bias = (1-config.sigma)*mean_b \
                + config.sigma*self.bias
            print('Bias generated')
            self.true = torch.linalg.solve(mean_m, -mean_b)
            print('Solution found')

        dist.broadcast(self.bias, src=self.master_node)
        dist.broadcast(self.matrix, src=self.master_node)
        # print(self.matrix)
        dist.broadcast(self.players, src=self.master_node)
        # dist.broadcast(self.true, src=self.master_node)

    def operator(self, index) -> torch.Tensor:
        grads = torch.matmul(self.matrix[index],
                             self.players) + self.bias[index]
                             # self.players.squeeze()) + self.bias[index]
        grads = grads.mean(dim=0)
        return grads

    def dist(self) -> float:
        dist = torch.linalg.norm(self.true - self.players)
        return float(dist)

    def hamiltonian(self) -> float:
        hamiltonian = torch.linalg.norm(self.true - self.players)
        index = self.sample_batch()
        grad = self.operator(index)
        grad /= self.size

        hamiltonian = (grad**2).sum()
        hamiltonian /= 2
        return float(hamiltonian)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        filename = "model"
        if self.rank is not None:
            filename += "_%i" % self.rank
        filename = path / ("%s.pth" % filename)

        torch.save({"config": self.config, "players": self.players,
                    "matrix": self.matrix, "bias": self.bias}, filename)

    def load(self, path: Path, copy: bool = False) -> Game:
        filename = "model"
        if self.rank is not None:
            filename += "_%i" % self.rank
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