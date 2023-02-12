import torch
from .base import Game


class QuadraticGame(Game):
    def __init__(self, rank, config, data):
        super().__init__(rank, config)

        if rank is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(rank)

        self.matrix = data.matrix
        self.bias = data.bias
        self.players = data.players.clone()
        self.true = data.true

        self.dim = sum(p.numel() for p in self.players)

    def operator(self, index) -> torch.Tensor:
        grad = 0.
        for i in index:
            grad += torch.matmul(self.matrix[i], self.players) + self.bias[i]
        return grad/len(index)

        grads = torch.matmul(self.matrix, self.players) + self.bias
        grads = torch.index_select(grads, 0, index).mean(dim=0)
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