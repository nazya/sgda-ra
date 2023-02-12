import torch
import torch.distributed as dist
from copy import deepcopy
from .base import Optimizer
from gamesopt.aggregator import AggregatorType, load_aggregator, load_bucketing


class SGDARA(Optimizer):
    def __init__(self, config, data, rank):
        super().__init__(config, data, rank)
        self.aggregator = load_bucketing(config)

    def step(self) -> None:
        index = self.sample()
        # index = torch.sort(index).values
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # attack
                grads = [grads[i] for i in self.peers_to_aggregate]
                self.attack(grads, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(grads)
                self.game.players.data = self.game.players - self.lr * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

        # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1


class MSGDARA(Optimizer):
    def __init__(self, config, data, rank) -> None:
        super().__init__(config, data, rank)
        self.aggregator = load_bucketing(config)
        self.alpha = config.alpha
        self.momentum = None

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()

        if self.k == 0:
            self.momentum = grad.clone()
        self.momentum = (1-self.alpha)*self.momentum+self.alpha*grad
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                momentums = [torch.empty_like(self.momentum)
                             for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=momentums)

                # attack
                momentums = [momentums[i] for i in self.peers_to_aggregate]
                self.attack(momentums, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(momentums)
                self.game.players.data = self.game.players - self.lr * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=self.momentum, dst=self.game.master_node)

        # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1


class SEGRA(Optimizer):
    def __init__(self, config, data, rank) -> None:
        super().__init__(config, data, rank)
        self.aggregator = load_bucketing(config)
        self.lr_inner = config.lr_inner
        self.lr_outer = config.lr_outer

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        data_copy = self.game.players.data.clone().detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # attack
                grads = [grads[i] for i in self.peers_to_aggregate]
                self.attack(grads, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(grads)
                self.game.players.data = self.game.players - self.lr_inner * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)
        # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)

        index = self.sample()
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # attack
                grads = [grads[i] for i in self.peers_to_aggregate]
                self.attack(grads, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(grads)
                self.game.players.data = data_copy - self.lr_outer * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

        # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1


class RDEG(Optimizer):
    def __init__(self, config, data, rank) -> None:
        super().__init__(config, data, rank)
        aggregation_config = deepcopy(config)
        aggregation_config.aggregator_type = AggregatorType.UnivariateTM
        self.aggregator = load_aggregator(aggregation_config)
        self.lr_inner = config.lr_inner
        self.lr_outer = config.lr_outer

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        data_copy = self.game.players.data.detach().clone()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # attack
                grads = [grads[i] for i in self.peers_to_aggregate]
                self.attack(grads, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(grads)
                self.game.players.data = self.game.players \
                    - self.lr_inner * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)
        # broadcast new point
        dist.barrier()
        dist.broadcast(self.game.players, src=self.game.master_node)

        index = self.sample()
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # attack
                grads = [grads[i] for i in self.peers_to_aggregate]
                self.attack(grads, self.peers_to_aggregate)

                # aggregation
                agg_grad = self.aggregator(grads)
                self.game.players.data = data_copy - self.lr_outer * agg_grad
                self.num_grad += len(index)*len(self.peers_to_aggregate)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

        # broadcast new point
        dist.barrier()
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1


class SGDACC(Optimizer):
    def __init__(self, config, data, rank):
        super().__init__(config, data, rank)
        aggregation_config = deepcopy(config)
        aggregation_config.aggregator_type = AggregatorType.Mean
        self.aggregator = load_aggregator(aggregation_config)
        self.sigmaC = config.sigmaC

        self.checking = []
        self.peers_to_ban = []
        self.not_banned_peers = self.peers_to_aggregate.copy()
        self.n_cc = 1  # n_cc = 1 tested only

    def set_peers_to_aggregate(self):
        self.peers_to_aggregate = []
        for peer in self.not_banned_peers:
            if peer not in self.checking:
                self.peers_to_aggregate.append(peer)

    def ban(self):
        if self.n_cc == 0:
            return

        for peer in self.peers_to_ban:
            self.not_banned_peers.remove(peer)
        self.peers_to_ban = []
        self.set_peers_to_aggregate()
        if len(self.peers_to_aggregate) == 0:
            print('aggregate nothing')
            if self.n_cc > 1:
                self.n_cc -= 1

    def simulate_cc(self, attacked_peers):
        if self.n_cc == 0:
            return 0

        n = len(self.not_banned_peers)
        if n <= 2:
            self.checking = []
            self.n_cc = 0
            return 0

        p = torch.ones(n) / n
        indices = torch.multinomial(p, 2*self.n_cc, replacement=False)
        self.checking = []
        checking = [self.not_banned_peers[int(idx)] for idx in indices]
        # self.checking = checking.copy()
        num_grads = 0
        for i in range(self.n_cc):
            self.checking.append(checking[2*i])
            if checking[2*i] not in self.attack.byzantines:
                num_grads += 1
                if checking[2*i+1] in attacked_peers:
                    self.peers_to_ban.append(checking[2*i])
                    self.peers_to_ban.append(checking[2*i+1])
                    # print(checking[2*i+1], ' will be banned by ', checking[2*i])
        return num_grads

    def step(self) -> None:
        index = self.sample()
        # index = torch.sort(index).values
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # ban Bizantines
                self.ban()
                # print(self.not_banned_peers)

                # cut off banned and checking peers
                grads = [grads[i] for i in self.peers_to_aggregate]
                attacked_peers = self.attack(grads, self.peers_to_aggregate,
                                             self.not_banned_peers)
                # print(attacked_peers, ' attacked')

                # simulate check of computantions
                n_grads_checked = self.simulate_cc(attacked_peers)

                agg_grad = self.aggregator(grads)  # mean only
                verification_fail = False
                for grad in grads:
                    if torch.linalg.norm(grad - agg_grad) > self.sigmaC:
                        verification_fail = True
                        break
                if not verification_fail:
                    self.game.players = self.game.players - self.lr * agg_grad

                self.num_grad += len(index)*(len(self.peers_to_aggregate)
                                             - len(attacked_peers)
                                             + n_grads_checked)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

            # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1


class SEGCC(Optimizer):
    def __init__(self, config, data, rank):
        super().__init__(config, data, rank)
        aggregation_config = deepcopy(config)
        aggregation_config.aggregator_type = AggregatorType.Mean
        self.aggregator = load_aggregator(aggregation_config)
        self.sigmaC = config.sigmaC

        self.checking = []
        self.peers_to_ban = []
        self.not_banned_peers = self.peers_to_aggregate.copy()
        self.n_cc = 1  # n_cc = 1 tested only

    def set_peers_to_aggregate(self):
        self.peers_to_aggregate = []
        for peer in self.not_banned_peers:
            if peer not in self.checking:
                self.peers_to_aggregate.append(peer)

    def ban(self):
        if self.n_cc == 0:
            return

        for peer in self.peers_to_ban:
            self.not_banned_peers.remove(peer)
        self.peers_to_ban = []
        self.set_peers_to_aggregate()
        if len(self.peers_to_aggregate) == 0:
            print('aggregate nothing')
            if self.n_cc > 1:
                self.n_cc -= 1

    def simulate_cc(self, attacked_peers):
        if self.n_cc == 0:
            return 0

        n = len(self.not_banned_peers)
        if n <= 2:
            self.checking = []
            self.n_cc = 0
            return 0

        p = torch.ones(n) / n
        indices = torch.multinomial(p, 2*self.n_cc, replacement=False)
        self.checking = []
        checking = [self.not_banned_peers[int(idx)] for idx in indices]
        # self.checking = checking.copy()
        num_grads = 0
        for i in range(self.n_cc):
            self.checking.append(checking[2*i])
            if checking[2*i] not in self.attack.byzantines:
                num_grads += 1
                if checking[2*i+1] in attacked_peers:
                    self.peers_to_ban.append(checking[2*i])
                    self.peers_to_ban.append(checking[2*i+1])
                    # print(checking[2*i+1], ' will be banned by ', checking[2*i])
        return num_grads

    def step(self) -> None:
        index = self.sample()
        grad = self.game.operator(index).detach()
        data_copy = self.game.players.data.clone().detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # ban Bizantines
                self.ban()
                # print(self.not_banned_peers)

                # cut off banned and checking peers
                grads = [grads[i] for i in self.peers_to_aggregate]
                attacked_peers = self.attack(grads, self.peers_to_aggregate,
                                             self.not_banned_peers)
                # print(attacked_peers, ' attacked')

                # simulate check of computantions
                n_grads_checked = self.simulate_cc(attacked_peers)

                agg_grad = self.aggregator(grads)  # mean only
                verification_fail = False
                for grad in grads:
                    if torch.linalg.norm(grad - agg_grad) > self.sigmaC:
                        verification_fail = True
                        break
                if not verification_fail:
                    self.game.players = self.game.players - self.lr_inner * agg_grad

                self.num_grad += len(index)*(len(self.peers_to_aggregate)
                                             - len(attacked_peers)
                                             + n_grads_checked)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

        index = self.sample()
        grad = self.game.operator(index).detach()
        with torch.no_grad():
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.n_peers)]
                dist.gather(grad, gather_list=grads)

                # ban Bizantines
                self.ban()
                # print(self.not_banned_peers)

                # cut off banned and checking peers
                grads = [grads[i] for i in self.peers_to_aggregate]
                attacked_peers = self.attack(grads, self.peers_to_aggregate,
                                             self.not_banned_peers)
                # print(attacked_peers, ' attacked')

                # simulate check of computantions
                n_grads_checked = self.simulate_cc(attacked_peers)

                agg_grad = self.aggregator(grads)  # mean only
                verification_fail = False
                for grad in grads:
                    if torch.linalg.norm(grad - agg_grad) > self.sigmaC:
                        verification_fail = True
                        break
                if not verification_fail:
                    self.game.players = data_copy - self.lr_outer * agg_grad

                self.num_grad += len(index)*(len(self.peers_to_aggregate)
                                             - len(attacked_peers)
                                             + n_grads_checked)
            else:
                dist.gather(tensor=grad, dst=self.game.master_node)

            # broadcast new point
        dist.broadcast(self.game.players, src=self.game.master_node)
        self.k += 1