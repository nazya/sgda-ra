import torch
import sys
import torch.distributed as dist
from .base import DistributedOptimizer, OptimizerOptions
<<<<<<< HEAD
from gamesopt.games import Game
=======
from gamesopt.aggregator import AggregatorType, load_aggregator, load_bucketing
from gamesopt.games import Game

>>>>>>> 246dfe4... speed up

class SGDARA(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions):
        super().__init__(game, options)
        self.aggregator = load_bucketing(self.options.aggregation_options)

    def step(self) -> None:
        index = self.sample()
        # index = torch.sort(index).values
        grad = self.game.operator(index).detach()
<<<<<<< HEAD

        # if self.k == 5:
        #     print(str(self.k)+ ' ' + str(self.game.rank)
        #           # +' '+str(grad) + ' '
        #           + str(self.game.players[0])+ ' '
        #           + str(self.game.players[1]) + ' '
        #           + str(index))

        with torch.no_grad():
            # collect all the gradients to simulate Byzantinesf
            grads = [torch.empty_like(grad) for _ in range(self.size)]
            dist.all_gather(tensor=grad, tensor_list=grads)
=======
        with torch.no_grad():
>>>>>>> 246dfe4... speed up
            # server
            if self.game.rank == self.game.master_node:
                # collect all the gradients to simulate Byzantines
                grads = [torch.empty_like(grad) for _ in range(self.size)]
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
<<<<<<< HEAD
        self.num_grad += len(index)

class MSGDARA(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
            super().__init__(game, options)
            self.alpha = options.alpha
            self.mo = None

    def step(self) -> None:
        index = self.sample()
        index = torch.sort(index).values
        grad = self.game.operator(index).detach()

        if self.k==0:
            self.mo=grad.clone().detach()
        self.mo  = (1-self.alpha)*self.mo+self.alpha*grad

        # if self.game.rank==1:
        #     print('m',mo[1],'g',grad[1])

        with torch.no_grad():
            # collect all the gradients to simulate Byzantinesf
            mos = [torch.empty_like(self.mo) for _ in range(self.size)]
            dist.all_gather(tensor=self.mo, tensor_list=mos)

            # server
            if self.game.rank == self.game.master_node:
                # attack
                for rank in range(self.size):
                    self.attack(mos, rank)
                    # dist.all_gather(tensor=grad, tensor_list=grads)

                # aggregation
                agg_grad = self.aggregator(mos)
                for i in range(self.game.num_players):
                    g = self.game.unflatten(i, agg_grad)
                    self.game.players[i].data = self.game.players[i] - self.lr*g

            # broadcast new point
            for i in range(self.game.num_players):
                dist.broadcast(self.game.players[i], src=self.game.master_node)

        self.k += 1
        self.num_grad += len(index)

class SEGDARA(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions) -> None:
        super().__init__(game, options)
        self.p = 1 / game.num_samples
        self.N = torch.tensor([0])
        self.set_state()

    def set_state(self) -> None:
        self.game_copy = self.game.copy()
        self.full_grad = self.game_copy.full_operator().detach()
        self.N.geometric_(self.p)
        # self.num_grad += self.game.num_samples

    def step(self) -> None:
        index = self.sample()
        # game_copy = self.game.copy()
        grad = self.game.operator(index).detach()
        # grad_copy = self.game_copy.operator(index).detach()
        # update = grad - grad_copy + self.full_grad
        with torch.no_grad():
            # collect all the gradients to simulate Byzantinesf
            updates = [torch.empty_like(grad) for _ in range(self.size)]
            dist.all_gather(tensor=grad, tensor_list=updates)

            # server
            if self.game.rank == self.game.master_node:
                # attack
                for rank in range(self.size):
                    self.attack(updates, rank)
                    # dist.all_gather(tensor=grad, tensor_list=grads)

                # aggregation
                agg_grad = self.aggregator(updates)
                for i in range(self.game.num_players):
                    g = self.game.unflatten(i, agg_grad)
                    self.game.players[i].data = self.game.players[i] - self.lr*g

            # broadcast new point
            for i in range(self.game.num_players):
                dist.broadcast(self.game.players[i], src=self.game.master_node)

        # self.num_grad += 2*len(index)

        index = self.sample()
        grad = self.game.operator(index).detach()
        # grad_copy = self.game_copy.operator(index).detach()
        # update = grad - grad_copy + self.full_grad
        with torch.no_grad():
            # collect all the gradients to simulate Byzantinesf
            updates = [torch.empty_like(grad) for _ in range(self.size)]
            dist.all_gather(tensor=grad, tensor_list=updates)

            # server
            if self.game.rank == self.game.master_node:
                # attack
                for rank in range(self.size):
                    self.attack(updates, rank)
                    # dist.all_gather(tensor=grad, tensor_list=grads)

                # aggregation
                agg_grad = self.aggregator(updates)
                for i in range(self.game.num_players):
                    g = self.game.unflatten(i, agg_grad)
                    self.game.players[i].data = self.game.players[i] - self.lr*g

            # broadcast new point
            for i in range(self.game.num_players):
                dist.broadcast(self.game.players[i], src=self.game.master_node)
    
        self.num_grad += len(index)
        self.k += 1
        # if (self.k % self.N) == 0:
        #     self.set_state()



# class QSGDA(DistributedOptimizer):
#     def step(self) -> None:
#         index = self.sample()
#         grad = self.game.operator(index)
#         with torch.no_grad():
#             grad, n_bits =self.quantization(grad)
            
#             self.n_bits += n_bits
#             dist.all_reduce(grad)
#             grad /= self.size
#             for i in range(self.game.num_players):
#                 lr = self.lr(self.k)
#                 g = self.game.unflatten(i, grad) # Reshape the grad to match players shape
#                 self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)

#             self.k += 1
#             self.num_grad += len(index)


# class DIANA_SGDA(DistributedOptimizer):
#     def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
#         super().__init__(game, options, prox)
#         self.alpha = options.alpha
#         if self.alpha is None and isinstance(self.quantization, RandKQuantization):
#             self.alpha = self.quantization.k / self.game.dim
#         elif self.alpha is None:
#             self.alpha = 0.

#         self.buffer = 0
#         self.buffer_server = 0

#     def step(self) -> None:
#         index = self.sample()
#         grad = self.game.operator(index).detach()
#         with torch.no_grad():
#             delta: torch.Tensor = grad - self.buffer
#             delta, n_bits = self.quantization(delta)
#             self.buffer = self.buffer + self.alpha*delta

#             self.n_bits += n_bits
#             dist.all_reduce(delta)
#             delta /= self.size
#             full_grad = self.buffer_server + delta
#             self.buffer_server = self.buffer_server + self.alpha*delta
#             for i in range(self.game.num_players):
#                 lr = self.lr(self.k)
#                 g = self.game.unflatten(i, full_grad)
#                 self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)
            
#             self.k += 1
#             self.num_grad += len(index)

# class VR_DIANA_SGDA(DistributedOptimizer):
#     def __init__(self, game: Game, options: OptimizerOptions = OptimizerOptions(), prox: Prox = Prox()) -> None:
#         super().__init__(game, options, prox)
#         self.alpha = options.alpha
#         if self.alpha is None and isinstance(self.quantization, RandKQuantization):
#             self.alpha = self.quantization.k / self.game.dim
#         elif self.alpha is None:
#             self.alpha = 0.

#         self.p = options.p
#         if self.p is None:
#             self.p = 1/game.num_samples
#         self.p = torch.as_tensor(self.p)

#         self.buffer = 0
#         self.buffer_server = 0
        
#         self.set_state()

#     def set_state(self) -> None:
#         self.game_copy = self.game.copy()
#         self.full_grad = self.game.full_operator().detach()
#         self.num_grad += self.game.num_samples

#     def step(self) -> None:
#         index = self.sample()
#         grad = self.game.operator(index).detach()
#         grad_copy = self.game.operator(index).detach()

#         update = (grad - grad_copy + self.full_grad)

#         if torch.bernoulli(self.p):
#             self.set_state()

#         with torch.no_grad():
#             delta: torch.Tensor = update - self.buffer
#             delta, n_bits = self.quantization(delta)
#             self.buffer = self.buffer + self.alpha*delta

#             self.n_bits += n_bits
#             dist.all_reduce(delta)
#             delta /= self.size
#             full_grad = self.buffer_server + delta
#             self.buffer_server = self.buffer_server + self.alpha*delta

#             for i in range(self.game.num_players):
#                 lr = self.lr(self.k)
#                 g = self.game.unflatten(i, full_grad)
#                 self.game.players[i].data = self.prox(self.game.players[i] - lr*g/self.size, lr)    
            
#             self.k += 1
#             self.num_grad += 2*len(index)
=======


class SGDACC(DistributedOptimizer):
    def __init__(self, game: Game, options: OptimizerOptions):
        super().__init__(game, options)
        self.options.aggregation_options.aggregator_type = AggregatorType.Mean
        self.aggregator = load_aggregator(self.options.aggregation_options)
        self.sigmaC = options.sigmaC

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
            if checking[2*i] not in self.attack_options.byzantines:
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
                grads = [torch.empty_like(grad) for _ in range(self.size)]
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
>>>>>>> 246dfe4... speed up
