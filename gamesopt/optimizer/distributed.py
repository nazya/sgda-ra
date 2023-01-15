import torch
import torch.distributed as dist
from .base import DistributedOptimizer


class SGDARA(DistributedOptimizer):
    def step(self) -> None:
        index = self.sample()
        index = torch.sort(index).values
        grad = self.game.operator(index).detach()
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

            # server
            if self.game.rank == self.game.master_node:
                # attack
                for rank in range(self.size):
                    self.attack(grads, rank)
                    # dist.all_gather(tensor=grad, tensor_list=grads)

                # aggregation
                agg_grad = self.aggregator(grads)
                for i in range(self.game.num_players):
                    g = self.game.unflatten(i, agg_grad)
                    self.game.players[i].data = self.game.players[i] - self.lr*g

            # broadcast new point
            for i in range(self.game.num_players):
                dist.broadcast(self.game.players[i], src=self.game.master_node)

        self.k += 1
        self.num_grad += len(index)




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
