from gamesopt.db import Record
from .optimizer.base import DistributedOptimizer
from .games import load_game
from .optimizer import load_optimizer, OptimizerOptions
from dataclasses import dataclass
from collections import defaultdict
from .train import TrainConfig
import torch.multiprocessing as mp
import torch.distributed as dist
import os
import sys
import random


class PortNotAvailableError(Exception):
    pass


@dataclass
class TrainDistributedConfig(TrainConfig):
    n_process: int
    optimizer: OptimizerOptions
    # save_file: Optional[Path] = None
    # load_file: Optional[Path] = None


def _train(rank: int, port: str, config: TrainDistributedConfig, record: Record = Record()) -> None:
    setup(rank, config.n_process, port)
    # print("Init... ", rank)
    game = load_game(config.game, rank)
    game.set_master_node(0, config.n_process)
    # if config.load_file is not None:
    #     game_copy = game.load(config.load_file, copy=True)
    optimizer: DistributedOptimizer = load_optimizer(game, config.optimizer)

    # print("Starting...")
    metrics = defaultdict(list)
    # print(record.id)
    for _ in range(config.num_iter):
        hamiltonian = game.hamiltonian()
        num_grad = optimizer.get_num_grad()
        if rank == 0:
            metrics["hamiltonian"].append(hamiltonian)
            metrics["n_iter"].append(optimizer.k)
            metrics["num_grad"].append(num_grad)
            # metrics["n_bits"].append(n_bits)
            # metrics["prox_dist"].append(prox_dist)
            # if config.load_file:
            #     metrics["dist2opt"].append(game.dist(game_copy))
            record.save_metrics(metrics)
            sys.stdout.write('\r'+str(config.num_iter - 1 - optimizer.k)+9*' ')
            sys.stdout.flush()

        optimizer.step()
    dist.destroy_process_group()
    # if config.save_file is not None:
    #     game.save(config.save_file)


def setup(rank: int, size: int, port: str, backend: str = 'gloo') -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    try:
        dist.init_process_group(backend, rank=rank, world_size=size)
    except:
        raise PortNotAvailableError


def train(config: TrainDistributedConfig, record: Record = Record()) -> Record:
    record.save_config(config)
    # torch.manual_seed(config.seed)

    # Tries to allocate a port until a port is available
    while True:
        port = str(random.randrange(1030, 49151))
        print("Trying port %s" % port)
        try:
            mp.spawn(_train, args=(port, config, record),
                     nprocs=config.n_process,
                     join=True)
            break
        except PortNotAvailableError:
            print("Port %s not available" % port)
        else:
            raise

    return record


if __name__ == "__main__":
    record = train()
    print("Saved results to: %s" % record.path)