import os
import sys
import torch
import random
from mlflow import MlflowClient
import torch.distributed as dist
import torch.multiprocessing as mp
from dataclasses import dataclass
from gamesopt.aggregator import AggregatorType
from gamesopt.attacks import AttackType
from gamesopt.games import GameType
from .optimizer.base import Optimizer, OptimizerType
from .optimizer.distributed import SGDARA, MSGDARA, SEGRA, SGDACC, SEGCC, RDEG
# from .train import TrainConfig


def load_distributed_optimizer(config, data, rank):
    if config.optimizer_type == OptimizerType.SGDARA:
        return SGDARA(config, data, rank)
    elif config.optimizer_type == OptimizerType.MSGDARA:
        return MSGDARA(config, data, rank)
    elif config.optimizer_type == OptimizerType.SEGRA:
        return SEGRA(config, data, rank)
    elif config.optimizer_type == OptimizerType.SGDACC:
        return SGDACC(config, data, rank)
    elif config.optimizer_type == OptimizerType.SEGCC:
        return SEGCC(config, data, rank)
    elif config.optimizer_type == OptimizerType.RDEG:
        return RDEG(config, data, rank)
    else:
        raise NotImplementedError()


class PortNotAvailableError(Exception):
    pass


@dataclass
class BaseConfig:
    n_iter: int

    n_peers: int
    n_byzan: int

    game_type: GameType
    num_samples: int
    dim: int
    with_bias: bool
    mu: float
    ell: float

    attack_type: AttackType
    n_attacking: int
    ipm_epsilon: float
    rn_sigma: float
    alie_z: float

    use_bucketing: bool
    bucketing_s: int
    aggregator_type: AggregatorType
    trimmed_mean_b: int
    krum_m: int
    clipping_tau: int
    clipping_n_iter: int
    rfa_T: int
    rfa_nu: float

    optimizer_type: OptimizerType
    alpha: float
    lr: float
    lr_inner: float
    lr_outer: float
    sigmaC: float
    batch_size: int


@dataclass
class BaseData:
    matrix: torch.Tensor
    bias: torch.Tensor
    true: torch.Tensor
    players: torch.Tensor


def _train(rank: int, port: str, config: BaseConfig, data: BaseData):
    setup(rank, config.n_peers, port)
    # print("Init... ", rank)
    optimizer: Optimizer = load_distributed_optimizer(config, data, rank)
    verbose = os.environ['MLFLOW_VERBOSE'] == 'True'

    # print("Starting... ", rank == dist.get_rank())
    if verbose and rank == 0:
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        tracking_uri = os.environ['MLFLOW_TRACKING_URI']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        e_id = client.create_experiment(experiment_name) if e is None else e.experiment_id
        r = client.create_run(experiment_id=e_id, run_name=str(config))
        r_id = r.info.run_id

    for _ in range(config.n_iter):
        if verbose and rank == 0:
            if optimizer.k % int(config.n_iter / 200) == 0 or optimizer.k == config.n_iter - 1:
                client.log_metric(r_id, 'dist',
                                  optimizer.game.dist(),
                                  timestamp=optimizer.num_grad,
                                  step=optimizer.k)
                # sys.stdout.write(str(optimizer.game.dist())+'\r')
                sys.stdout.write(str(config.n_iter - 1 - optimizer.k) + 9*' ' + '\r')
                sys.stdout.flush()

        optimizer.step()
    if verbose and rank == 0:
        client.set_terminated(r_id)
    dist.destroy_process_group()


def setup(rank: int, size: int, port: str, backend: str = 'gloo') -> None:
    os.environ['MASTER_ADDR'] = '127.0.1.1'
    os.environ['MASTER_PORT'] = port
    try:
        dist.init_process_group(backend, rank=rank, world_size=size)
    except:
        raise PortNotAvailableError


def train(config: BaseConfig, data: BaseData):
    # Tries to allocate a port until a port is available
    while True:
        port = str(random.randrange(1030, 49151))
        print("Trying port %s" % port)
        try:
            mp.spawn(_train,
                     args=(port, config, data),
                     nprocs=config.n_peers,
                     join=True)
            break
        except PortNotAvailableError:
            print("Port %s not available" % port)
        else:
            raise


if __name__ == "__main__":
    train()