import os
import sys
import json
import random
from collections import namedtuple
# import mlflow
from mlflow import MlflowClient
import torch.distributed as dist
import torch.multiprocessing as mp
from .optimizer.base import Optimizer
from .optimizer.distributed import SGDA, SGDARA, MSGDARA, SEGRA, SGDACC, SEGCC, RDEG
# from .train import TrainConfig


def load_distributed_optimizer(config, data, rank):
    if config.optimizer == Optimizer.SGDA:
        return SGDA(config, data, rank)
    elif config.optimizer == Optimizer.SGDARA:
        return SGDARA(config, data, rank)
    elif config.optimizer == Optimizer.MSGDARA:
        return MSGDARA(config, data, rank)
    elif config.optimizer == Optimizer.SEGRA:
        return SEGRA(config, data, rank)
    elif config.optimizer == Optimizer.SGDACC:
        return SGDACC(config, data, rank)
    elif config.optimizer == Optimizer.SEGCC:
        return SEGCC(config, data, rank)
    elif config.optimizer == Optimizer.RDEG:
        return RDEG(config, data, rank)
    else:
        raise NotImplementedError()


class PortNotAvailableError(Exception):
    pass


def _train(rank: int, port: str, config, data):
    # setup(rank, config['n_peers'], port)
    config = json.loads(config)
    setup(rank, config['n_peers'], port)
    # print("Init... ", rank)
    # config = BaseConfig(**config)
    # data = BaseData(**data)
    verbose = os.environ['MLFLOW_VERBOSE'] == 'True'
    # print("Starting... ", rank == dist.get_rank())
    if verbose and rank == 0:
        tracking_uri = os.path.expanduser('~/mlruns/')
        experiment_name = os.environ['MLFLOW_EXPERIMENT_NAME']
        client = MlflowClient(tracking_uri=tracking_uri)
        e = client.get_experiment_by_name(experiment_name)
        e_id = client.create_experiment(experiment_name) if e is None else e.experiment_id
        r = client.create_run(experiment_id=e_id, run_name=os.environ['MLFLOW_RUN_NAME'])
        r_id = r.info.run_id
        client.log_dict(r_id, config, 'config.json')
        client.log_param(r_id, 'Title', os.environ['MLFLOW_RUN_TITLE'])

    config = namedtuple('Config', config.keys())(**config)
    data = namedtuple('Data', data.keys())(**data)
    optimizer = load_distributed_optimizer(config, data, rank)
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


def train(config, data):
    nprocs = config.n_peers
    # config = config.__dict__
    config = json.dumps(config.__dict__)
    data = data.__dict__
    # Tries to allocate a port until a port is available
    while True:
        port = str(random.randrange(1030, 49151))
        print("Trying port %s" % port)
        try:
            mp.spawn(_train,
                     args=(port, config, data),
                     # nprocs=config['n_peers'],
                     # nprocs=config.n_peers,
                     nprocs=nprocs,
                     join=True)
            break
        except PortNotAvailableError:
            print("Port %s not available" % port)
        else:
            raise


if __name__ == "__main__":
    train()