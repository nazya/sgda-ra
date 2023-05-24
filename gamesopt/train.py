from .games import load_game, GameOptions, Game
from .optimizer.base import Optimizer, OptimizerType, OptimizerOptions
from .optimizer.single_thread import SGDARA, MSGDARA, SEGRA, SGDACC, SEGCC, RDEG
from dataclasses import dataclass
from collections import defaultdict
import sys
from .db import Record


def load_optimizer(game: Game, options: OptimizerOptions) -> Optimizer:
    if options.optimizer_type == OptimizerType.SGDARA:
        return SGDARA(game, options)
    elif options.optimizer_type == OptimizerType.MSGDARA:
        return MSGDARA(game, options)
    elif options.optimizer_type == OptimizerType.SEGRA:
        return SEGRA(game, options)
    elif options.optimizer_type == OptimizerType.SGDACC:
        return SGDACC(game, options)
    elif options.optimizer_type == OptimizerType.SEGCC:
        return SEGCC(game, options)
    elif options.optimizer_type == OptimizerType.RDEG:
        return RDEG(game, options)
    else:
        raise NotImplementedError()


@dataclass
class TrainConfig:
    num_iter: int
    n_peers: int
    game: GameOptions
    optimizer: OptimizerOptions


def train(config: TrainConfig, record: Record = Record()) -> Record:
    record.save_config(config)
    # torch.manual_seed(config.seed)

    # print("Init...")
    game = load_game(config.game)
    optimizer = load_optimizer(game, config.optimizer)

    metrics = defaultdict(list)
    for _ in range(config.num_iter):
        if optimizer.k % int(config.num_iter / 100) == 0 or optimizer.k == config.num_iter-1:
            metrics["dist"].append(game.dist())
            # metrics["hamiltonian"].append(game.hamiltonian())
            metrics["n_iter"].append(optimizer.k)
            metrics["num_grad"].append(optimizer.num_grad)
            sys.stdout.write(str(config.num_iter - 1 - optimizer.k)+9*' '+'\r')
            sys.stdout.flush()
        optimizer.step()
    record.save_metrics(metrics)
    return record


if __name__ == '__main__':
    record = train()
    print("Saved results to: %s" % record.path)