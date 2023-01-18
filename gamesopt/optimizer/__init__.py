from gamesopt.optimizer.distributed import SGDARA, MSGDARA, SEGRA, SGDACC
from gamesopt.games import Game
from .base import Optimizer, OptimizerOptions, OptimizerType


def load_optimizer(game: Game, options: OptimizerOptions) -> Optimizer:
    if options.optimizer_type == OptimizerType.SGDARA:
        return SGDARA(game, options)
    elif options.optimizer_type == OptimizerType.MSGDARA:
        return MSGDARA(game, options)
    elif options.optimizer_type == OptimizerType.SEGRA:
        return SEGRA(game, options)
    elif options.optimizer_type == OptimizerType.SGDACC:
        return SGDACC(game, options)
    # elif options.optimizer_type == OptimizerType.PROX_LSVRGDA:
    #     return ProxLSVRGDA(game, options, prox)
    # elif options.optimizer_type == OptimizerType.SVRG:
    #     return SVRG(game, options, prox)
    # elif options.optimizer_type == OptimizerType.VRAGDA:
    #     return VRAGDA(game, options, prox)
    # elif options.optimizer_type == OptimizerType.VRFORB:
    #     return VRFoRB(game, options, prox)
    # elif options.optimizer_type == OptimizerType.SVRE:
    #     return SVRE(game, options, prox)
    # elif options.optimizer_type == OptimizerType.EG_VR:
    #     return EGwithVR(game, options, prox)
    # elif options.optimizer_type == OptimizerType.QSGDA:
    #     return QSGDA(game, options, prox)
    # elif options.optimizer_type == OptimizerType.DIANA_SGDA:
    #     return DIANA_SGDA(game, options, prox)
    # elif options.optimizer_type == OptimizerType.VR_DIANA_SGDA:
    #     return VR_DIANA_SGDA(game, options, prox)
    else:
        raise NotImplementedError()