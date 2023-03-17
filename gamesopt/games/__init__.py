from typing import Optional
from .base import Game
from .utils import create_bias, create_matrix
from .quadratic_games import QuadraticGame


def load_game(config, data, rank: Optional[int] = None, size: Optional[int] = None) -> Game:
    if config.game == Game.Quadratic:
        return QuadraticGame(rank, config, data)
    # elif options.game_type == GameType.KELLY_AUCTION:
    #     return KellyAuction(options.kelly_auction_options, rank)
    # elif options.game_type == GameType.ROBUST_LINEAR_REG:
    #     return RobustLinReg(options.robust_linear_reg_options, rank)
    # elif options.game_type == GameType.BILINEAR:
    #     return BilinearGame(options.bilinear_options, rank)
    # elif options.game_type == GameType.ROBUST_LOGISTIC_REG:
    #     return RobustLogReg(rank)
    else:
        raise ValueError()
