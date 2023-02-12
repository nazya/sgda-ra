from enum import Enum
from typing import Optional
from .base import Game
from .quadratic_games import QuadraticGame
# from .kelly_auction import KellyAuction, KellyAuctionConfig
# from .bilinear import BilinearGame, BilinearGameConfig
# from .robust_regression import RobustLinRegConfig, RobustLinReg, RobustLogReg


class GameType(Enum):
    QUADRATIC = "quadratic"
    # KELLY_AUCTION = "kelly_auction"
    # ROBUST_LINEAR_REG = "robust_linear_reg"
    # BILINEAR = "bilinear"
    # ROBUST_LOGISTIC_REG = "robust_logistic_regression"


def load_game(config, data, rank: Optional[int] = None, size: Optional[int] = None) -> Game:
    if config.game_type == GameType.QUADRATIC:
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

