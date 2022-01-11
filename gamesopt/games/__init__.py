from .base import Game
from .quadratic_games import QuadraticGame, QuadraticGameConfig
from .kelly_auction import KellyAuction, KellyAuctionConfig
from enum import Enum
from dataclasses import dataclass


class GameType(Enum):
    QUADRATIC = "quadratic"
    KELLY_AUCTION = "kelly_auction"


@dataclass
class GameOptions:
    game_type: GameType = GameType.QUADRATIC
    quadratic_options: QuadraticGameConfig = QuadraticGameConfig()
    kelly_auction_options: KellyAuctionConfig = KellyAuctionConfig()


def load_game(options: GameOptions = GameOptions()) -> Game:
    if options.game_type == GameType.QUADRATIC:
        return QuadraticGame(options.quadratic_options)
    elif options.game_type == GameType.KELLY_AUCTION:
        return KellyAuction(options.kelly_auction_options)
