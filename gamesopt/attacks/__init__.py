from .base import Attack
from .attacks import BitFlipping, ALIE, IPM, RandomNoise


def load_attack(config):
    if config.attack == Attack.BF:
        return BitFlipping(config)
    elif config.attack == Attack.ALIE:
        return ALIE(config)
    elif config.attack == Attack.IPM:
        return IPM(config)
    elif config.attack == Attack.RN:
        return RandomNoise(config)
    else:
        raise NotImplementedError()


