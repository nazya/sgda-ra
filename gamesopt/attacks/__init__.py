from .base import AttackType
from .attacks import BitFlipping, ALIE, IPM, RandomNoise


def load_attack(config):
    if config.attack_type == AttackType.BF:
        return BitFlipping(config)
    elif config.attack_type == AttackType.ALIE:
        return ALIE(config)
    elif config.attack_type == AttackType.IPM:
        return IPM(config)
    elif config.attack_type == AttackType.RN:
        return RandomNoise(config)
    else:
        raise NotImplementedError()