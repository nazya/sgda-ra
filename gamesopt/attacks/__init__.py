from .base import AttackType, AttackOptions
from .attacks import BitFlipping, ALIE, IPM, RandomNoise


def load_attack(options: AttackOptions):
    if options.attack_type == AttackType.BF:
        return BitFlipping(options)
    elif options.attack_type == AttackType.ALIE:
        return ALIE(options)
    elif options.attack_type == AttackType.IPM:
        return IPM(options)
    elif options.attack_type == AttackType.RN:
        return RandomNoise(options)
    else:
        raise NotImplementedError()