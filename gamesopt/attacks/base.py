"""
Attacks which take in gradients.
"""
from enum import Enum


class AttackType(Enum):
    BF = "BitFlipping"
    ALIE = "ALittleIsEnough"
    IPM = "InnerProductManipulation"
    RN = "RandomNoise"


class _BaseAttack(object):
    """Base class of attacks.
    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        """Performe an attack in-place.
        Args:
            inputs (list): A list of true gradients to be attacked.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError