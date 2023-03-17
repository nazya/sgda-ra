"""
Attacks which take in gradients.
"""
from enum import auto
from gamesopt import DictEnum


class Attack(DictEnum):
    BF = auto()
    RN = auto()
    IPM = auto()
    ALIE = auto()


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