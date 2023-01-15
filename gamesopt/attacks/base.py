"""
Attacks which take in gradients.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AttackType(Enum):
    BF = "BitFlipping"
    ALIE = "ALittleIsEnough"
    IPM = "InnerProductManipulation"
    RN = "RandomNoise"


@dataclass
class AttackOptions:
    n_total: int
    n_byzan: int
    attack_type: AttackType
    rn_sigma: float = 0.
    ipm_epsilon: float = 0.
    alie_z: float = None

    def __post_init__(self):
        self.byzan_index = np.arange(self.n_total - self.n_byzan, self.n_total)


class _BaseAttack(object):
    """Base class of attacks.
    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, inputs):
        # Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})
        return

    def __call__(self, inputs):
        """Performe an attack in-place.
        Args:
            inputs (list): A list of true gradients to be attacked.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError