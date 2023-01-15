"""
Aggregators which takes in weights and gradients.
"""
import torch
from dataclasses import dataclass
from enum import Enum


class AggregatorType(Enum):
    Mean = "Mean"
    Clipping = "Clipping"
    Krum = "Krum"
    CM = "CM"
    TM = "TM"
    RFA = "RFA"

    
@dataclass
class AggregationOptions:
    n_total: int
    n_byzan: int
    aggregator_type: AggregatorType
    use_bucketing: bool
    bucketing_s: int
    clipping_tau: int
    clipping_n_iter: int
    trimmed_mean_b: int
    krum_m: int
    rfa_T: int
    rfa_nu: int
    # bucketing: int=10


class _BaseAggregator(object):
    """Base class of aggregators.
    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self, inputs):
        # Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})
        return

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class _BaseAsyncAggregator(object):
    """AsyncAggregator base object"""

    def __init__(self):
        # Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})
        return

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


# class Mean(_BaseAggregator):
#     def __call__(self, inputs):
#         values = torch.stack(inputs, dim=0).mean(dim=0)
#         return values

#     def __str__(self):
#         return "Mean"
