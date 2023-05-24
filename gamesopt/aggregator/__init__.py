import numpy as np
from .base import Aggregator
from .aggregation import Mean, CM, Clipping, Krum, TM, RFA, UnivariateTM


def load_aggregator(config):
    # if config.use_bucketing:
    #     load_bucketing(config)
    if config.aggregator == Aggregator.Mean:
        return Mean(config)
    elif config.aggregator == Aggregator.Clipping:
        return Clipping(config)
    elif config.aggregator == Aggregator.Krum:
        return Krum(config)
    elif config.aggregator == Aggregator.TM:
        return TM(config)
    elif config.aggregator == Aggregator.CM:
        return CM(config)
    elif config.aggregator == Aggregator.RFA:
        return RFA(config)
    elif config.aggregator == Aggregator.UnivariateTM:
        return UnivariateTM(config)
    else:
        raise NotImplementedError()


def load_bucketing(config):
    if config.use_bucketing:
        return Bucketing(config)
    else:
        return load_aggregator(config)


class Bucketing(object):
    def __init__(self, config) -> None:
        self.n = config.n_peers
        self.aggregator = load_aggregator(config)
        self.s = config.bucketing_s

    def __call__(self, inputs):
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)

        T = int(np.ceil(self.n / self.s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * self.s:(t + 1) * self.s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        # reshuffled_inputs = torch.stack(reshuffled_inputs, 0)
        return self.aggregator(reshuffled_inputs)

    def __str__(self):
        return "Bucketing (agg={}, s={})".format(self.aggregator, self.s)
