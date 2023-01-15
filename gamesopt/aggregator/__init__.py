import numpy as np
from .base import AggregatorType, AggregationOptions
from .aggregation import Mean, CM, Clipping, Krum, TM, RFA


def load_bucketing(options: AggregationOptions):
    if options.use_bucketing:
        return Bucketing(options)
    else:
        return load_aggregator(options)


def load_aggregator(options: AggregationOptions):
    # if options.use_bucketing:
    #     load_bucketing(options)
    if options.aggregator_type == AggregatorType.Mean:
        return Mean(options)
    elif options.aggregator_type == AggregatorType.Clipping:
        return Clipping(options)
    elif options.aggregator_type == AggregatorType.Krum:
        return Krum(options)
    elif options.aggregator_type == AggregatorType.TM:
        return TM(options)
    elif options.aggregator_type == AggregatorType.CM:
        return CM(options)
    elif options.aggregator_type == AggregatorType.RFA:
        return RFA(options)
    else:
        raise NotImplementedError()


class Bucketing(object):
    def __init__(self, options: AggregationOptions) -> None:
        self.n = options.n_total
        self.aggregator = load_aggregator(options)
        self.s = options.bucketing_s

    def __call__(self, inputs):
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)

        T = int(np.ceil(self.n / self.s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * self.s:(t + 1) * self.s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return self.aggregator(reshuffled_inputs)

    def __str__(self):
        return "Bucketing (agg={}, s={})".format(self.aggregator, self.s)



