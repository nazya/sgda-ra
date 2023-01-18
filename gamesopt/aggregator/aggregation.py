# from typing import Optional
import torch
# import torch.nn.functional as F
from .base import _BaseAggregator, AggregationOptions


class Mean(_BaseAggregator):
    def __init__(self, options: AggregationOptions) -> None:
        pass

    def __call__(self, inputs):
        if len(inputs) == 0:
            print('len = 0')
            return 0
        inputs = torch.stack(inputs, dim=0)
        values = inputs.mean(dim=0)
        return values

    def __str__(self):
        return "Mean"


class TM(_BaseAggregator):
    def __init__(self, options: AggregationOptions) -> None:
        self.b = options.trimmed_mean_b
        # super(TM, self).__init__()

    def __call__(self, inputs):
        if len(inputs) - 2 * self.b > 0:
            b = self.b
        else:
            b = self.b
            while len(inputs) - 2 * b <= 0:
                b -= 1
            if b < 0:
                raise RuntimeError

        stacked = torch.stack(inputs, dim=0)
        largest, _ = torch.topk(stacked, b, 0)
        neg_smallest, _ = torch.topk(-stacked, b, 0)
        new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
        new_stacked /= len(inputs) - 2 * b
        return new_stacked

    def __str__(self):
        return "Trimmed Mean (b={})".format(self.b)


class CM(_BaseAggregator):
    def __init__(self, options: AggregationOptions) -> None:
        pass

    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2


class Clipping(_BaseAggregator):
    def __init__(self, options: AggregationOptions) -> None:
        self.tau = options.clipping_tau
        self.n_iter = options.clipping_n_iter
        # super(Clipping, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        for _ in range(self.n_iter):
            self.momentum = (
                    sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                    + self.momentum
            )

        # print(self.momentum[:5])
        # raise NotImplementedError
        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)

# krum

def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def multi_krum(distances, n, f, m):
    """Multi_Krum algorithm
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)
    vectors = [v.flatten() for v in vectors]

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances


class Krum(_BaseAggregator):
    r"""
    This script implements Multi-KRUM algorithm.
    Blanchard, Peva, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    Advances in Neural Information Processing Systems. 2017.
    """

    def __init__(self, options:AggregationOptions) -> None:
        self.n = options.n_total
        self.f = options.n_byzan
        self.m = options.krum_m
        self.top_m_indices = None
        # super(Krum, self).__init__()

    def __call__(self, inputs):
        distances = pairwise_euclidean_distances(inputs)
        top_m_indices = multi_krum(distances, self.n, self.f, self.m)
        values = sum(inputs[i] for i in top_m_indices)
        self.top_m_indices = top_m_indices
        return values

    def __str__(self):
        return "Krum (m={})".format(self.m)


def smoothed_weiszfeld(weights, alphas, z, nu, T):
    m = len(weights)
    if len(alphas) != m:
        raise ValueError

    if nu < 0:
        raise ValueError

    for t in range(T):
        betas = []
        for k in range(m):
            distance = _compute_euclidean_distance(z, weights[k])
            betas.append(alphas[k] / max(distance, nu))

        z = 0
        for w, beta in zip(weights, betas):
            z += w * beta
        z /= sum(betas)
    return z


class RFA(_BaseAggregator):
    r""""""

    def __init__(self, options:AggregationOptions) -> None:
        self.T = options.rfa_T
        self.nu = options.rfa_nu
        # super(RFA, self).__init__()

    # def rfa(self, weights, z):
    #     # print("=> ", [w.norm() for i, w in enumerate(weights)])
    #     return smoothed_weiszfeld(weights, alphas, z, nu=self.nu, T=self.T)

    def __call__(self, inputs):
        alphas = [1 / len(inputs) for _ in inputs]
        z = torch.zeros_like(inputs[0])
        return smoothed_weiszfeld(inputs, alphas, z=z, nu=self.nu, T=self.T)

    def __str__(self):
        return "RFA(T={},nu={})".format(self.T, self.nu)
