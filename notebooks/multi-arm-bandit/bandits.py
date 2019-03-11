
from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, probabilities = None):
        assert probabilities is None or len(probabilities) == n
        self.n = n
        if probabilities is None:
            np.random.seed(int(time.time()))
            self.probabilities = [np.random.random() for _ in range(self.n)]
        else:
            self.probabilities = probabilities

        self.best_proba = max(self.probabilities)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.probabilities[i]:
            return 1
        else:
            return 0