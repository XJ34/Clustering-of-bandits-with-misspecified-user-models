from statistics import mean

import numpy as np
from utlis import generate_items


def get_best_reward(items, theta, epsilons):
    return np.max(np.dot(items, theta) + epsilons)


class Environment:
    # p: frequency vector of users
    def __init__(self, L, d, m, num_users, p, theta, epsilon):
        self.L = L
        self.d = d
        self.p = p  # probability distribution over users
        self.epsilons = np.random.uniform(-epsilon, epsilon, size=(1000, 1000))
        self.epsilons_t = np.random.uniform(-epsilon, epsilon, size=(1000, 20))
        self.items = generate_items(num_items=1000, d=d)
        self.theta = theta
        self.items_t = generate_items(num_items=20, d=d)
        self.index_t = np.arange(1000)

    def get_items(self):
        np.random.shuffle(self.index_t)
        self.items_t = self.items[self.index_t[0:20]]
        return self.items_t

    def get_epsilons(self):
        self.epsilons_t = self.epsilons[:, self.index_t[0:20]]
        return self.epsilons_t

    def feedback(self, i, k):
        x = self.items_t[k, :]
        r = np.dot(self.theta[i], x) + self.epsilons_t[i][k]
        y = r + np.random.normal(0, 0.1, size=1)
        br = get_best_reward(self.items_t, self.theta[i], self.epsilons_t[i])
        return y, r, br

    def generate_users(self):
        X = np.random.multinomial(1, self.p)
        I = np.nonzero(X)[0]
        return I
