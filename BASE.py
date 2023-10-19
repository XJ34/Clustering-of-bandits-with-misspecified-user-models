### Our codes are derived from the released code of the work 'Improved Algorithm on Online Clustering of Bandits'
### the original codes are in this link: https://github.com/shuaili8/OnlineClusteringOfBandits
### We use the algorithms in the original codes as baselines, and we don't modify the code but add misspecification to the feedback
### Our algorithms are in the RCLUMB.py and RSCLUMB.py, the common hyper-parameter of these algorithms is the \alpha, and we keep it same
### in all algorithms.


import numpy as np
from utlis import isInvertible
import random


class Base:
    # Base agent for online clustering of bandits
    def __init__(self, d, T):
        self.d = d
        self.T = T
        # self.beta = np.sqrt(self.d * np.log(self.T / self.d)) # parameter for select item
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)
        self.Xs = {}

    def _beta(self, N, t):
        return np.sqrt(self.d * np.log(1 + N / self.d) + 4 * np.log(t) + np.log(2)) + 1

    def _select_item_ucb(self, S, Sinv, theta, items, N, t):
        # print('beta', self._beta(N, t))
        return np.argmax(np.dot(items, theta) + self._beta(N, t) * (np.matmul(items, Sinv) * items).sum(axis=1))

    def recommend(self, i, items, t):
        # items is of type np.array (L, d)
        # select one index from items to user i

        return

    def store_info(self, i, x, y, t, r, br):
        if i in self.Xs.keys():
            self.Xs[i].append(x)
        else:
            self.Xs.update({i: [x]})
        return

    def _update_inverse(self, S, b, Sinv, x, t):
        Sinv = np.linalg.inv(S)
        theta = np.matmul(Sinv, b)
        return Sinv, theta

    def update(self, t):
        return

    def run(self, envir):
        for t in range(self.T):
            if t % 5000 == 0:
                print(t // 5000, end=' ')
            self.I = envir.generate_users()
            for i in self.I:
                items = envir.get_items()
                epsilons = envir.get_epsilons()
                kk = self.recommend(i=i, items=items, t=t)
                x = items[kk]
                y, r, br = envir.feedback(i=i, k=kk)
                self.store_info(i=i, x=x, y=y, t=t, r=r, br=br)

            self.update(t)

        print()


class LinUCB(Base):
    def __init__(self, d, T):
        super(LinUCB, self).__init__(d, T)
        self.S = np.eye(d)
        self.b = np.zeros(d)
        self.Sinv = np.eye(d)
        self.theta = np.zeros(d)

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S, self.Sinv, self.theta, items, t, t)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        self.S += np.outer(x, x)
        self.b += y * x

        self.Sinv, self.theta = self._update_inverse(self.S, self.b, self.Sinv, x, t)


class LinUCB_Cluster(Base):
    def __init__(self, indexes, m, d, T):
        super(LinUCB_Cluster, self).__init__(d, T)
        self.indexes = indexes

        self.S = {i: np.eye(d) for i in range(m)}
        self.b = {i: np.zeros(d) for i in range(m)}
        self.Sinv = {i: np.eye(d) for i in range(m)}
        self.theta = {i: np.zeros(d) for i in range(m)}

        self.N = np.zeros(m)

    def recommend(self, i, items, t):
        j = self.indexes[i]
        return self._select_item_ucb(self.S[j], self.Sinv[j], self.theta[j], items, self.N[j], t)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        j = self.indexes[i]
        self.S[j] += np.outer(x, x)
        self.b[j] += y * x
        self.N[j] += 1

        self.Sinv[j], self.theta[j] = self._update_inverse(self.S[j], self.b[j], self.Sinv[j], x, self.N[j])


class LinUCB_IND(Base):
    # each user is an independent LinUCB
    def __init__(self, nu, d, T):
        super(LinUCB_IND, self).__init__(d, T)
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.w = {i: 1 for i in range(nu)}

        self.N = np.zeros(nu)

    def recommend(self, i, items, t):
        return self._select_item_ucb(self.S[i], self.Sinv[i], self.theta[i], items, self.N[i], t)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br

        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i], self.theta[i] = self._update_inverse(self.S[i], self.b[i], self.Sinv[i], x, self.N[i])
