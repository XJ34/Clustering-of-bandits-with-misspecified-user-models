import numpy as np
from utlis import isInvertible



class Cluster:
    def __init__(self, users, S, b, N, checks):
        self.users = users  # a list/array of users
        self.S = S
        self.b = b
        self.N = N
        self.checks = checks

        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)
        self.checked = len(self.users) == sum(self.checks.values())
        self.Xs = {}

    def update_check(self, i):
        self.checks[i] = True
        self.checked = len(self.users) == sum(self.checks.values())


class Mis_SCLUB():
    def __init__(self, nu, d, num_stages):
        self.T = 2 ** num_stages - 1
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)
        self.nu = nu
        self.d = d
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.N = np.zeros(nu)
        self.clusters = {0: Cluster(users=[i for i in range(nu)], S=np.eye(d), b=np.zeros(d), N=0,
                                    checks={i: False for i in range(nu)})}
        self.cluster_inds = np.zeros(nu)
        self.num_stages = num_stages
        # self.alpha = 4 * np.sqrt(d)
        # self.alpha_p = np.sqrt(4) # 2
        self.num_clusters = np.ones(self.T)
        self.max_epsilon = 0.2

    def _init_each_stage(self):
        for c in self.clusters:
            self.clusters[c].checks = {i: False for i in self.clusters[c].users}
            self.clusters[c].checked = False

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        # print(cluster.users)
        bonus = np.zeros((20,))
        # print('b', bonus[0])
        for i in cluster.users:
            if i in cluster.Xs.keys():
                Xs = np.array(cluster.Xs[i])
                bonus += self.max_epsilon * abs(np.matmul(np.matmul(items, cluster.Sinv), Xs.T)).sum(axis=1)

        return np.argmax(np.dot(items, cluster.theta) + 2 * (
                np.matmul(items, cluster.Sinv) * items).sum(axis=1) + bonus)

    def store_info(self, i, x, y, t, r, br=1):
        self.rewards[t] += r
        self.best_rewards[t] += br

        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i] = np.linalg.inv(self.S[i])
        self.theta[i] = np.matmul(self.Sinv[i], self.b[i])

        c = self.cluster_inds[i]
        self.clusters[c].S += np.outer(x, x)
        self.clusters[c].b += y * x
        self.clusters[c].N += 1

        self.clusters[c].Sinv = np.linalg.inv(self.clusters[c].S)
        self.clusters[c].theta = np.matmul(self.clusters[c].Sinv, self.clusters[c].b)
        if i in self.clusters[c].Xs.keys():
            self.clusters[c].Xs[i].append(x)
        else:
            self.clusters[c].Xs.update({i: [x]})

    def _factT(self, T):
        return np.sqrt((1 + np.log(1 + T)) / (1 + T))

    def _split_or_merge(self, theta, N1, N2, split=True):
        alpha = 1
        alpha2 = 1.5
        if split:
            return np.linalg.norm(theta) > alpha * (self._factT(N1) + self._factT(N2)) + alpha2*self.max_epsilon
        else:
            return np.linalg.norm(theta) < alpha * (self._factT(N1) + self._factT(N2)) / 2 + 0.5*alpha2*self.max_epsilon

    def _cluster_avg_freq(self, c, t):
        return self.clusters[c].N / (len(self.clusters[c].users) * t)

    def split(self, i, t):
        c = self.cluster_inds[i]
        cluster = self.clusters[c]

        cluster.update_check(i)

        if self._split_or_merge(self.theta[i] - cluster.theta, self.N[i],
                                                                      cluster.N, split=True):

            def _find_available_index():
                cmax = max(self.clusters)
                for c1 in range(cmax + 1):
                    if c1 not in self.clusters:
                        return c1
                return cmax + 1

            cnew = _find_available_index()
            self.clusters[cnew] = Cluster(users=[i], S=self.S[i], b=self.b[i], N=self.N[i], checks={i: True})
            self.cluster_inds[i] = cnew

            cluster.users.remove(i)
            cluster.S = cluster.S - self.S[i] + np.eye(self.d)
            cluster.b = cluster.b - self.b[i]
            cluster.N = cluster.N - self.N[i]
            del cluster.checks[i]

    def merge(self, t):
        cmax = max(self.clusters)

        for c1 in range(cmax + 1):
            if c1 not in self.clusters or self.clusters[c1].checked == False:
                continue

            for c2 in range(c1 + 1, cmax + 1):
                if c2 not in self.clusters or self.clusters[c2].checked == False:
                    continue

                if self._split_or_merge(self.clusters[c1].theta - self.clusters[c2].theta, self.clusters[c1].N,
                                        self.clusters[c2].N, split=False):

                    for i in self.clusters[c2].users:
                        self.cluster_inds[i] = c1

                    self.clusters[c1].users = self.clusters[c1].users + self.clusters[c2].users
                    self.clusters[c1].S = self.clusters[c1].S + self.clusters[c2].S - np.eye(self.d)
                    self.clusters[c1].b = self.clusters[c1].b + self.clusters[c2].b
                    self.clusters[c1].N = self.clusters[c1].N + self.clusters[c2].N
                    self.clusters[c1].checks = {**self.clusters[c1].checks, **self.clusters[c2].checks}

                    del self.clusters[c2]

    def run(self, envir):
        for s in range(self.num_stages):
            self._init_each_stage()
            for t in range(2 ** s):
                if t % 5000 == 0:
                    print(t // 5000, end=' ')

                tau = 2 ** s + t - 1

                I = envir.generate_users()
                for i in I:
                    items = envir.get_items()
                    epsilons = envir.get_epsilons()
                    kk = self.recommend(i, items, tau)
                    x = items[kk]
                    y, r, br = envir.feedback(i, kk)
                    self.store_info(i, x, y, tau, r, br)

                for i in I:
                    # c = self.cluster_inds[i]
                    self.split(i, tau)

                self.merge(tau)
                self.num_clusters[tau] = len(self.clusters)

        print()