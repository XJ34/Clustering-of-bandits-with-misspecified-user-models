import networkx as nx
import numpy as np
from utlis import edge_probability, is_power2, isInvertible


class Cluster:
    def __init__(self, users, S, b, N):
        self.users = users
        self.S = S
        self.b = b
        self.N = N
        self.Sinv = np.linalg.inv(self.S)
        self.theta = np.matmul(self.Sinv, self.b)
        self.Xs = {}


class RCLUMB():
    def __init__(self, nu, d, T, edge_probability=1):
        self.nu = nu
        self.d = d
        self.T = T
        self.max_epsilon = 0.2
        self.G = nx.gnp_random_graph(nu, edge_probability)
        self.clusters = {0: Cluster(users=range(nu), S=np.eye(d), b=np.zeros(d), N=0)}
        self.cluster_inds = np.zeros(nu)
        self.num_clusters = np.zeros(T)
        self.rewards = np.zeros(self.T)
        self.best_rewards = np.zeros(self.T)
        self.Xs = {}
        self.S = {i: np.eye(d) for i in range(nu)}
        self.b = {i: np.zeros(d) for i in range(nu)}
        self.Sinv = {i: np.eye(d) for i in range(nu)}
        self.theta = {i: np.zeros(d) for i in range(nu)}
        self.N = np.zeros(nu)

    def recommend(self, i, items, t):
        cluster = self.clusters[self.cluster_inds[i]]
        neighbors = [a for a in self.G.neighbors(i)]
        neighbors.append(i)
        neighbors_S = sum([self.S[k] - np.eye(self.d) for k in neighbors]) + np.eye(self.d)
        neighbors_b = np.zeros((50,))
        for k in neighbors:
            neighbors_b += self.b[k]
        neighbors_Sinv = np.linalg.inv(neighbors_S)
        neighbors_theta = np.matmul(neighbors_Sinv, neighbors_b)
        bonus = np.zeros((20,))
        for k in neighbors:
            if k in cluster.Xs.keys():
                Xs = np.array(cluster.Xs[k])
                bonus += self.max_epsilon * abs(np.matmul(np.matmul(items, neighbors_Sinv), Xs.T)).sum(axis=1)
        return np.argmax(np.dot(items, neighbors_theta) + 1.5 * (
                np.matmul(items, neighbors_Sinv) * items).sum(axis=1) + bonus)

    def store_info(self, i, x, y, t, r, br):
        self.rewards[t] += r
        self.best_rewards[t] += br
        self.S[i] += np.outer(x, x)
        self.b[i] += y * x
        self.N[i] += 1

        self.Sinv[i]= np.linalg.inv(self.S[i])
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

    def _if_split(self, theta, N1, N2):
        alpha = 1
        alpha_2 = 1.5

        def _factT(T):
            return np.sqrt((1 + np.log(1 + T)) / (1 + T))

        return np.linalg.norm(theta) > (alpha * (_factT(N1) + _factT(N2)) + alpha_2 * self.max_epsilon)

    def update(self, t):
        update_clusters = False
        for i in self.I:
            c = self.cluster_inds[i]

            A = [a for a in self.G.neighbors(i)]
            for j in A:
                if self.N[i] and self.N[j] and self._if_split(self.theta[i] - self.theta[j], self.N[i], self.N[j]):
                    self.G.remove_edge(i, j)
                    # print(i, j)
                    update_clusters = True

        if update_clusters:
            C = set()
            for i in self.I:  # suppose there is only one user per round
                C = nx.node_connected_component(self.G, i)
                if len(C) < len(self.clusters[c].users):
                    remain_users = set(self.clusters[c].users)
                    self.clusters[c] = Cluster(list(C), S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(self.d),
                                               b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))

                    remain_users = remain_users - set(C)
                    c = max(self.clusters) + 1
                    while len(remain_users) > 0:
                        j = np.random.choice(list(remain_users))
                        C = nx.node_connected_component(self.G, j)

                        self.clusters[c] = Cluster(list(C),
                                                   S=sum([self.S[k] - np.eye(self.d) for k in C]) + np.eye(
                                                       self.d),
                                                   b=sum([self.b[k] for k in C]), N=sum([self.N[k] for k in C]))
                        for j in C:
                            self.cluster_inds[j] = c

                        c += 1
                        remain_users = remain_users - set(C)

            # print(len(self.clusters))

        self.num_clusters[t] = len(self.clusters)

    def run(self, envir):
        for t in range(self.T):
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