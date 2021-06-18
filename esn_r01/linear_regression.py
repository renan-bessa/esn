import numpy as np
import numpy.matlib as npm
import os
import matplotlib.pyplot as plt
import scipy.io


class LrRLS(object):
    """description of class"""

    def __init__(self, num_in, num_out, reg_const=1e+4):
        self.alfa = 1
        self.inv_corr = reg_const * np.eye(num_in)
        self.weight = np.zeros([num_in, num_out])
        self.erro = []
        self.k_gain = np.asarray([])
        print('número de entradas', num_in)

    def update(self, sys_target, sys_out, sys_in):
        self.erro.append(sys_target - sys_out)
        self.k_gain = np.inner(self.inv_corr, sys_in) / (self.alfa + np.inner(sys_in, np.inner(self.inv_corr, sys_in)))
        self.k_gain = np.reshape(self.k_gain, (self.k_gain.shape[0], 1))
        self.weight = self.weight + np.inner(self.k_gain, self.erro[-1].T)
        self.inv_corr = (1 / self.alfa) * (self.inv_corr - np.inner(np.outer(self.k_gain, sys_in), self.inv_corr))

        return np.reshape(self.weight, (self.weight.shape[0], sys_out.shape[1])).copy()


class LrLMS(object):
    """description of class"""

    def __init__(self, num_in, num_out, mu=0.6):
        self.mu = mu
        self.weight = np.zeros([num_in, num_out])
        self.erro = []
        self.k_gain = np.asarray([])

    def update(self, sys_target, sys_out, sys_in):
        self.erro.append(sys_target - sys_out)
        self.k_gain = np.dot(sys_in[:, np.newaxis], self.erro[-1]) / np.inner(sys_in, sys_in)
        self.weight = self.weight + self.mu * self.k_gain

        return np.reshape(self.weight, (self.weight.shape[0], sys_out.shape[1])).copy()


class LrLMS2(object):
    """description of class"""

    def __init__(self, num_in, num_out, mu=0.01):
        self.mu = mu
        self.por = 0.9
        self.weight = np.zeros([int(self.por*num_in), num_out])
        self.weight2 = 2 * np.random.rand(num_in, int(self.por*num_in)) - 1
        self.erro = []
        self.k_gain = np.asarray([])

    def update(self, sys_target, sys_out, sys_in1):
        sys_in = np.dot(self.weight2.T, sys_in1[:, np.newaxis])
        #print(sys_in.shape)
        self.erro.append(sys_target - sys_out)
        self.k_gain = np.dot(sys_in, self.erro[-1]) / np.dot(sys_in.T, sys_in)
        #print(np.dot(sys_in, self.erro[-1]).shape)
        #print(np.dot(sys_in.T, sys_in).shape)
        #print(self.k_gain.shape)
        self.weight = self.weight + self.mu * self.k_gain
        #print(self.weight.shape)

        #return np.reshape(self.weight, (self.weight.shape[0], sys_out.shape[1])).copy()
        return np.dot(self.weight2, self.weight).copy()


class LrLMS3(object):
    """description of class"""

    def __init__(self, num_in, num_out, mu=0.01):
        self.mu = mu
        self.por = 0.02
        self.weight = 0.01*(2 * np.random.rand(int(self.por*num_in), num_out) - 1) # np.zeros([int(self.por*num_in), num_out])
        self.weight2 = 2 * np.random.rand(num_in-3, int(self.por*num_in)-3) - 1
        self.erro = []
        self.k_gain = np.asarray([])
        self.count = 0
        self.matriz = []

    def update(self, sys_target, sys_out, sys_in1):
        if self.count <= 600:
            self.matriz.append(sys_in1[:-3])
            if self.count == 600:
                matrix = np.asarray(self.matriz)[:, :]
                print(matrix.shape)
                cov = np.dot(matrix.T, matrix)/matrix.shape[0]
                print(cov.shape)
                evals, evecs = np.linalg.eigh(cov)
                idx = np.argsort(evals)[::-1]
                #print(idx)
                evecs = evecs[:, idx]
                evals = evals[idx]
                print(evals[:self.weight2.shape[1]])
                self.weight2 = evecs[:, :self.weight2.shape[1]]

        else:
            sys_in = np.append(np.dot(self.weight2.T, sys_in1[:-3]), sys_in1[-3:])[:, np.newaxis]
            self.erro.append(sys_target - sys_out)
            self.k_gain = np.dot(sys_in, self.erro[-1]) / np.dot(sys_in.T, sys_in)
            self.weight = self.weight + self.mu * self.k_gain

        self.count = self.count + 1
        return np.append(np.dot(self.weight2, self.weight[:-3]), self.weight[-3:])[:, np.newaxis]


class LrLmsBF(object):
    """description of class"""

    def __init__(self, num_in, num_out, mu=0.1):
        self.mu = mu
        self.len = int(np.sqrt(num_in))
        self.weight = np.zeros([num_in, num_out])
        self.weight_h = np.zeros([self.len, 1])
        # self.weight_h[0] = 1
        self.weight_g = np.zeros([self.len, 1]) + 1/self.len
        self.indenty = np.identity(self.len)
        self.erro = []
        self.k_gain = np.asarray([])

    def update(self, sys_target, sys_out, sys_in_):
        sys_in = sys_in_[:, np.newaxis]
        self.erro.append(sys_target - sys_out)
        # sys_in_bf = sys_in.reshape(self.len, -1, order='F')
        # H
        sys_in_h = np.dot(np.kron(self.weight_g, self.indenty).T, sys_in)
        erro_h = sys_target - np.dot(self.weight_h.T, sys_in_h)
        self.weight_h = self.weight_h + self.mu * erro_h * sys_in_h

        # G
        sys_in_g = np.dot(np.kron(self.indenty, self.weight_h).T, sys_in)
        erro_g = sys_target - np.dot(self.weight_g.T, sys_in_g)
        self.weight_g = self.weight_g + self.mu * erro_g * sys_in_g

        return np.kron(self.weight_g, self.weight_h).copy()

class LrLmsTF(object):
    """description of class"""

    def __init__(self, num_in, num_out, mu=0.1):
        self.mu = mu
        self.len = int(num_in**(1/3))
        self.weight = np.zeros([num_in, num_out])
        self.weight_h1 = np.zeros([self.len, 1])
        # self.weight_h[0] = 1
        self.weight_h2 = np.zeros([self.len, 1]) + 1 / self.len
        self.weight_h3 = np.zeros([self.len, 1]) + 1 / self.len
        self.indenty = np.identity(self.len)
        self.erro = []
        self.k_gain = np.asarray([])

    def update(self, sys_target, sys_out, sys_in_):
        sys_in = sys_in_[:, np.newaxis]
        self.erro.append(sys_target - sys_out)
        # sys_in_bf = sys_in.reshape(self.len, -1, order='F')
        # H
        sys_in_h = np.dot(np.kron(self.weight_g, self.indenty).T, sys_in)
        erro_h = sys_target - np.dot(self.weight_h.T, sys_in_h)
        self.weight_h = self.weight_h + self.mu * erro_h * sys_in_h

        # G
        sys_in_g = np.dot(np.kron(self.indenty, self.weight_h).T, sys_in)
        erro_g = sys_target - np.dot(self.weight_g.T, sys_in_g)
        self.weight_g = self.weight_g + self.mu * erro_g * sys_in_g

        # F

        return np.kron(np.kron(self.weight_h3, self.weight_g), self.weight_f).copy()