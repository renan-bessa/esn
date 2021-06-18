import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


class LorenzSys(object):
    """description of class"""

    def __init__(self, start, stop, step, state_0=[1.0, 1.0, 1.0], rho=28.0, sigma=10.0, beta=8.0/3.0):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.state_0 = state_0
        self.t = np.arange(start, stop, step)
        self.states = self.get_states()

    def lorenz_f(self, state, t):
        x, y, z = state  # Unpack the state vector
        return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z  # Derivatives

    def get_states(self):
        return odeint(self.lorenz_f, self.state_0, self.t)


class RationalJerks(object):
    """description of class"""

    def __init__(self, start, stop, step, state_0=[46, 1.0, 10], alpha=10.3):
        self.alpha = alpha
        self.state_0 = state_0
        self.t = np.arange(start, stop, step)
        self.states = self.get_states()

    def lorenz_f(self, state, t):
        x, y, z = state  # Unpack the state vector
        return z, -self.alpha * y + z, -x + x * y  # Derivatives

    def get_states(self):
        return odeint(self.lorenz_f, self.state_0, self.t)


class Windmi(object):
    """description of class"""

    def __init__(self, start, stop, step, state_0=[0, 0.8, 0], alpha=0.7, beta=2.5):
        self.alpha = alpha
        self.beta = beta
        self.state_0 = state_0
        self.t = np.arange(start, stop, step)
        self.states = self.get_states()

    def lorenz_f(self, state, t):
        x, y, z = state  # Unpack the state vector
        return y, z, -self.alpha * z - y + self.beta - np.exp(x)  # Derivatives

    def get_states(self):
        return odeint(self.lorenz_f, self.state_0, self.t)