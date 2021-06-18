import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from synthetic_systems import *
from sys_data import *
"""
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 40.0, 0.01)

states = odeint(f, state0, t)
"""

lorenz = LorenzSys(0.0, 70, 0.01)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.plot(lorenz.states[:, 0], lorenz.states[:, 1], lorenz.states[:, 2])

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(9, 7))
axs[0].plot(lorenz.states[:, 0])
axs[1].plot(lorenz.states[:, 1])
axs[2].plot(lorenz.states[:, 2])

data_lorenz = SysData(lorenz.states[:, 0][:, np.newaxis])
