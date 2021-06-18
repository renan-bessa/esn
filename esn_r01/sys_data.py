import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy


class SysData(object):
    """description of class"""

    def __init__(self, sys_data, train_percent=0.5):
        # """
        self.sys_data = sys_data.copy()
        self.sys_target = sys_data.copy()
        self.final_point = self.sys_data.shape[0]
        self.train_point = int(train_percent * self.sys_data.shape[0])

        self.num_series = self.sys_data.shape[1]

        # Database normalization
        self.mean_data = np.mean(self.sys_data[:self.train_point, :], axis=0)
        self.std_data = 3 * np.std(self.sys_data[:self.train_point, :], axis=0)
        for step in range(self.num_series):
            self.sys_data[:, step] = (self.sys_data[:, step] - self.mean_data[step]) / self.std_data[step]

        # #########################################3

        fig, axs = plt.subplots(nrows=2, ncols=1)
        axs[0].plot(self.sys_data, marker='.', linewidth=0.6)
        axs[1].plot(self.sys_target, marker='.', linewidth=0.6)


