
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class spike_trains(object):

    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.t_his = []
        self.count_his = []
        self.int_his = []

    def __create_spikes_step(self, firing, t):
        N = int(t/self.delta_t)
        res = []
        proba = firing * self.delta_t
        for t in self.delta_t * np.arange(N+1):
            if np.random.random() < proba:
                if len(res) > 0:
                    self.int_his.append(t - res[-1])
                res.append(t)
        self.t_his.append(res)
        self.count_his.append(len(res))

    def create_spikes(self, firing, t, n = 1):
        for _ in range(n):
            self.__create_spikes_step(firing, t)

    def plot_spikes(self, t, figsize, ofs, xlab = True):
        fig, ax = plt.subplots(figsize = figsize)
        plt.eventplot(self.t_his, colors = [[0,0,0]], lineoffsets = ofs)
        ax.yaxis.set_visible(False)
        plt.margins(0.01,0)
        if xlab:
            plt.xlabel("time (s)")
        plt.tight_layout()

    def plot_counts(self, xmin, xmax, mu, sigma):
        bins = np.arange(xmin,xmax)
        plt.hist(self.count_his, bins, color = "moccasin", normed = 1)
        plt.xlabel("spike count")
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, "r--")
        plt.margins(0.05)
        plt.ylim(ymin = 0)
        plt.xlim(xmin,xmax-1)

    def plot_int(self, loc, scale):
        bins = np.arange(min(self.int_his), max(self.int_his), self.delta_t)
        plt.hist(self.int_his, bins, color = "turquoise", normed = 1, lw = 0)
        plt.xlabel("interspike interval (s)")
        y = expon.pdf(bins, loc, scale)
        plt.plot(bins, y, "--", color = "darkblue")
        plt.margins(0,None)


