
import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


class poisson_spike_trains(object):

    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.t_his = []
        self.count_his = []
        self.int_his = []

    def _create_spikes_step(self, firing, t):
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

    def create_spikes(self, firing, t, n=1):
        for _ in range(n):
            self._create_spikes_step(firing, t)

    def plot_spikes(self, t, figsize, ofs, xlab=True):
        fig, ax = plt.subplots(figsize=figsize)
        plot_spike_trains(self.t_his, (0,t), ax, ofs, xlab)

    def plot_counts(self, xmin, xmax, mu, sigma):
        bins = np.arange(xmin,xmax)
        plt.hist(self.count_his, bins, color="moccasin", normed=1)
        plt.xlabel("spike count")
        y = mlab.normpdf(bins, mu, sigma)
        plt.plot(bins, y, "r--")
        plt.margins(0.05)
        plt.ylim(ymin=0)
        plt.xlim(xmin, xmax-1)

    def plot_int(self, loc, scale):
        bins = np.arange(min(self.int_his), max(self.int_his), self.delta_t)
        plt.hist(self.int_his, bins, color="turquoise", normed=1, lw=0)
        plt.xlabel("interspike interval (s)")
        y = expon.pdf(bins, loc, scale)
        plt.plot(bins, y, "--", color="darkblue")
        plt.margins(0, None)


# time unit of spike trains must be in second
def plot_spike_trains(trains, ts, ax, ofs, xlab=True):
    ax.eventplot(trains, colors=[[0,0,0]], lineoffsets=ofs)
    ax.yaxis.set_visible(False)
    ax.margins(None, 0.01)
    ax.set_xlim(ts[0], ts[1])
    if xlab:
        ax.set_xlabel("time (s)")
    plt.tight_layout()

# plot several group of spike trains
def plot_spike_train_groups(train_groups, ts, ax, ofs, xlab=True):
    res = []
    ypos = 0
    colors = []
    for i, trains in enumerate(train_groups):
        res.extend(trains)
        newypos = ypos + len(trains) * ofs
        if i%2 == 0:
            plt.axhspan(ypos, newypos, color=[0.95]*3)
            color = [0] * 3
        else:
            color = [0.2] * 3
        ypos = newypos
        colors.extend([color for _ in range(len(trains))])
    plot_spike_trains(res, ts, ax, ofs, xlab)
    plt.margins(None, 0)

