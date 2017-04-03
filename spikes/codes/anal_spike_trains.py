
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


cell = loadmat("simdata.mat")


def events_matrix(M, T):
    res = []
    for row in M:
        events = []
        for i,t in enumerate(row):
            if t == 1:
                events.append(T[i])
        res.append(events)
    return res
    
def spike_trains_S1():
    trains = cell["spt"][0,1]
    fig, ax = plt.subplots(figsize = (10,2))
    res = events_matrix(trains, cell["t"][0])
    plt.eventplot(res, colors = [[0,0,0]], linestyles = "dashed")
    plt.tight_layout()
    ax.set_axis_bgcolor([0.95]*3)
    spike_trains_post_plot(ax)
    plt.show()

def spike_trains_all():
    res = []
    fig, ax = plt.subplots(figsize = (10,6))
    ypos = 0
    colors = []
    for i,trains in enumerate(cell["spt"][0]):
        res.extend(events_matrix(trains, cell["t"][0]))
        newypos = ypos + trains.shape[0] * 1.3
        if i % 2 == 0:
            plt.axhspan(ypos, newypos, color = [0.95]*3)
            color = [0.2] * 3
        else:
            plt.axhspan(ypos, newypos, color = [0.85]*3)
            color = [0] * 3
        ypos = newypos
        colors.extend([color for _ in range(trains.shape[0])])
    plt.eventplot(res, colors = colors, lineoffsets = 1.3)
    spike_trains_post_plot(ax)
    plt.show()

def spike_trains_post_plot(ax):
    plt.margins(0)
    plt.xlim(-0.2,1000.2)
    plt.xlabel("time (ms)")
    plt.tight_layout()
    ax.invert_yaxis()
    ax.yaxis.set_visible(False)


def mean_var_spikes(M, T, t_start, t_end):
    start = np.searchsorted(T, t_start)
    end = np.searchsorted(T, t_end)
    mean_counts, vas = [], []
    for trains in M:
        simu_spikes = trains[:,start:end]
        spike_counts = np.sum(simu_spikes, axis = 1)
        mean_counts.append(np.mean(spike_counts))
        vas.append(np.var(spike_counts))
    return mean_counts, vas

def mean_var_plot():
    m,v = mean_var_spikes(cell["spt"][0], cell["t"][0], 200, 700)
    plt.scatter(m, v)
    plt.show()

def tunning_curve():
    mean_counts = mean_var_spikes(cell["spt"][0], cell["t"][0], 200, 700)[0]
    firing_rates = [2 * c for c in mean_counts]
    plt.plot(cell["f1"][0], firing_rates)
    plt.xlabel("stimulus frequency (Hz)")
    plt.ylabel("firing rate (spike count / sec)")
    plt.margins(0.05)
    plt.show()
