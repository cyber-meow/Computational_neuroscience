#! /usr/bin/python3

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sys


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
    plt.eventplot(res, colors = [[0,0,0]], lineoffsets = 1.5)
    spike_trains_post_plot(ax)

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
            color = [0] * 3
        else:
            color = [0.2] * 3
        ypos = newypos
        colors.extend([color for _ in range(trains.shape[0])])
    plt.eventplot(res, colors = colors, lineoffsets = 1.3)
    spike_trains_post_plot(ax)

def spike_trains_post_plot(ax):
    plt.margins(0)
    plt.xlim(-0.2,1000.2)
    plt.xlabel("time (ms)")
    plt.tight_layout()
    ax.invert_yaxis()
    ax.yaxis.set_visible(False)


def mean_std_spikes(M, T, t_start, t_end):
    start = np.searchsorted(T, t_start)
    end = np.searchsorted(T, t_end)
    mean_counts, stds = [], []
    for trains in M:
        simu_spikes = trains[:,start:end]
        spike_counts = np.sum(simu_spikes, axis = 1)
        mean_counts.append(np.mean(spike_counts))
        stds.append(np.std(spike_counts))
    return mean_counts, stds

def tunning_curve():
    mean_counts, s = mean_std_spikes(cell["spt"][0], cell["t"][0], 200, 700)
    f = open("../figures/analStMeanStd", 'w')
    f.write(str(mean_counts) + "\n")
    f.write(str(s))
    firing_rates = [2 * c for c in mean_counts]
    plt.plot(cell["f1"][0], firing_rates)
    plt.xlabel("stimulus frequency (Hz)")
    plt.ylabel("firing rate (spike count / sec)")
    plt.margins(0.05)


cmd_functions = ([ spike_trains_S1, spike_trains_all, tunning_curve ])

usage = "usage: ./anal_spike_trains.py <1-3>"


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        plt.savefig("../figures/analSt{}".format(n))
        plt.show()

    except:
        raise; print(usage); exit(1)
