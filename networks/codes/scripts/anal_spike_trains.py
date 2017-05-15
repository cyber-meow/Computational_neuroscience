#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sys

from spike_train import plot_spike_trains, plot_spike_train_groups


cell = loadmat("simdata.mat")

# The result time is in s
def events_matrix(M, T):
    res = []
    for row in M:
        events = []
        for i,s in enumerate(row):
            if s == 1:
                events.append(T[i]*1e-3)
        res.append(events)
    return res
    
def spike_trains_S1():
    trains = cell["spt"][0,0]
    fig, ax = plt.subplots(figsize = (10,2))
    res = events_matrix(trains, cell["t"][0])
    plot_spike_trains(res, (-0.001, 1.001), ax, 1.5)
    ax.invert_yaxis()

def spike_trains_all():
    fig, ax = plt.subplots(figsize = (10,6))
    res = [events_matrix(trains, cell["t"][0]) for trains in cell["spt"][0]]
    plot_spike_train_groups(res, (-0.001, 1.001), ax, 1.3)
    ax.invert_yaxis()


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
    f.write(str(cell["f1"][0]) + "\n")
    f.write(str(mean_counts) + "\n")
    f.write(str(s))
    firing_rates = [2 * c for c in mean_counts]
    plt.plot(cell["f1"][0], firing_rates)
    plt.xlabel("stimulus frequency (Hz)")
    plt.ylabel("firing rate (spikes/sec)")
    plt.margins(0.05)


cmd_functions = ([ spike_trains_S1, spike_trains_all, tunning_curve ])

usage = "usage: ./anal_spike_trains.py <1-3>"


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        plt.savefig("../../figures/analSt{}".format(n))
        plt.show()

    except:
        raise; print(usage); exit(1)
