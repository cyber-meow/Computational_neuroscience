#! /usr/bin/python3

import sys
from spike_train_poisson import *


def save_show(n):
    plt.savefig("../figures/stPoisson{}".format(n))
    plt.show()

#1 
def poisson_process():
    st = spike_trains(1)
    st.create_spikes(1/4, 1000)
    st.plot_spikes(1000, (10,1.2), 1, False)
    save_show(1)

#2
def spike_train():
    st = spike_trains(2e-3)
    st.create_spikes(25, 1)
    st.plot_spikes(1, (10, 1.4), 1)
    save_show(2)

#3
def spike_trains_n(n, fignum):
    st = spike_trains(2e-3)
    st.create_spikes(25, 1, n)
    st.plot_spikes(1, (10, 3), 1.2)
    save_show(fignum)
    st.plot_counts(10, 41, 25, 4.87) # n*mu, sqrt(n*sigma^2)
    save_show(fignum + 1)
    st.plot_int(1e-3, 1/25.64) # exp(-0.02*lambda) = 0.95
    save_show(fignum + 2)


cmd_functions = ([ poisson_process, spike_train, 
                   lambda : spike_trains_n(50,3), 
                   lambda : spike_trains_n(500,6) ])

usage = "usage: ./spike_train_poisson_main.py <1-4>"

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()

    except:
        raise; print(usage); exit(1)
