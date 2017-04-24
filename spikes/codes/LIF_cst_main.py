#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys
from LIF import *
from spike_train_poisson import plot_spike_trains


# Constant input

#1
def cst_current(I, spiking, delta_t, ax=None, label=None, xylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    neuron = LIFICst(I, spiking, delta_t=delta_t)
    neuron.computeV(0.1)
    neuron.plot_V(ax, label, xylabel)

#2
def cst_currents(Is, spiking, delta_t):
    fig, ax = plt.subplots()
    for I in Is:
        cst_current(I, spiking, delta_t, ax, "$I = {}$ nA".format(I))
    plt.legend(loc='best')

#3
def cst_current_dts(I, spiking, delta_ts):
    fig, ax = plt.subplots()
    for dt in delta_ts:
        cst_current(I, spiking, dt, ax, "$\\Delta t = {}$ ms".format(dt))
    plt.legend(loc='best')

#4
def anal_nume_cmp():
    fig, ax = plt.subplots()
    neuron_anal = LIFNumeric(1)
    neuron_anal.computeV(0.1)
    neuron_anal.plot_V(ax, "analytic solution")
    neuron_nume = LIFICst(1, False)
    neuron_nume.computeV(0.1)
    neuron_nume.plot_V(ax, "numerical solution")
    plt.legend(loc='best')


def cst_current_spike(I, ax, title):
    cst_current(I, True, 0.1, ax, xylabel=False)
    ax.set_title(title)
    ax.title.set_position([.5, 1.03])

#5
def LIF_spikes():
    fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex='col', sharey='row')
    Is = [0.5, 1, 2, 4]
    for i, ax in enumerate(axs.reshape(4)):
        cst_current_spike(Is[i], ax, "$I = {}$ nA".format(Is[i]))
    fig.text(0.51, 0.04, "time $t$ (s)", ha='center', va='center')
    fig.text(0.06, 0.5, "membrane potential $V$ (mV)", 
             ha='center', va='center', rotation='vertical')


#6
def tunning_curve_cst(Is):
    num_spikes = []
    firing_rate_div10 = []
    for I in Is:
        neuron = LIFICst(I, True, delta_t=0.1)
        neuron.computeV(0.1)
        num_spikes.append(len(neuron.spike_moments))
        firing_rate_div10.append(neuron.firing_rate/10)
    plt.plot(Is, num_spikes, label="simulated")
    plt.plot(Is, firing_rate_div10, label="theoretical")
    plt.xlabel("input current $I$ (nA)")
    plt.ylabel("number of spikes within 100 ms")
    plt.legend(loc='best')


#7
def cst_current_ref(I, delta_t, ax=None, label=None, xylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    neuron = LIFICstRefractory(I, True, delta_t=delta_t)
    neuron.computeV(0.1)
    neuron.plot_V(ax, label, xylabel)

#8
def tunning_ref(Is):
    firing_rate_nor = [LIFICst(I, True).firing_rate for I in Is]
    firing_rate_ref = [LIFICstRefractory(I, True).firing_rate for I in Is]
    plt.plot(Is, firing_rate_nor, label="LIF without refractory period")
    plt.plot(Is, firing_rate_ref, label="LIF with refractory period")
    plt.xlabel("input current $I$ (nA)")
    plt.ylabel("firing rate $f_{firing}$ (Hz)")
    plt.legend(loc='best')


#9
def cst_current_noise(I, delta_t, ax=None, label=None, xylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    neuron = LIFRefractoryNoise(lambda t: I, True, delta_t=delta_t)
    neuron.computeV(0.1)
    neuron.plot_V(ax, label, xylabel)

#10
def LIF_noise_trains():
    fig, axs = plt.subplots(4, 1, figsize=(10,5), sharex='col')
    for i,sigma in enumerate([0.1, 1, 2, 5]):
        spike_trains = []
        for _ in range(10):
            neuron = LIFRefractoryNoise(
                lambda t: 1, True, delta_t=0.1, sigma=sigma)
            neuron.computeV(0.5)
            spike_trains.append(neuron.spike_moments)
        plot_spike_trains(
            spike_trains, (-0.001, 0.501), axs[i], 1.3, xlab=False)
        axs[i].set_title("$\\sigma = {}$".format(sigma))
        axs[i].title.set_position([0.5, 1.03])
    plt.xlabel("time $t$ (s)")
    plt.tight_layout()


# Oscillating input part

def I_data(f):
    def I(t):
        if 0.2 <= t < 0.7:
            return 2 + np.cos(np.pi*f*(t-200))
        return 0
    return I

def data_trains(f):
    fig, ax = plt.subplots(figsize=(10,2))
    spike_trains = []
    for _ in range(10):
        neuron = LIFRefractoryNoise(
            I_data(f), True, delta_t=0.1, sigma=1, Vth=-55)
        neuron.computeV(1)
        spike_trains.append(neuron.spike_moments)
    plot_spike_trains(spike_trains, (-0.001, 1.001), ax, 1.3)

def data_current(f):
    fig, ax = plt.subplots()
    neuron = LIFRefractory(
        I_data(f), True, delta_t=0.1, Vth=-55)
    neuron.computeV(1)
    neuron.plot_V(ax)


cmd_functions = (
    [ lambda : cst_current(1, False, 1),
      lambda : cst_currents([3,2,1], False, 1),
      lambda : cst_current_dts(1, False, [0.1, 1, 5, 10]),
      anal_nume_cmp, 
      LIF_spikes,
      lambda : tunning_curve_cst(np.linspace(0,2,100)),
      lambda : cst_current_ref(1, 0.1),
      lambda : tunning_ref(np.linspace(0,5,1000)),
      lambda : cst_current_noise(1, 0.1),
      LIF_noise_trains,
      lambda : data_trains(27.7), 
      lambda : data_current(27.7) ])


if __name__ == "__main__":

    n = int(sys.argv[1])
    cmd_functions[n-1]()
    plt.savefig("../figures/LIF{}".format(n))
    plt.show()
