#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

import numpy as np
import matplotlib.pyplot as plt
import sys

from LIF import *
from spike_train import plot_spike_trains


# Sinusoidal input

def I_sin(f):
    def I(t): return 1 + np.sin(2*np.pi*f*t)
    return I

#1
def stimuli_plot():
    fig, axs = plt.subplots(3, 1, sharex='col')
    t = np.linspace(0, 1, 10000)
    for i, f in enumerate([1, 5, 40]):
        I = I_sin(f)(t)
        axs[i].plot(t, I)
        axs[i].margins(None, 0.02)
        axs[i].set_title("$f = {}$ Hz".format(f))
        axs[i].title.set_position([.5, 1.03])
    plt.xlabel("time $t$ (s)")
    axs[1].set_ylabel("input current $I$")
    plt.tight_layout()

#2
def sin_current(f, C, spiking, ax=None, xylabel=True):
    if ax is None:
        fig, ax = plt.subplots()
    # recall that tau_m = R * C * 1e-3 (s)
    neuron = LIF(I_sin(f), spiking, delta_t=0.1, EL=0, Vth=1, R=1, C=C)
    neuron.computeV(1)
    neuron.plot_V(ax, xylabel=xylabel, unit=False)

#3, 4
def sin_current_spikes(C):
    fig, axs = plt.subplots(3, 1, sharex='col')
    for i, f in enumerate([1, 5, 40]):
        sin_current(f, C, True, axs[i], xylabel=False)
        axs[i].set_title("$f = {}$ Hz".format(f))
        axs[i].title.set_position([.5, 1.03])
    plt.xlabel("time $t$ (s)")
    axs[1].set_ylabel("membrane potential $V$")
    plt.tight_layout(h_pad=-0.2)

#5
def tunning_curve_sin(fs):
    num_spikes = []
    for f in fs:
        neuron = LIF(I_sin(f), delta_t=0.1, EL=0, Vth=1, R=1, C=100)
        neuron.computeV(25)
        num_spikes.append(len(neuron.spike_moments)/25)
    plt.plot(fs, num_spikes)
    plt.xlim(0.99, 40.01)
    plt.xlabel("input current frequency $f$ (Hz)")
    plt.ylabel("firing rate $f_{firing}$ (Hz)")

#6
def sin_trains():
    fig, ax = plt.subplots(figsize=(10,3))
    spike_trains = []
    for f in [1, 2, 5, 10, 20, 40, 100]:
        neuron = LIF(I_sin(f), delta_t=0.1, EL=0, Vth=1, R=1, C=100)
        neuron.computeV(2)
        spike_trains.append(neuron.spike_moments)
    plot_spike_trains(spike_trains, (-0.001, 2.001), ax, 1.3)
    ax.invert_yaxis()

# t in s, the larger is t, the better is the precision 
def compute_frequency_spectrum(f, t_max):
    spikes = np.zeros(t_max*10000+1)
    neuron = LIF(I_sin(f), delta_t=0.1, EL=0, Vth=1, R=1, C=100)
    neuron.computeV(t_max)
    for spike_t in neuron.spike_moments:
        spikes[int(spike_t * 10000)] = 1
    spike_counts = np.convolve(spikes, np.hanning(3500), 'same')    
    # fig = plt.figure(1)
    # plt.plot(np.linspace(0, t_max, t_max*10000+1), spike_counts)    
    # fig.show()
    # this represents frequency from 0 to 10000 Hz
    return np.abs(np.fft.fft(spike_counts))

#7 
def plot_frequency_spectrum(f, t_max):
    fs = compute_frequency_spectrum(f, t_max)
    # f = plt.figure(2)
    plt.plot(np.linspace(0, 50, t_max*50+1), fs[:t_max*50+1])
    print((np.argmax(fs[1:t_max*5000+1])+1)/t_max)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("amplitude")
    
#8
def tunning_curve_sin2(fs, t_max):
    lowest_f_comp = []
    for f in fs:
        print(f)
        response_fs = compute_frequency_spectrum(f, t_max)
        lowest_f_comp.append((np.argmax(response_fs[1:t_max*5000+1])+1)/t_max)
    plt.plot(fs, lowest_f_comp)
    plt.xlim(0.99, 40.01)
    plt.xlabel("input current frequency $f$ (Hz)")
    plt.ylabel("frequency of the neuron response $f_r$ (Hz)")
    
    
cmd_functions = (
    [ stimuli_plot,
      lambda : sin_current(1, 100, False),
      lambda : sin_current_spikes(100),
      lambda : sin_current_spikes(10),
      lambda : tunning_curve_sin(np.linspace(1, 40, 400)),
      sin_trains,
      lambda : plot_frequency_spectrum(20, 20),
      lambda : tunning_curve_sin2(np.linspace(1, 40, 400), 10) ])


if __name__ == "__main__":

    n = int(sys.argv[1])
    cmd_functions[n-1]()
    plt.savefig("../../figures/LIFSin{}".format(n))
    plt.show()
