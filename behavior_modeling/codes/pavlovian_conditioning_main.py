#! /usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import sys
from pavlovian_conditioning import *


# Some differents experiments

# During the first n trials, both stimuli and rewards are presents,
# and during the next n trials, only the stimulus is present
def simple_conditioning(n):
    sample = [[1,1], [1,0]]
    return np.repeat(sample, n, axis = 1)

# The partial conditioning experiment
def partial_condtioning(n, p):
    aux = lambda x : 1 if np.random.random() < 0.4 else 0
    aux = np.vectorize(aux)
    rewards = aux(np.empty(n))
    return np.array([np.ones(n), rewards])


# Ad-hoc functions to generate experiment samples and plot figures

def simple_experiment(flag = False):
    trials = simple_conditioning(25)
    if flag:
        plt.plot(np.arange(50), trials[0], 'x')
        plt.plot(np.arange(50), trials[1], 'o')
    else:
        plt.plot(np.arange(50), trials[0], 'x', label = "stimuli")
        plt.plot(np.arange(50), trials[1], 'o', label = "rewards")
    return trials

def plot_simple_experiment():
    simple_experiment()
    plt.xlabel("trial $n$")
    plt.ylabel("stimuli and rewards")
    plt.ylim(-0.35, 1.35)

def plot_simple_simulation():
    trials = np.transpose(simple_experiment())
    RW_model = pavlovian_conditioning()
    RW_model.learn(trials)
    RW_model.plot()
    plt.ylim(-0.2, 1.2)

def plot_ss_epsi():
    trials = np.transpose(simple_experiment(True))
    for epsilon in [0, 0.02, 0.05, 0.1, 0.3, 1]:
        RW_model = pavlovian_conditioning(learning_rate = epsilon)
        RW_model.learn(trials)
        RW_model.plot("$\\epsilon$ = {}".format(epsilon))
        plt.ylim(-0.2, 1.2)
        

cmd_functions = (
[plot_simple_experiment, plot_simple_simulation, plot_ss_epsi])

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: ./pavlovian_conditioning_main.py <1-?>")
        exit(1)

    n = int(sys.argv[1])
    cmd_functions[n-1]()
    plt.legend()
    plt.savefig("../figures/pavCond{}".format(n))
    plt.show()

    if sys.argv[1] == '4':
        trials = partial_condtioning(2000, 0.4)
        plt.plot(np.arange(2000), trials[1], '1', label = "reward")
        RW_model = pavlovian_conditioning(1, learning_rate = 0.01)
        RW_model.learn(np.transpose(trials))
        RW_model.plot()
        plt.ylim(-0.2, 1.2)
        plt.show()
