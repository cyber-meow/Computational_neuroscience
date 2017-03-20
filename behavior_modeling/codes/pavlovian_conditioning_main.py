#! /usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
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
    aux = lambda x : 1 if np.random.random() < p else 0
    aux = np.vectorize(aux)
    rewards = aux(np.empty(n))
    return np.array([np.ones(n), rewards])

# The blocking effect
def blocking(n):
    sample = [[1,1],[0,1],[1,1]]
    return np.repeat(sample, n, axis = 1)

# For the overshadowing effect
def all_present(n, num_stimuli):
    return np.ones(shape = (num_stimuli + 1, n))


# Ad-hoc functions to generate experiment samples and plot figures

def simple_experiment(flag = False):
    trials = simple_conditioning(25)
    if flag:
        plt.plot(trials[0], 'x')
        plt.plot(trials[1], 'o')
    else:
        plt.plot(trials[0], 'x', label = "stimulus")
        plt.plot(trials[1], 'o', label = "reward")
    return trials

#1
def plot_simple_experiment():
    simple_experiment()
    plt.xlabel("trial $n$")
    plt.ylabel("stimulus and reward")
    plt.ylim(-0.35, 1.35)

#2
def plot_simple_simulation():
    trials = np.transpose(simple_experiment())
    RW_model = pavlovian_conditioning()
    RW_model.learn(trials)
    RW_model.plot()
    plt.ylim(-0.2, 1.2)

#3
def plot_ss_epsi():
    trials = np.transpose(simple_experiment(True))
    for epsilon in [0, 0.02, 0.05, 0.1, 0.3, 1]:
        RW_model = pavlovian_conditioning(learning_rate = epsilon)
        RW_model.learn(trials)
        RW_model.plot("$\\epsilon$ = ${}$".format(epsilon))
        plt.ylim(-0.2, 1.2)

#4
def plot_parcond_simulation():
    trials = partial_condtioning(160, 0.4)
    plt.plot(trials[1], '1', label = "reward")
    RW_model = pavlovian_conditioning()
    RW_model.learn(np.transpose(trials))
    RW_model.plot()
    plt.ylim(-0.2, 1.2)

#5
def plot_pcs_epsi():
    trials = np.transpose(partial_condtioning(500,0.4))
    fig, ax = plt.subplots()
    ax.set_color_cycle(["lightblue", "gold", "forestgreen", "purple"])
    for epsilon in reversed([0.01, 0.02, 0.05, 0.1]):
        RW_model = pavlovian_conditioning(learning_rate = epsilon)
        RW_model.learn(trials)
        RW_model.plot("$\\epsilon$ = ${}$".format(epsilon))
        plt.ylim(-0.2, 1.2)

#6
def plot_blocking():
    trials = blocking(25)
    plt.plot(trials[0], 'x', label = "CS1")
    plt.plot(trials[1], '*', label = "CS2")
    RW_model = pavlovian_conditioning(2)
    RW_model.learn(np.transpose(trials))
    RW_model.plot()
    plt.ylim(-0.35, 1.35)

#7
def plot_overshawdowing():
    trials = np.transpose(all_present(50, 2))
    RW_model = pavlovian_conditioning(2, learning_rate = np.array([0.2, 0.1]))
    RW_model.learn(trials)
    RW_model.plot()
    plt.plot(np.ones(50), '--', color = "black")
    plt.ylim(-0.2, 1.2)


cmd_functions = (
[ plot_simple_experiment, plot_simple_simulation, plot_ss_epsi,
  plot_parcond_simulation, plot_pcs_epsi, plot_blocking, plot_overshawdowing ])

usage = "usage: ./pavlovian_conditioning_main.py <1-7>"


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        if n == 6:
            plt.legend(ncol = 2)
        else:
            plt.legend()
        plt.savefig("../figures/pavCond{}".format(n))
        plt.show()

    except:
        print(usage); exit(1)

