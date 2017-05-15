#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

import numpy as np
import matplotlib.pyplot as plt

from autapse import AutapseNeuron


def f(s):
    return 50 * (1 + np.tanh(s))

#1
def f_plot(ss):
    plt.plot(ss, f(ss))
    plt.xlabel("input of the neuron $s$")
    plt.ylabel("$f$ $(s)$")

#2 
def x_derivative(xs):
    x_drivatives = list(map(lambda x: AutapseNeuron(f,x).x_derivative, xs))
    plt.plot(xs, x_drivatives)
    plt.xlabel("the firing rate $x$")
    plt.ylabel("the derivative of the firing rate $\dot{x}$")

#3
def x_evolution(x0, T):
    neuron = AutapseNeuron(f, x0)
    neuron.simulate(T)
    neuron.plot_x_his()


cmd_functions = ([
    lambda: f_plot(np.linspace(-10, 10, 1000)),
    lambda: x_derivative(np.linspace(-200, 200, 1000)),
    lambda: x_evolution(49, 100),
    lambda: x_evolution(50, 100),
    lambda: x_evolution(51, 100),
])


if __name__ == "__main__":

    n = int(sys.argv[1])
    cmd_functions[n-1]()
    plt.savefig("../../figures/autapse{}".format(n))
    plt.show()
