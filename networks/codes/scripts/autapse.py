#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

import numpy as np
import matplotlib.pyplot as plt

from network import AutapseNeuron


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
    plt.axhline(y=0, linestyle='--', color='r')
    plt.plot(xs, x_drivatives)
    plt.xlabel("the firing rate $x$")
    plt.ylabel("the derivative of the firing rate $\dot{x}$")

#3,4,5
def x_evolution(x0, T=100, sigma=0, lw=None, label=None):
    neuron = AutapseNeuron(f, x0, sigma=sigma)
    neuron.simulate(T)
    neuron.plot_x_his(lw, label)

#6,7,8
def x_evolution_noise(x0, sigma, T=100):
    for _ in range(5):
        x_evolution(x0, T, sigma, 0.8)

#9
def x_evolution_greater_noise(x0s, sigma, T=100):
    for x0 in x0s:
        x_evolution(x0, T, sigma, 0.6, "$x(0)={}$".format(x0))
    plt.legend()


cmd_functions = ([
    lambda: f_plot(np.linspace(-10, 10, 1000)),
    lambda: x_derivative(np.linspace(-200, 200, 1000)),
    lambda: x_evolution(49),
    lambda: x_evolution(50),
    lambda: x_evolution(51),
    lambda: x_evolution_noise(49, 5),
    lambda: x_evolution_noise(50, 5),
    lambda: x_evolution_noise(51, 5),
    lambda: x_evolution_greater_noise([49, 50, 51], 80),
])


if __name__ == "__main__":

    n = int(sys.argv[1])
    cmd_functions[n-1]()
    plt.savefig("../../figures/autapse{}".format(n))
    plt.show()
