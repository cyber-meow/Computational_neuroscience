#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

import numpy as np
import matplotlib.pyplot as plt

from network import MutualInhibit


def f(s):
    return 50 * (1 + np.tanh(s))

#1
def plot_nullclines(ls='-'):
    MutualInhibit(f, np.r_[0,0]).plot_nullclines(ls=ls)
    plt.legend()

#2
def x_evolution(x0s):
    plot_nullclines('--')
    for x0 in x0s:
        network = MutualInhibit(f, x0)
        network.simulate(100)
        network.plot_x_his()

#3
def vector_field():
    X, Y = np.meshgrid(np.linspace(-50, 150, 15), np.linspace(-50, 150, 15))
    U = -X + f(-0.1*Y+5)
    V = -Y + f(-0.1*X+5)
    Q = plt.quiver(X, Y, U, V, units='xy', scale=20, lw=0.8)
    plt.quiverkey(Q, 0.05, 1.05, 200, '200', labelpos='E')
    MutualInhibit(f, np.r_[0,0]).plot_nullclines(ls='--', lw=0.8)
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.77, 1.12), ncol=2)
    return (lgd,)

x0s = np.array([[-50, -50], [-25, -50], [100, 130], [3, 50], [-40, 150],
                [150, 90], [140, 141], [75, -50]])

cmd_functions = ([
    plot_nullclines,
    lambda: x_evolution(x0s),
    vector_field,
])


if __name__ == "__main__":

    n = int(sys.argv[1])
    art = cmd_functions[n-1]()
    plt.savefig(
        "../../figures/mutual{}".format(n),
        bbox_extra_artists=art, bbox_inches='tight')
    plt.show()
