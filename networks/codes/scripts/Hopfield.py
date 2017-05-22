#! /usr/bin/python3

import sys
import os
sys.path.insert(0, os.path.abspath('../models'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib .animation as animation

from network import Network


# x is a vector of length n*n
def plot_neurons(x, n=8):
    assert x.shape == (n**2,)
    X = x.reshape(n,n)
    plt.imshow(X, cmap='gray', interpolation='none')

def random_pattern(N=64):
    return np.random.choice([1,-1], N)

def W_store(p):
    N = p.shape[0]
    W = np.outer(p,p)/N
    return W


def pattern_evolution(W, picname):
    W = W_store(x)
    x0 = random_pattern(64)
    Hopfield = Network(64, np.sign, W, np.zeros(64), x0)
    fig, ax = plt.subplots()
    x_old = Hopfield.x.copy()
    i = 0
    while True:
        plot_neurons(Hopfield.x)
        plt.savefig("{}_{}".format(picname, i), bbox_inches='tight')
        plt.show()
        i += 1
        Hopfield.simulate(1)
        if np.linalg.norm(Hopfield.x - x_old) < 0.1:
            break
        x_old = Hopfield.x.copy()

def one_pattern(picname):
    x = random_pattern()
    W = W_store(x)
    plot_neurons(x)
    plt.savefig("{}_p".format(picname), bbox_inches='tight')
    plt.show()
    pattern_evolution(W, picname)


cmd_functions = ([
    lambda: one_pattern("../figures/hopfield_one")
])


if __name__ == "__main__":

    n = int(sys.argv[1])
    cmd_functions[n-1]()
