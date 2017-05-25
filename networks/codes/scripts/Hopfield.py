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
<<<<<<< HEAD
    plt.imshow(X, cmap='gray', interpolation='none')
=======
    plt.imshow(X, cmap='gray', interpolation='none', vmin=-1, vmax=1)
>>>>>>> report

def random_pattern(N=64):
    return np.random.choice([1,-1], N)

<<<<<<< HEAD
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
=======
def W_store(ps):
    N = ps[0].shape[0]
    W = np.zeros((N,N))
    for p in ps:
        W += np.outer(p,p)/N
    return W

# x not modified
def mutation(x, prob):
    x_copy = x.copy()
    for i in range(len(x)):
        if np.random.random() < prob:
            x_copy[i] = -x[i]
    return x_copy

def new_fig_off_axis():
    fig, ax = plt.subplots()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return fig, ax


class pattern_evolution(object):
    """press c to do another trial, q to leave"""

    def __init__(self, f, W, x0s, ts=10, sigma=0.1, picname=None, save=False):
        self.f = f
        self.W = W
        self.x0s = x0s
        self.time_step = ts
        self.picname = picname
        self.save = save
        self.i, self.j = 0, 0
        self.next_trial = False
        self.end = False

    def init_trial(self):
        x0 = self.x0s[self.i] if self.i<len(self.x0s) else random_pattern(64)
        self.Hopfield = Network(
            64, self.f, self.W, np.zeros(64), x0, sigma=sigma)
    
    def press(self, event):
        if event.key == 'c':
            self.next_trial = True
        if event.key == 'q':
            self.end = True

    def one_plot(self):
        plot_neurons(self.Hopfield.x)
        if self.save:
            assert self.picname is not None
            plt.savefig("{}_{}_{}".format(self.picname, self.i, self.j), 
                        bbox_inches='tight')
        plt.show()
        while not plt.waitforbuttonpress(1): pass
        if self.next_trial:
            self.i += 1; self.j = 0
            self.init_trial()
            self.next_trial = False
        else:
            self.Hopfield.simulate(self.time_step/10)
            self.j += 1

    def run(self):
        print("press c to do a next trial and press q to quit, and any other"
              " key to continue")
        fig, ax = new_fig_off_axis()
        fig.canvas.mpl_connect('key_press_event', self.press)
        self.init_trial()
        while not self.end: self.one_plot()


def patterns(n, f, ts=10, prob=0.15, noise=0.1, picname=None, save=False):
    xs = [random_pattern() for _ in range(n)]
    W = W_store(xs)
    x0s = [np.zeros(64)]
    plt.ion()
    for i, x in enumerate(xs):
        x0s.extend([mutation(x, prob), mutation(-x, prob)])
        new_fig_off_axis()
        plot_neurons(x)
        if save:
            assert picname is not None
            plt.savefig("{}_p{}".format(picname, i), bbox_inches='tight')
        plt.show()
        while not plt.waitforbuttonpress(1): pass
    pattern_evolution(f, W, x0s, ts, noise, picname=picname, save=save).run()

>>>>>>> report


if __name__ == "__main__":

<<<<<<< HEAD
    n = int(sys.argv[1])
    cmd_functions[n-1]()
=======
    print("how many patterns?")
    n = int(input())
    assert n > 0
    
    print("noise level?")
    sigma = float(input())
    assert sigma >= 0

    print("mutation probabilty for initial conditions?")
    prob = float(input())
    assert 0 <= prob <= 1

    print("time steps between two picutres")
    ts = int(input())

    print("sgn or tanh for the activation function? (1 or 2)")
    while True:
        try:
            k = int(input())
            assert k in [1, 2]
            break
        except:
            print("please enter 1 or 2")
    f = np.sign if k == 1 else np.tanh

    print("save? (y or n)")
    while True:
        r = input()
        if r in ['y', 'yes']:
            print("please give the prifix (path included) of the picture name")
            patterns(n, f, ts, prob, sigma, input(), True)
            break
        elif r in ['n', 'no']:
            patterns(n, f, ts, prob, sigma)
            break
        else:
            print("please enter y or n")
>>>>>>> report
