#! /usr/bin/python3


import numpy as np
import matplotlib as plt
from functools import partial
import sys
from TD_learning import *


# Two possible strategies bases on estimated state values

# Totally random
def rand(V):
    return 1/V.shape[0] * np.ones(V.shape[0])

# Softmax
def softmax(beta, V):
    proba_ = np.exp(beta * V)
    return proba_ / np.sum(proba_)


# For the report

A, B, C, D, E, F, G, H = 0, 1, 2, 3, 4, 5, 6, 7

maze_obj = maze(8)

maze_obj.set_edge(A,B); maze_obj.set_edge(A,C)
maze_obj.set_edge(B,D); maze_obj.set_edge(B,E)
maze_obj.set_edge(C,F); maze_obj.set_edge(C,G)

for a in range(D,H):
    maze_obj.set_edge(a,H)

maze_obj.set_reward(E,5)
maze_obj.set_reward(F,2)

def pre_plot():
    labels = [chr(x) for x in range(ord('A'), ord('H')+1)]
    fig, ax = plt.subplots()
    colors = reversed(plt.cm.hsv(np.linspace(0,1,10)))
    ax.set_color_cycle(colors)
    plt.ylim(-1,6.5)
    return labels

def random_rat():
    TDL = TD_learning(maze_obj, rand)
    TDL.trials(100)
    TDL.print_visit_times()
    labels = pre_plot()
    TDL.plot_V(labels)

def smart_rat(beta):
    TDL = TD_learning(maze_obj, partial(softmax, beta))
    TDL.trials(100)
    TDL.print_visit_times()
    labels = pre_plot()
    TDL.plot_V(labels)


cmd_functions = ([ random_rat, lambda:smart_rat(0.05), 
                   lambda:smart_rat(0.3), lambda:smart_rat(1) ])

usage = "usage ./TD_learning_main.py <1-4>"


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        plt.legend(ncol = 4)
        plt.savefig("../figures/TDlear{}".format(n))
        plt.show()

    except:
        raise; print(usage); exit(1)
