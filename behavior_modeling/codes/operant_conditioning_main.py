#! /usr/bin/python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from operant_conditioning import *


#1
def softmax_beta():
    beta = np.linspace(0,5,1000)
    pbneg2 = 1/(1 + np.exp(beta * (-2)))
    pb2 = 1/(1 + np.exp(beta * 2))
    plt.plot(beta, pbneg2, label = "$m_y - m_b$ = $-2$", color = 'b')
    plt.plot(beta, pb2, label = "$m_y - m_b$ = $2$", color = 'y')
    plt.xlabel("expoitation-exploration trade-off parameter $\\beta$")
    plt.ylabel("probability to choose blue flowers $p_b$")
    plt.legend()

#2
def softmax_diff():
    diff = np.linspace(-30,30,10000)
    pb2 = 1/(1 + np.exp(0.2 * diff))
    pb5 = 1/(1 + np.exp(0.5 * diff))
    plt.plot(diff, pb2, label = "$\\beta$ = $0.2$")
    plt.plot(diff, pb5, label = "$\\beta$ = $0.5$")
    plt.xlabel("difference $m_y - m_b$")
    plt.ylabel("probability to choose blue flowers $p_b$")
    plt.legend()

def flower_nectars():
    sample = [[8,2],[2,8]]
    return np.repeat(sample, 100, axis = 1)

# choice_his must be a numpy array
def plot_behaviors(choice_his, ax):
    blues = np.arange(200)[choice_his == 0]
    yellows = np.arange(200)[choice_his == 1]
    ax.eventplot(blues, colors = 'b', linewidths = 1.5)
    ax.eventplot(yellows, colors = 'y', linewidths = 1.5)

#3,4
def dumb_bee(beta):
    ist = np.array([0.,5.])
    opercond = operant_conditioning(
                tolearn = False, inter_states = ist, beta = beta)
    samples = flower_nectars()
    opercond.learn(np.transpose(samples))
    fig, ax = plt.subplots(figsize = (10,1))
    plot_behaviors(np.array(opercond.choice_his), ax)
    ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.xlim(-1, 200)
    #plt.axis("off")
    

#5,6,7
def smart_bee(beta):
    
    ist = np.array([0.,5.])
    opercond = operant_conditioning(inter_states = ist, beta = beta)
    samples = flower_nectars()
    opercond.learn(np.transpose(samples))
    gs = GridSpec(3, 1,  height_ratios = [5,1,5])

    ax0 = plt.subplot(gs[0])
    ax0.plot(samples[0], '.', label = "$r_b$", color = 'b')
    ax0.xaxis.set_visible(False)
    opercond.plot_inter(0, ax0, "$m_b$", 'r')
    
    ax1 = plt.subplot(gs[1])
    plot_behaviors(np.array(opercond.choice_his), ax1)
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    
    ax2 = plt.subplot(gs[2])
    ax2.plot(samples[1], '.', label = "$r_y$", color = 'y')
    opercond.plot_inter(1, ax2, "$m_y$", 'g')
    
    for ax in [ax0, ax2]:
        ax.set_ylim(-1,11)
        ax.legend(ncol = 2)
    plt.tight_layout()


cmd_functions = (
[ softmax_beta, softmax_diff, lambda:dumb_bee(0), lambda:dumb_bee(0.8),
  lambda:smart_bee(0.2), lambda:smart_bee(0), lambda:smart_bee(1)])

usage = "usage: ./operant_conditioning.py <1-?>"

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        plt.savefig("../figures/insCond{}".format(n))
        plt.show()

    except:
        raise; print(usage); exit(1)

