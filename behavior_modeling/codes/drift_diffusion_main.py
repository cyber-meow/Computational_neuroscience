#! /usr/bin/python3


import numpy as np
import matplotlib as plt
import sys
import time
from drift_diffusion import *


def many_runs(n):
    dd = drift_diffusion(1, 0.95, 0.4)
    dd.simulate(1, n)
    dd.plot_curve(1)

def reactime_distr():
    dd = drift_diffusion(1, 0.95, 0.4)
    dd.simulate(1, 1000, False)
    dd.plot_reactime()

def proba_outA(mE):
    dd = drift_diffusion(mE, 0, 0.4)
    t = time.time()
    dd.simulate(1, 1000, False)
    numA = len(dd.reactime_hisA)
    numB = len(dd.reactime_hisB)
    print(mE, numA/(numA+numB), time.time() - t)
    return numA/(numA+numB)

def proba_outA_plot():
    pA = np.vectorize(proba_outA)
    pA_theo = lambda x: 1/(1 + np.exp(0.8/0.25 * (-x)))
    xaxis = np.linspace(-0.2,0.2,100)
    plt.plot(xaxis, pA(xaxis), label = "simulation result")
    plt.plot(xaxis, pA_theo(xaxis), "--", label = "theoretical result")
    plt.xlim(-0.2, 0.2)
    plt.xlabel("evidence for outcome A versus outcome B  $m_E$")
    plt.ylabel("probability of outcome A  $p_A$")
    plt.legend()


cmd_functions = ([ lambda:many_runs(8), reactime_distr, proba_outA_plot ])

usage = "usage ./drift_diffusion_main.py <1-3>"


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(usage); exit(1)

    try:
        n = int(sys.argv[1])
        cmd_functions[n-1]()
        plt.savefig("../figures/Drifdiff{}".format(n))
        plt.show()

    except:
        raise; print(usage); exit(1)

