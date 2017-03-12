#! /usr/bin/python3

"""
- written in python3
- a number needs to be given as argument in command line in order to 
  show the correspondant picture
"""


import matplotlib.pyplot as plt
import sys
from population_growth import *


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("usage: python3 population_growth.py <1-7>")
        exit(1)

    if sys.argv[1] == '1':
        population = population_growth(2)
        population.grow(100)
        population.savefig("../figures/fig1")
        population.show()

    if sys.argv[1] == '2':
        for alpha in [0.01,0.02,0.05,0.1]:
            population = population_growth(2, alpha = alpha)
            population.grow(100)
            label = "$\\alpha$ = {}".format(alpha)
            population.plot(label = label)
        plt.legend()
        plt.savefig("../figures/fig2")
        plt.show()
    
    if sys.argv[1] == '3':
        for p0 in [2,10,50,100,200]:
            population = population_growth(p0)
            population.grow(100)
            label = "$p_0$ = {}".format(p0)
            population.plot(label = label)
        plt.legend()
        plt.savefig("../figures/fig3")
        plt.show()

    if sys.argv[1] == '4':
        y = [200 - p for p in range(501)]
        plt.plot(range(501), y, label = "$\\alpha = 200 - p$")
        plt.xlabel("population size $p$")
        plt.ylabel("population growth rate $\\alpha$")
        plt.legend()
        plt.savefig("../figures/fig4")
        plt.show()

    if sys.argv[1] == '5':
        alpha_fun = lambda x : 0.001 * x * (200-x)
        population = population_growth(2, alpha_fun = alpha_fun)
        population.grow(100)
        population.savefig("../figures/fig5")
        population.show()

    if sys.argv[1] == '6':
        for k in [0.001, 0.002, 0.005, 0.01]:
            alpha_fun = lambda x : k * x * (200-x)
            population = population_growth(2, alpha_fun = alpha_fun)
            population.grow(100)
            label = r"$\delta_n = " + str(k) + r"p_{n-1}(200-p_{n-1})$"
            population.plot(label = label)
        plt.legend(loc = "lower right")
        plt.savefig("../figures/fig6")
        plt.show()


    if sys.argv[1] == '7':
        for p0 in [2, 10, 50, 100, 200, 250, 300, 400]:
            alpha_fun = lambda x : 0.001 * x * (200-x)
            population = population_growth(p0, alpha_fun = alpha_fun)
            population.grow(100)
            label = "$p_0 = {}$".format(p0)
            population.plot(label = label)
        plt.legend(ncol = 2)
        plt.savefig("../figures/fig7")
        plt.show()
