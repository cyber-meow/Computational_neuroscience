
"""
- numpy is not used here because it's not necessary and 
  it's simpler using list for this question
"""

import matplotlib.pyplot as plt


class population_growth(object):

    # alpha_fun is a function that gives the growth of the population 
    # according to the size of the current population
    # by default we pose self.alpha_fun = lambda x : (1+self.alpha) * x
    alpha = 0.1
    alpha_fun = None

    def __init__(self, p0, **kwargs):
        self.p = [p0]
        for k in list(kwargs.keys()):
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
            else:
                print("Warning: parameter name {} not found!".format(k))
        if self.alpha_fun is None:
            self.alpha_fun = lambda x : (1 + self.alpha) * x
    
    def grow(self, num_year):
        for _ in range(num_year):
            self.p.append(self.p[-1] + self.alpha_fun(self.p[-1]))

    def plot(self, label):
        if label is None:
            plt.plot(range(len(self.p)), self.p)
        else:
            plt.plot(range(len(self.p)), self.p, label=label)
        plt.xlabel("years $n$")
        plt.ylabel("population size $p$")
    
    def show(self, label = None):
        self.plot(label)
        if label is not None:
            plt.legend()
        plt.show()
        
    def savefig(self, pathname, label = None):
        self.plot(label)
        if label is not None:
            plt.legend()
        plt.savefig(pathname)

