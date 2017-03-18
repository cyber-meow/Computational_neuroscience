
"""
Implantation of Rescola-Wagner Model in Python3
"""


import numpy as np
import matplotlib.pyplot as plt


class pavlovian_conditioning(object):

    def __init__(self, num_stimuli = 1, **kwargs):
        self.num_stimuli = num_stimuli
        self.learning_rate = 0.1 * np.ones(num_stimuli)
        self.weights = np.zeros(num_stimuli)
        for k in kwargs.keys():
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
            else:
                print("Warning: parameter name {} not found!".format(k))
        self.w_his = [self.weights.copy()]

    # stimuli must be a np array
    def learn_step(self, stimuli, reward):
        assert(len(stimuli) == self.num_stimuli)
        reward_supposed = np.sum(self.weights * stimuli)
        delta = reward - reward_supposed
        self.weights += delta * self.learning_rate * stimuli
        self.w_his.append(self.weights.copy())

    def learn(self, experiences):
        for exp in experiences:
            self.learn_step(exp[:-1], exp[-1])

    def plot(self, label = "$w$"):
        w_his = np.array(self.w_his)
        if self.num_stimuli == 1:
            plt.plot(range(len(w_his)), w_his[:,0], label = label)
        else:
            for i in range(self.num_stimuli):
                plt.plot(range(len(w_his)), w_his[:,i], 
                         label = "$w^{{({})}}$".format(i+1))
        plt.xlabel("trial $n$")
        plt.ylabel("animal's prediction parameter $w$")

