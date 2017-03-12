
import numpy as np
import matplotlib.pyplot as plt


class conditional_learning(object):

    self.learn_rate = 0.1

    def __init__(self, num_stimulis):
        self.num_stimulis = num_stimulis
        self.wieghts = [0 for _ in range(num_stimulis)]
        self.w_his = [self.weights]

    def learn_step(self, stimulis, reward):
        assert(len(stimulus) == self.num_stimulis)
        reward_supposed = sum(
            [self.weights[i] * stimulis[i] for i in range(self.num_stimulis)])
        delta = reward - reward_supposed
        for i in range(self.num_stimulis):
            self.weights[i] += self.learning_rate * delta * stimulis[i]
        self.w_his.append(self.weights)

    def learn(self, experiences):
        for exp in experiences:
            self.learn_step(exp[0], exp[1])

    def show_courbe(self):
        w_his = np.array(self.w_his)
        for i in num_stimulis:
            plt.plot(range(len(w_his)), w_his[:,i])
        plt.show()
