
"""
Implementation of a simple instrumental learning model
The policy is given by the softmax strategy
"""


import numpy as np
from utility import set_all_args 

class operant_conditioning(object):

    beta = 0.2
    tolearn = True

    # We must have "dtype = float" for all the arrays here (inter_states!)
    def __init__(self, num_stimuli = 2, **kwargs):
        self.num_stimuli = num_stimuli
        self.learning_rate = 0.2 * np.ones(num_stimuli)
        self.inter_states = np.zeros(num_stimuli)
        set_all_args(self, kwargs)
        self.choice_his = []
        self.proba_ = np.exp(self.beta * self.inter_states)
        self.proba_nor = self.proba_ / np.sum(self.proba_)
        self.m_his = [self.inter_states.copy()]

    def learn_step(self, rewards):
        ch = np.random.choice(self.num_stimuli, p = self.proba_nor)
        self.choice_his.append(ch)
        if self.tolearn:
            delta = rewards[ch] - self.inter_states[ch]
            self.inter_states[ch] += self.learning_rate[ch] * delta
            self.proba_[ch] = np.exp(self.beta * self.inter_states[ch])
            self.proba_nor = self.proba_ / np.sum(self.proba_)
            self.m_his.append(self.inter_states.copy())

    def learn(self, experiments):
        for exp in experiments:
            self.learn_step(exp)

    def plot_inter(self, i, ax, label, color):
        assert(self.tolearn)
        m_his = np.array(self.m_his)
        ax.plot(m_his[:,i], label = label, color = color)
