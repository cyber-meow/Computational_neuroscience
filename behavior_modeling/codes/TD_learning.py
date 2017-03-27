
"""
Implementation of TD-learning a rat-maze experiment
"""


import numpy as np
import matplotlib.pyplot as plt
from utility import set_all_args


# We work on deterministic mazes, so at each state with a particular action
# the agent has probility 1 to go to another specific state.
# In this case the maze is simply a directed graph with reward value for each
# state given, if a vertice is of outdegree 0 it's regarded as a finite state

class maze(object):

    def __init__(self, num_states, edges = None, rewards = None):
        self.num_states = num_states
        if edges is None:
            self.edges = [[] for _ in range(num_states)]
        else:
            assert(len(edges) == num_states)
            self.edges = edges
        if rewards is None:
            self.rewards = np.zeros(num_states)
        else:
            assert(len(rewards) == num_states)
            self.rewards = rewards

    def set_edge(self, s1, s2):
        self.edges[s1].append(s2)

    def set_reward(self, s, r):
        self.rewards[s] = r


class TD_learning(object):

    learning_rate = 0.2

    def __init__(self, maze, proba_func, **kwargs):
        self.maze = maze
        self.proba_func = proba_func
        self.state_v = np.zeros(maze.num_states)
        set_all_args(self, kwargs)
        self.sv_his = [self.state_v.copy()]
        self.trial_his = []

    def one_trial(self):
        curr_state = 0
        state_seq = [0]
        while self.maze.edges[curr_state] != []:
            poss_s = self.maze.edges[curr_state]
            vs = np.array([self.state_v[s] for s in poss_s])
            ps = self.proba_func(vs)
            next_s = np.random.choice(poss_s, p = ps)
            state_seq.append(next_s)
            curr_state = next_s
        self.trial_his.append(state_seq)
        for t,s in enumerate(state_seq):
            nV = 0 if t == len(state_seq)-1 else self.state_v[state_seq[t+1]]
            self.state_v[s] += self.learning_rate * (
                self.maze.rewards[s] + nV - self.state_v[s])
        self.sv_his.append(self.state_v.copy())

    def trials(self, n):
        for _ in range(n):
            self.one_trial()

    def print_visit_times(self):
        res = np.zeros(self.maze.num_states)
        for trial in self.trial_his:
            for s in trial:
                res[s] += 1
        print(res)

    def plot_V(self, labels):
        sv_his = np.transpose(self.sv_his)
        for i, sv in enumerate(sv_his):
            plt.plot(sv, label = labels[i])
        plt.xlabel("trial $n$")
        plt.ylabel("states values $V$")

