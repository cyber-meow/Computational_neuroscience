
import numpy as np
import matplotlib.pyplot as plt

from utility import set_all_args


class AutapseNeuron(object):
    """A Neuron with autapse
    The firing rate of the neuron x is given by the equation
    dx(t)/dt = -x(t) + f(wx(t) + I)
    """

    w = 0.04
    I = -2
    delta_t = 0.1

    def __init__(self, f, x0, **kwargs):
        self.f = f
        self._x = x0
        self._t = 0
        self._x_his = [x0]
        self._t_his = [0]

    @property
    def t(self):
        return self._t

    @property
    def t_his(self):
        return self._t_his

    @property
    def x(self):
        return self._x

    @property
    def x_his(self):
        return self._x_his

    @property
    def x_derivative(self):
        return -self.x + self.f(self.w*self.x+self.I)

    def _step(self):
        self._t += self.delta_t
        self._t_his.append(self.t)
        self._x += self.x_derivative * self.delta_t
        self._x_his.append(self.x)

    def simulate(self, T):
        N = int(T/self.delta_t)
        for _ in range(N):
            self._step()

    def plot_x_his(self, label=None):
        if label == None:
            plt.plot(self.t_his, self.x_his)
        else:
            plt.plot(self.t_his, self.x_his, label=label)
        plt.xlabel("time $t$")
        plt.ylabel("the firing rate of the neuron $x$")

