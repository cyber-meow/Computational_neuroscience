
import numpy as np
import matplotlib.pyplot as plt

from utility import set_all_args, add_arrow


class Network(object):

    delta_t = 0.1
    sigma = 0

    def __init__(self, d, f, W, I, x0, **kwargs):
        assert W.shape == (d, d)
        assert I.shape == x0.shape == (d,)
        self.d = d
        self.f = f
        self.W = W
        self.I = I
        self._x = x0.astype(float)
        self._t = 0
        self._x_his = [x0.copy()]
        self._t_his = [0]
        set_all_args(self, kwargs)

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
        return -self.x + self.f(np.dot(self.W, self.x) + self.I)

    def _step(self):
        self._t += self.delta_t
        self._t_his.append(self.t)
        self._x += (
            self.x_derivative * self.delta_t
            + self.sigma * np.random.randn(self.d) * np.sqrt(self.delta_t))
        self._x_his.append(self.x.copy())

    def simulate(self, T):
        N = int(T/self.delta_t)
        for _ in range(N):
            self._step()


class AutapseNeuron(Network):
    """A Neuron with autapse
    The firing rate of the neuron x is given by the equation
    dx(t)/dt = -x(t) + f(wx(t) + i)
    """

    w = 0.04
    i = -2

    def __init__(self, f, x0, **kwargs):
        set_all_args(self, kwargs)
        super().__init__(
            1, f, np.array([[self.w]]), np.r_[self.i], np.r_[x0], **kwargs)

    def plot_x_his(self, lw=None, label=None):
        if label == None:
            plt.plot(self.t_his, self.x_his, lw=lw)
        else:
            plt.plot(self.t_his, self.x_his, label=label, lw=lw)
        plt.xlabel("time $t$")
        plt.ylabel("the firing rate of the neuron $x$")


class MutualInhibit(Network):
    """ Circuit with mutual inhibition
    dx1(t)/dt = -x1(t) + f(wx2(t) +I)
    dx2(t)/dt = -x2(t) + f(wx1(t) +I)
    """

    w = -0.1
    i = 5

    def __init__(self, f, x0, **kwargs):
        set_all_args(self, kwargs)
        super().__init__(
            2, f, np.array([[0,self.w],[self.w,0]]),
            self.i * np.ones(2), x0, **kwargs)

    def plot_nullclines(self, ls='-', lw=None):
        # dx1(t)/dt = 0
        x2s = np.linspace(-50, 150, 1000)
        x1s = self.f(self.w * x2s + self.i)
        plt.plot(x1s, x2s,  label="$\dot{x}_1(t)=0$", ls=ls, lw=lw)
        # dx2(t)/dt = 0
        x1s = np.linspace(-50, 150, 1000)
        x2s = self.f(self.w * x1s + self.i)
        plt.plot(x1s, x2s, label="$\dot{x}_2(t)=0$", ls=ls, lw=0.8)
        plt.xlabel("the firing rate of the first neuron $x1$")
        plt.ylabel("the firing rate of the second neuron $x2$")

    def plot_x_his(self):
        x1_his, x2_his = np.transpose(self.x_his)
        line, = plt.plot(x1_his, x2_his)
        add_arrow(line)
