"""
Implement the FORCE learning described in the paper
<< Generating Coherent Patterns of Activity from Chaotic Neural Networks >>
for the simplest model (Figure 1A., learn only the readout weightà
"""

import numpy as np
import matplotlib.pyplot as plt

from utility import set_all_args


class Network(object):

    dt = 1e-4   # basic time step in simulation, in s
    tau = 1e-2  # characteristic time, in s
    pho = 0.1   # introduce a sparseness in the recurrent network
    g = 1.5     # strength of the recurrent connections
    gGz = 1     # strength of the connections from readout to network
    x_ma = 0.5  # the initial magnitude of x
    
    def __init__(self, N, **kwargs):
        self.N = N
        set_all_args(self, kwargs)
        self.init_parameters()

    def init_parameters(self):
        self.init_J()
        self.init_w()
        self.init_JGz()
        self.init_exp()

    def init_exp(self):
        self.init_t()
        self.init_x()
        self._z_his = [self.z]
    
    def init_t(self):
        self._t = 0
        self._t_his = [0]

    def init_x(self):
        self._x = self.x_ma * np.random.randn(self.N)
        self._r = np.tanh(self._x)
        self._x_his = [self.x.copy()]
        self._r_his = [self.r.copy()]

    def init_J(self):
        variance = 1/(self.pho*self.N)
        self.J = np.sqrt(variance) * np.random.randn(self.N, self.N)
        zeros = np.zeros_like(self.J)
        mask = np.random.choice(
            2, size=self.J.shape, p=[self.pho, 1-self.pho]).astype(bool)
        self.J[mask] = zeros[mask]

    def init_w(self):
        variance = 1/(self.pho*self.N)
        self.w = np.sqrt(variance) * np.random.randn(self.N)
        """
        self.w = np.zeros(self.N)
        """

    def init_JGz(self):
        self.JGz = np.random.uniform(-1, 1, self.N)

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
    def r(self):
        return self._r

    @property
    def r_his(self):
        return self._r_his

    @property
    def z(self):
        return np.dot(self.w, self.r)

    @property
    def z_his(self):
        return self._z_his

    @property
    def x_derivative(self):
        return (- self.x + self.g*np.dot(self.J, self.r) 
                + self.gGz*self.JGz*self.z) / self.tau

    def _step(self):
        self._t += self.dt
        self.t_his.append(self.t)
        self._x += self.x_derivative * self.dt
        self._r = np.tanh(self.x)
        self._x_his.append(self.x.copy())
        self._r_his.append(self.r.copy())
        

    def simulate(self, T):
        n = int(T/self.dt)
        for _ in range(n):
            self._step()            
            self._z_his.append(self.z)


class FORCE(Network):

    alpha = 1          # this is used to initialize P
    update_cycle = 10  # w and P are updated every update_cycle dt, must > 1

    def __init__(self, N, f, **kwargs):
        """
        f is a function that represents the activity pattern 
        we want to produce
        """
        self.f = f
        super().__init__(N, **kwargs)

    def init_parameters(self):
        super().init_parameters()
        self.init_P()

    def init_exp(self):
        super().init_exp()
        self._dws = []

    def init_P(self):
        self.P = np.eye(self.N)/self.alpha
        
    @property
    def dws(self):
        return self._dws

    @property
    def error(self):
        return self.z - self.f(self.t)

    def simulate(self, T, update=True):
        n = int(T/self.dt)
        for _ in range(n):
            self._step()
            if len(self.t_his)%self.update_cycle == 1:
                if update:
                    self._update()
                else:
                    self._dws.append(0)
            self._z_his.append(self.z)

    def _update(self):
        self._update_P()
        self._update_w()

    def _update_P(self):
        self.P -= (np.dot(np.dot(self.P, np.outer(self.r, self.r)), self.P)
                   / (1 + np.dot(np.dot(self.r, self.P), self.r)))

    def _update_w(self):
        delta_w = self.error * np.dot(self.P, self.r)
        self.w -= delta_w
        self._dws.append(np.linalg.norm(delta_w))

    def plot_rs(self, neuron, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        rs = [r[neuron] for r in self.r_his[start:end]]
        ax.plot(self.t_his[start:end], rs)
        
    def plot_zs(self, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.t_his[start:end], self.z_his[start:end], color='red')
    
    def plot_fs(self, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ts = self.t_his[start:end]
        fs = [self.f(t) for t in ts]
        ax.plot(ts, fs, color='green')
    
    def plot_dws(self, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ts = self.t_his[self.update_cycle::self.update_cycle]
        ax.plot(ts[start:end], self.dws[start:end])
        """
        ax.set_xlabel("time $t$ (s)")
        ax.set_ylabel("$\|\Delta \mathbb{w}\|$")
        """


