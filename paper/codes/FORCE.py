"""
Implement the FORCE learning described in the paper
<< Generating Coherent Patterns of Activity from Chaotic Neural Networks >>
for the simplest model (Figure 1A., learn only the readout weightà
"""

import numpy as np
import matplotlib.pyplot as plt

from utility import set_all_args


class exp(object):
    
    def __init__(self, N, w, x_ma, t_init=None, x_init=None):
        self.N = N
        self.x_ma = x_ma
        self.init_t(t_init)
        self.init_x(x_init)
        self._z_his = [np.dot(w, self.r)]
    
    def init_t(self, t_init=None):
        if t_init is None:
            self._t = 0
        else:
            self._t = t_init
        self._t_his = [self._t]

    def init_x(self, x_init=None):
        if x_init is None:
            self._x = self.x_ma * np.random.randn(self.N)
        else:
            self._x = x_init
        self._r = np.tanh(self._x)
        self._x_his = [self.x.copy()]
        self._r_his = [self.r.copy()]
                    
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
        return self._z_his[-1]        
        
    @property
    def z_his(self):
        return self._z_his
        
    @staticmethod
    def from0(t_his):
        return [t-t_his[0] for t in t_his]

    def plot_fs(self, f, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ts = self.t_his[start:end]
        fs = [f(t) for t in ts]
        ax.plot(self.from0(ts), fs, color='green')
    
    def plot_rs(self, neuron, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        rs = [r[neuron] for r in self.r_his[start:end]]
        ax.plot(self.from0(self.t_his[start:end]), rs)
        
    def plot_zs(self, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.from0((self.t_his[start:end])), 
                self.z_his[start:end], color='red')
        
        
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
        self.exps = dict()
        self.init_exp("main", cont=False)

    def init_parameters(self):
        self.init_J()
        self.init_w()
        self.init_JGz()

    def init_exp(self, exp_name, cont=True):
        if cont:
            t = self.main_exp.t
            x = self.main_exp.x.copy()
            exper = exp(self.N, self.w, self.x_ma, t, x)
        else:
            exper = exp(self.N, self.w, self.x_ma)
        self.exps[exp_name] = exper
        
    @property
    def main_exp(self):
        return self.exps["main"]

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

    def x_derivative(self, exp_name='main'):
        exp = self.exps[exp_name]
        return (- exp.x + self.g*np.dot(self.J, exp.r) 
                + self.gGz*self.JGz*exp.z) / self.tau

    def _step(self, exp_name):
        exp = self.exps[exp_name]
        exp._t += self.dt
        exp.t_his.append(exp.t)
        exp._x += self.x_derivative(exp_name) * self.dt
        exp._r = np.tanh(exp.x)
        exp._x_his.append(exp.x.copy())
        exp._r_his.append(exp.r.copy())
        

    def simulate(self, T, exp_name='main'):
        exp = self.exps[exp_name]
        n = int(T/self.dt)
        for _ in range(n):
            self._step(exp_name)
            exp._z_his.append(np.dot(self.w, exp.r))


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
        self._dws = []

    def init_P(self):
        self.P = np.eye(self.N)/self.alpha
        
    @property
    def dws(self):
        return self._dws

    @property
    def error(self):
        return self.main_exp.z - self.f(self.main_exp.t)

    def simulate(self, T, exp_name='main', update=True):
        if update:
            assert exp_name == "main"            
        n = int(T/self.dt)
        exp = self.exps[exp_name] 
        for _ in range(n):
            self._step(exp_name)
            if exp_name == "main" and len(exp.t_his)%self.update_cycle == 1:
                if update: 
                    self._update()
                else:
                    self._dws.append(0)
            exp._z_his.append(np.dot(self.w, exp.r))

    def _update(self):
        self._update_P()
        self._update_w()

    def _update_P(self):
        exp = self.main_exp
        self.P -= (np.dot(np.dot(self.P, np.outer(exp.r, exp.r)), self.P)
                   / (1 + np.dot(np.dot(exp.r, self.P), exp.r)))

    def _update_w(self):
        exp = self.main_exp
        delta_w = self.error * np.dot(self.P, exp.r)
        self.w -= delta_w
        self._dws.append(np.linalg.norm(delta_w))

    def plot_dws(self, ax=None, start=0, end=None):
        if ax is None:
            fig, ax = plt.subplots()
        ts = self.main_exp.t_his[self.update_cycle::self.update_cycle]
        ax.plot(ts[start:end], self.dws[start:end])
        """
        ax.set_xlabel("time $t$ (s)")
        ax.set_ylabel("$\|\Delta \mathbb{w}\|$")
        """
