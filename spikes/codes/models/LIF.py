
"""
Implementation of the leaky integrate-and-fire model
Equation: C*dV/dt = gL(EL-V(t)) + I(t)
Threshold: Vth
"""

import numpy as np
import matplotlib.pyplot as plt
from utility import set_all_args



class LIF(object):

    # biophysical parameters of the neuron
    _C = 1  # nF
    _gL = 0.1  # \muS
    _R = 1/_gL  # MOm
    _tau_m = _R * _C  * 1e-3  # s
    EL = -70  # mV
    Vth = -63  # mV

    # simulation parameters
    delta_t = 1  # ms

    # I a function of t (in s)
    def __init__(self, I, spiking=True, **kwargs):
        self.I = I
        self.spiking = spiking
        set_all_args(self, kwargs)
        self.current_V = self.EL
        self._just_spike = False
        self._spike_moments = []
        self._V_his, self._t_his = [self.current_V], [0] # t_his in s

    @property
    def I(self):
        return self._I

    @I.setter
    def I(self, I):
        assert(callable(I))
        self._I = I

    @property
    def gL(self):
        return self._gL

    @gL.setter
    def gL(self, gL):
        self._gL = gL
        self._R = 1 / gL
        self._tau_m = self.C / gL * 1e-3

    @property 
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R
        self._gL = 1 / R
        self._tau_m = R * self.C * 1e-3

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        self._C = C
        self._tau_m = self.R * C * 1e-3

    @property
    def spike_moments(self):
        return self._spike_moments

    def _computeV_step(self):
        self._just_spike = False
        # we can verify the homogeneity of the eqution
        self.current_V += ((
            self.gL * (self.EL - self.current_V) + self.I(self._t_his[-1])) 
            / self.C * self.delta_t)
        self._V_his.append(self.current_V)
        self._t_his.append(self._t_his[-1] + self.delta_t * 1e-3)
        if self.spiking and self.current_V >= self.Vth:
            self._just_spike = True
            self._spike_moments.append(self._t_his[-1])
            self.current_V = self.EL
            self._V_his.append(self.current_V)
            self._t_his.append(self._t_his[-1])

    # t in s
    def computeV(self, t):
        N = int(t * 1e3 / self.delta_t)
        for _ in range(N):
            self._computeV_step()

    def plot_V(self, ax, label=None, xylabel=True, unit=True):
        if label is None:
            ax.plot(self._t_his, self._V_his)
        else:
            ax.plot(self._t_his, self._V_his, label=label)
        ax.margins(None, 0.2)
        ax.set_xlim(self._t_his[0], self._t_his[-1])
        ax.set_ylim(ymin=self.EL)
        if self.spiking:
            for t in self._spike_moments:
                ax.axvline(t, 0, 0.9)
            ax.axhline(self.Vth, linestyle = '--')
            ax.set_ylim(self.EL, self.Vth*2.5 - self.EL*1.5)
        if xylabel:
            ax.set_xlabel("time $t$ (s)")
            if unit:
                ax.set_ylabel("membrane potential $V$ (mV)")
            else:
                ax.set_ylabel("membrane potential $V$")


class LIFICst(LIF):
    """When the input current is constant"""

    # self.I is the function while self._I is the constant value
    @property
    def I(self):
        def I_cst(t): return self._I
        return I_cst

    @I.setter
    def I(self, I):
        assert isinstance(I, (int, float))
        self._I = I
        self._Vinfty = self.EL + self._I * self.R  # mV
    
    @property
    def firing_rate(self):
        if self._Vinfty <= self.Vth:
            return 0
        return (1 / (self._tau_m
                * np.log((self.EL-self._Vinfty)/(self.Vth-self._Vinfty))))


class LIFNumeric(LIFICst):

    # I must be a constant
    def __init__(self, I, **kwargs):
        # The analytic formula can of course be used when the spiking
        # mechanism is also considered, I just don't implement it here
        super().__init__(I, False, **kwargs)
        assert(not self.spiking)

    def computeV(self, t):
        self._t_his = np.arange(0, t, self.delta_t*1e-3)
        self._V_his = (
            self._Vinfty
            + (self.EL-self._Vinfty) * np.exp(-self._t_his/self._tau_m))


class LIFRefractory(LIF):
    """Take into account the refractory period"""

    delta = 3  # ms

    def __init__(self, I, spiking=True, **kwargs):
        super().__init__(I, spiking, **kwargs)
        self._refrac_time = -1
    
    def _computeV_step(self):
        if self._refrac_time < 0:
            super()._computeV_step()
            if self._just_spike:
                self._refrac_time = 0
        elif self._refrac_time >= self.delta:
            self._refrac_time = -1
            super()._computeV_step()
        else:
            self._just_spike = False
            self._refrac_time += self.delta_t
            self._V_his.append(self.current_V)
            self._t_his.append(self._t_his[-1] + self.delta_t*1e-3)


class LIFICstRefractory(LIFRefractory, LIFICst):

    @property
    def firing_rate(self):
        f = super().firing_rate
        if f == 0:
            return 0
        return 1/(self.delta*1e-3 + 1/f)


class LIFNoise(LIF):
    "Add gaussian white noise"

    sigma = 1  # mV * ms^(1/2)

    def _computeV_step(self):
        self._just_spike = False
        self.current_V += ((
            self.gL * (self.EL - self.current_V) + self.I(self._t_his[-1])) 
            / self.C * self.delta_t 
            + self.sigma * np.sqrt(self.delta_t) * np.random.randn())
        self._V_his.append(self.current_V)
        self._t_his.append(self._t_his[-1] + self.delta_t * 1e-3)
        if self.spiking and self.current_V >= self.Vth:
            self._just_spike = True
            self._spike_moments.append(self._t_his[-1])
            self.current_V = self.EL
            self._V_his.append(self.current_V)
            self._t_his.append(self._t_his[-1])


class LIFRefractoryNoise(LIFRefractory, LIFNoise):
    pass
