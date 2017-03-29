
"""
Implementation of drift-diffusion model
"""


import numpy as np
import matplotlib.pyplot as plt
from utility import set_all_args


class drift_diffusion(object):

    delta_t = 1e-4  # in second
    sigma = 0.5  # noise level

    def __init__(self, mA, mB, mu, **kwargs):
        self.mE = mA - mB
        self.mu = mu
        set_all_args(self, kwargs)
        self.simu_his = []
        self.reactime_hisA, self.reactime_hisB = [], []

    def simulate_once(self,t, store = True):
        N = int(t/self.delta_t)
        res = np.zeros(N + 1)
        sca = self.sigma * np.sqrt(self.delta_t)
        noises = np.random.randn(N)
        for k in np.arange(1, N + 1):
            res[k] = res[k-1] + self.mE * self.delta_t + noises[k-1] * sca
            if res[k] > self.mu:
                self.reactime_hisA.append(100 + k)
                break
            if res[k] < -self.mu:
                self.reactime_hisB.append(100 + k)
                break
        if store:
            self.simu_his.append(res[:k])

    def simulate(self, t, times = 1, store = True):
        for _ in range(times):
            self.simulate_once(t, store)

    def plot_curve(self, t):
        colors = plt.cm.jet(np.linspace(0,1,len(self.simu_his)))
        fig, ax = plt.subplots()
        ax.set_color_cycle(colors)
        for simu in self.simu_his:
            plt.plot(self.delta_t * np.arange(len(simu)), simu)
        plt.plot((0,t), (self.mu,self.mu), '--', color = "black")
        plt.plot((0,t), (-self.mu,-self.mu), '--', color = "black")
        plt.xlim(0,t)
        plt.ylim(-1.15 * self.mu, 1.15 * self.mu)
        plt.xlabel("time $t$ (s)")
        plt.ylabel("integration variable $x$")

    def plot_reactime(self):
        hisA = [self.delta_t * t for t in self.reactime_hisA]
        hisB = [self.delta_t * t for t in self.reactime_hisB]
        print(len(hisA), len(hisB))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))
        ax1.hist(hisA, 40, color = "lightskyblue")
        ax1.set_title("outcome A")
        ax2.hist(hisB, 40, color = "greenyellow")
        ax2.set_title("outcome B")
        for ax in [ax1, ax2]:
            ax.set_xlabel("reaction time $\\mathtt{RT}$ (s)")
            ax.margins(0.05)
            ax.set_ylim(ymin = 0)
            ax.title.set_position([.5,1.03])
