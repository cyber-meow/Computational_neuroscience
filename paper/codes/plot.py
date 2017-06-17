import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FORCE import FORCE
from patterns import triangles


def add_bottom_line(fig, ax):
    ax.get_xaxis().tick_bottom()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(
        Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    fig.subplots_adjust(hspace=0)

def hide_ax(ax):
    ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def plot_rs(force, neurons):
    fig, axs = plt.subplots(len(neurons), sharex=True, facecolor='white')
    for i,ax in enumerate(axs):
        force.plot_rs(neurons[i], ax)
        ax.set_frame_on(False)
        if i<len(axs)-1:
            ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    add_bottom_line(fig, axs[-1])
    
def compare_generate(force, T, pic_name=None):
    force.init_exp("new")
    force.simulate(T, exp_name="new", update=False)
    fig, ax = plt.subplots()
    exp = force.exps["new"]
    exp.plot_fs(force.f, ax)
    exp.plot_zs(ax)
    ax.set_xlabel('time $t$ (s)')
    ax.axes.get_yaxis().set_visible(False)
    plt.margins(0.02, None)
    if pic_name is not None:
        plt.savefig("../figures/{}".format(pic_name))


def exp_PCA(exp):
    pca = PCA(n_components=100)
    r_pca = pca.fit_transform(exp.r_his)
    return pca, r_pca

def plot_rs_pca(r_pca):
    rs = np.transpose(r_pca[:,:8])
    fig, axs = plt.subplots(8, sharex=True, facecolor='white')
    for i,ax in enumerate(axs):
        ax.plot(rs[i], color='darkgoldenrod')
        hide_ax(ax)

def plot_zs_pca(force, exp, pca, r_pca):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, facecolor='white')
    exp.plot_zs(ax1)
    hide_ax(ax1)
    pca8 = np.zeros_like(r_pca)
    pca8[:,:8] = r_pca[:,:8]
    rs_pca8 = pca.inverse_transform(pca8)
    zs_pca8 = np.dot(rs_pca8, force.w)
    ax2.plot(exp.t_his, zs_pca8, color='darkgoldenrod')
    hide_ax(ax2)

def plot_eigenvalues(pca):
    plt.plot(pca.explained_variance_)
