import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from FORCE import FORCE
from patterns import triangles


def add_bottom_line(fig, ax):
    ax.get_xaxis().tick_bottom()
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(
        Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    fig.subplots_adjust(hspace=0)

def super_graph(force, t1, t2):
    fig = plt.figure(facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[9,1])
    ax = plt.subplot(gs[0])
    rs = np.transpose(np.array(force.main_exp.r_his)[:,:10])
    offset = 0
    ts = force.main_exp.t_his
    for i in range(10):
        ax.plot(ts, rs[i]+offset, color='blue')
        if i < 9:
            offset += 1.2 * (np.max(rs[i]) - np.min(rs[i+1]))
        else:
            offset += 1.2 * (np.max(rs[i]) - np.min(force.main_exp.z_his))
    ax.plot(ts, np.array(force.main_exp.z_his)+offset, color="red")
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    axb = plt.subplot(gs[1])
    force.plot_dws(axb)
    axb.set_frame_on(False)
    axb.axes.get_yaxis().set_visible(False)
    axb.set_ylim(ymin=-0.1)
    axb.set_xlabel("time $t$ (s)")
    add_bottom_line(fig, axb)

def compare_generate(force, T, pic_name=None, sep=False):
    force.init_exp("new")
    force.simulate(T, exp_name="new", update=False)    
    exp = force.exps["new"]
    if sep:
        fig, ax = plt.subplots(figsize=(8,12))
        dec = np.ptp(exp.z_his)*1.2
        exp.plot_fs(lambda t: force.f(t)-dec, ax)
    else:
        fig, ax = plt.subplots()
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

def plot_rs_pca(exp, r_pca):
    rs = np.transpose(r_pca[:,:8])
    fig, ax = plt.subplots(facecolor='white')
    offset = 0
    for r in rs[::-1]:
        ax.plot(exp.from0(exp.t_his), r+offset, color='darkgoldenrod')
        offset += 1.3*np.ptp(r)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.margins(0.03, 0.05)    
    add_bottom_line(fig, ax)
    ax.set_xlabel("time $t$ (s)")

def plot_zs_pca(force, exp, pca, r_pca):
    fig, ax= plt.subplots(facecolor='white')
    exp.plot_zs(ax)
    pca8 = np.zeros_like(r_pca)
    pca8[:,:8] = r_pca[:,:8]
    rs_pca8 = pca.inverse_transform(pca8)
    zs_pca8 = np.dot(rs_pca8, force.w)
    oft = 1.25*np.ptp(exp.z_his)
    ax.plot(exp.from0(exp.t_his), zs_pca8-oft, color='darkgoldenrod')
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.set_ylim(ymin-0.3)
    add_bottom_line(fig, ax)
    ax.set_xlabel("time $t$ (s)")

def plot_eigenvalues(pca):
    fig, ax = plt.subplots()
    ax.plot(pca.explained_variance_)
    ax.set_yscale('log')
    ax.axhline(1, color='black')
    ax.set_ylabel("eigenvalue")
    ax.set_xlabel("principal components")
