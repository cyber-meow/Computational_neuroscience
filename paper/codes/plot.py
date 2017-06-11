import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from FORCE import FORCE
from patterns import triangles

def plot_rs(force, neurons):
    fig, axs = plt.subplots(len(neurons), sharex=True, facecolor='white')
    for i,ax in enumerate(axs):
        force.plot_rs(neurons[i], ax)
        ax.set_frame_on(False)
        if i<len(axs)-1:
            ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    axs[-1].get_xaxis().tick_bottom()
    xmin, xmax = axs[-1].get_xaxis().get_view_interval()
    ymin, ymax = axs[-1].get_yaxis().get_view_interval()
    axs[-1].add_artist(
        Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    fig.subplots_adjust(hspace=0)
    
def compare_generate(force, T):
    start = len(force.t_his)
    force.simulate(T, update=False)
    fig, ax = plt.subplots()
    force.plot_zs(ax, start=start)
    force.plot_fs(ax, start=start)
    ax.set_xlabel('time $t$ (s)')
    ax.axes.get_yaxis().set_visible(False)
    