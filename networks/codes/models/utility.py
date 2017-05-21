
"""
Utility functions
"""

import numpy as np

def set_all_args(obj, argdict):
    for k in argdict.keys():
        if hasattr(obj, k):
            setattr(obj, k, argdict[k])
        else:
            print("Warning: parameter name {} not found!".format(k))

def add_arrow(line, direction='right', size=15, color=None):
    if color is None:
        color = line.get_color()
    xdata = line.get_xdata()
    ydata = line.get_ydata()
    x_position = (xdata.max() + xdata.min())/2
    y_position = (ydata.max() + ydata.min())/2
    start_ind = np.argmin(
        np.absolute(xdata-x_position) + np.absolute(ydata-y_position))
    end_ind = start_ind + 1 if direction == "right" else start_ind - 1
    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy = (xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle='->', color=color), size=size)
