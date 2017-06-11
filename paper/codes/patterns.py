from functools import partial
import numpy as np
import matplotlib.pyplot as plt

def pattern(func):
    """
    By using this decorator, you can supply each pattern function with
    some parameters and then the right function that only takes time t
    as argument is returned
    """
    def new_func(*args, **kwargs):
        return partial(func, *args, **kwargs)
    return new_func

@pattern
def triangles(T, mag, t):
    n = t // T
    y_delta = (t-T*n) / T * 2 * mag
    if n % 2 == 0:
        return mag-y_delta
    return -mag+y_delta

@pattern
def four_sin(T, t):
    s1 = np.sin(2*np.pi*t/T)
    s2 = np.sin(2*np.pi*t/T+np.pi/2)
    s3 = np.sin(2*np.pi*t/T+np.pi)
    s4 = np.sin(2*np.pi*t/T+np.pi*3/2)
    return s1+s2+s3+s4
    
def plot_patterns(T, p):
    ts = np.arange(0,T,1e-3)
    fs = [p(t) for t in ts]
    fig, ax = plt.subplots()
    plt.plot(ts, fs)
    plt.xlabel("time $t$ (s)")
    ax.axes.get_yaxis().set_visible(False)