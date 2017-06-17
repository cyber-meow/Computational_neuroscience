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
def triangles(T, amp, t):
    n = t // (T/2)
    y_delta = (t-T*n/2) / T * 4 * amp
    if n % 2 == 0:
        return amp-y_delta
    return -amp+y_delta

@pattern
def four_sins(T, amp, t):
    s1 = amp * np.sin(2*np.pi*t/T)
    s2 = amp / 2 * np.sin(4*np.pi*t/T)
    s3 = amp / 6 * np.sin(6*np.pi*t/T)
    s4 = amp / 3 * np.sin(8*np.pi*t/T)
    return s1+s2+s3+s4

@pattern
def sixteen_sins(T, amp, t):
    ret = 0
    amp_divs = [1, 2, 6, 3, 9, 12, 4, 3, 8, 16, 10, 5, 4, 6, 2, 8]
    for i in range(16):
        ret += amp / amp_divs[i] * np.sin(2*i*np.pi*t/T)
    return ret

@pattern
def noisy_four_sins(T, amp, sigma, t):
    return four_sins(T, amp)(t) + sigma*np.random.randn()

@pattern
def rectangles(T, amp, t):
    n = t // (T/2)
    if n%2 == 0:
        return -amp
    return amp

def plot_patterns(T, p):
    ts = np.arange(0,T,1e-3)
    fs = [p(t) for t in ts]
    fig, ax = plt.subplots()
    plt.plot(ts, fs)
    plt.xlabel("time $t$ (s)")
    ax.axes.get_yaxis().set_visible(False)
