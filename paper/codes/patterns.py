from functools import partial
import numpy as np

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
    
