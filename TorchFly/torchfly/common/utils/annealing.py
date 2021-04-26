import numpy as np

# pct is from 0.0 to 1.0

def annealing_no(start, end, pct:float):
    "No annealing"
    return start

def annealing_linear(start, end, pct:float):
    "Linearly annealing"
    return start + pct * (end-start)

def annealing_exp(start, end, pct:float):
    "Exponential Annealing"
    return start * (end/start) ** pct

def annealing_cos(start, end, pct:float):
    "Cosine annealing"
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out