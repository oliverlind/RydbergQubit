import sys

import numpy as np
import matplotlib.pyplot as plt
import math


# class DetuningRegime:
#     def __init__(self, n, t, dt, δ_start, δ_end):
#         self.n = n
#         self.t = t
#         self.dt = dt
#         self.steps = int(t/dt)
#         self.δ_start = δ_start
#         self.δ_end = δ_end
#         self.detunning = np.linspace(self.δ_start, self.δ_end, self.steps)
#
#     def detuning(self, type, steps):
#
#         if type == 'rabi osc':
#             self.detunning = np.zeros(self.steps)


def global_detuning(t, dt, δ_start, δ_end, type=None, position=0.5):
    steps = int(t / dt)

    if type is None:
        detuning = np.linspace(δ_start, δ_end, steps)
        return detuning

    elif type == 'rabi osc':
        detuning = np.zeros(steps)
        return detuning

    elif type == 'quench':
        detuning = linear_detuning_quench(δ_start, δ_end, steps, position=position)
        return detuning

    else:
        raise ValueError('Detuning type not here')
        sys.exit()


def linear_detuning_quench(δ_start, δ_end, steps, position=0.5, show=False):
    linear_steps = math.floor(steps*position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(0, 0, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning
