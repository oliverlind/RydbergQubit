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
        detuning = np.array([detuning])
        return detuning

    elif type == 'rabi osc':
        detuning = np.zeros(steps)
        detuning = np.array([detuning])
        return detuning

    elif type == 'quench':
        detuning = linear_detuning_quench(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif type == 'linear flat':
        detuning = linear_detuning_flat(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif type == 'linear negative':
        detuning = linear_quench_negative(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    else:
        raise ValueError('Detuning type not here')
        sys.exit()


def linear_detuning_quench(δ_start, δ_end, steps, position=0.5, show=False):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(0, 0, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning


def linear_detuning_flat(δ_start, δ_end, steps, position=0.5, show=False):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(δ_end, δ_end, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning


def linear_quench_negative(δ_start, δ_end, steps, position=0.5, show=False):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(δ_start, δ_start, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning


def single_addressing(t, dt, δ_start, δ_end, single_addressing_list):
    detunning = []
    for i in single_addressing_list:
        indiv_d = global_detuning(t, dt, δ_start, δ_end, type=i)[0]
        detunning += [indiv_d]

    detunning = np.array(detunning)

    return detunning


if __name__ == "__main__":
    pass
