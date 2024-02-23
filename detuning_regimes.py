import sys

import numpy as np
import matplotlib.pyplot as plt
import math

pos=0.2

def global_detuning(t, dt, δ_start, δ_end, d_type='linear', position=pos):
    steps = int(t / dt)

    if d_type == 'linear':
        detuning = np.linspace(δ_start, δ_end, steps)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'rabi osc':
        detuning = np.zeros(steps)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'quench':
        detuning = linear_detuning_quench(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'linear flat':
        detuning = linear_detuning_flat(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'linear flat 2':
        detuning = linear_detuning_flat(δ_start, δ_end-20, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'linear negative':
        detuning = linear_quench_negative(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'flat zero':
        detuning = flat_zero(steps)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'flat positive':
        detuning = flat_positive(δ_end, steps)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'flat start':
        detuning = flat_start(δ_start, steps)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'short quench':
        detuning = quench_return(δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif d_type == 'driving quench':
        detuning = driving_quench(t, dt, δ_start, δ_end, steps, position=position)
        detuning = np.array([detuning])
        return detuning

    elif type(d_type) == int:
        detuning = quench_ramped(δ_start, δ_end, steps, d_type, position=position)
        detuning = np.array([detuning])
        return detuning

    elif type(d_type) == list:
        detuning = quench_short(δ_start, δ_end, steps, d_type[0], d_type[1], position=position, show=True)
        detuning = np.array([detuning])
        return detuning

    else:
        raise ValueError('Detuning type not here')
        sys.exit()


def linear_detuning_quench(δ_start, δ_end, steps, position=pos, show=False):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(0,0, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning


def linear_detuning_flat(δ_start, δ_end, steps, position=pos, show=False):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)
    quench = np.linspace(δ_end, δ_end, steps - linear_steps)
    detuning = np.hstack((sweep, quench))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning

def quench_return(δ_start, δ_end, steps, position=0.5, show=False):
    flat_steps = math.floor(steps * position)
    length=15

    flat_1 = np.linspace(δ_end, δ_end, flat_steps)
    quench = np.linspace(0, 0, length)
    flat_2 = np.linspace(δ_end, δ_end, steps-flat_steps-length)

    detuning = np.hstack((flat_1, quench, flat_2))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning

def quench_ramped(δ_start, δ_end, steps, quench_steps, position=0.5, show=False):
    flat_steps = math.floor(steps * position)

    flat_1 = np.linspace(δ_start, δ_end, flat_steps)
    quench = np.linspace(δ_end, 0, quench_steps)
    flat_2 = np.linspace(0, 0, steps - flat_steps - quench_steps)

    detuning = np.hstack((flat_1, quench, flat_2))


    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning

def quench_short(δ_start, δ_end, steps, quench_steps, duration, position=0.5, show=False):
    flat_steps = math.floor(steps * position)

    flat_1 = np.linspace(δ_start, δ_end, flat_steps)
    quench_down = np.linspace(δ_end, 0, quench_steps)
    flat_bottom = np.linspace(0, 0, duration)
    quench_up = np.linspace(0, δ_end, quench_steps)
    flat_2 = np.linspace(δ_end, δ_end, steps - flat_steps - 2*quench_steps - duration)

    detuning = np.hstack((flat_1, quench_down, flat_bottom, quench_up, flat_2))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, detuning)
        plt.show()

    return detuning



def flat_zero(steps):

    detuning = np.linspace(0, 0, steps)

    return detuning


def flat_positive(δ_end, steps):
    detuning = np.linspace(δ_end,  δ_end, steps)

    return detuning

def flat_start(δ_start, steps):
    detuning = np.linspace(δ_start,  δ_start, steps)

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
        indiv_d = global_detuning(t, dt, δ_start, δ_end, d_type=i)[0]
        detunning += [indiv_d]

    detunning = np.array(detunning)

    return detunning

def driving_quench(t, dt, δ_start, δ_end, steps, position=0.1, show=False, ax=None):
    linear_steps = math.floor(steps * position)

    sweep = np.linspace(δ_start, δ_end, linear_steps)

    t_quench = linear_steps*dt

    q_times = np.linspace(t_quench, t, steps-linear_steps)

    omega = 4*2 * np.pi
    mod_omega = 1.24*omega

    quench_detuning = 0.5*omega + 0.5*omega*np.cos(mod_omega*(q_times-t_quench)-np.pi)

    detuning = np.hstack((sweep, quench_detuning))

    if show:
        x = np.linspace(0, t, steps)
        plt.plot(x, detuning)
        plt.show()

    if ax is not None:
        x = np.linspace(0, t, steps)
        ax.plot(x, detuning)


    return detuning


if __name__ == "__main__":
    t=5
    dt=0.01
    steps= int(t/dt)
    start =200
    end= 200

    quench_short(start,end,steps,0,20, position=0, show=True)
