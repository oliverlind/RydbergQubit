import numpy as np
import matplotlib.pyplot as plt
import math

pos = 0.99


def global_rabi(t, dt, steps, type='constant', position=pos):
    if type == 'constant':
        rabi_regime = np.linspace(1, 1, steps)

        return rabi_regime

    if type == 'delayed pulse':
        rabi_regime = delayed_pulse(steps, position=position)

        return rabi_regime

    if type == 'pulse start':
        rabi_regime = pulse_start(steps, position=position)

        return rabi_regime





def delayed_pulse(steps, position=pos, show=False):
    zero_steps = math.floor(steps * position)

    zero = np.linspace(0, 0, zero_steps)
    pulse = np.linspace(1, 1, steps - zero_steps)
    rabi_regime = np.hstack((zero, pulse))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, rabi_regime)
        plt.show()

    return rabi_regime

def pulse_start(steps, position=pos, show=False):

    pulse_steps = math.floor(steps * position)

    pulse = np.linspace(1, 1, pulse_steps)
    zero = np.linspace(0, 0, steps - pulse_steps)

    rabi_regime = np.hstack((pulse, zero))

    if show:
        x = np.arange(0, steps)
        plt.plot(x, rabi_regime)
        plt.show()

    return rabi_regime



if __name__ == "__main__":
    pulse_start(500, position=pos, show=True)
