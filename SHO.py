import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, pi

# Parameters
m = 1.0  # Mass of the particle
omega = 1.0  # Angular frequency of the oscillator
x0 = 0.0  # Initial position
p0 = 1.0  # Initial momentum
sigma_x = 0.5  # Initial width of the wave packet
N = 1024  # Number of grid points
dx = 0.1  # Space step
dt = 0.01  # Time step
T = 2 * np.pi / omega  # One period of the oscillator

x = dx * (np.arange(N) - N / 2)  # Position grid
V = 0.5 * m * omega**2 * x**2  # Harmonic potential

# Initial wave function: Gaussian wave packet
psi_x = np.exp(-(x - x0)**2 / (4 * sigma_x**2) + 1j * p0 * x / hbar)
psi_x /= np.sqrt(np.sum(np.abs(psi_x)**2) * dx)  # Normalize

# Pre-calculate the kinetic and potential propagators
K = np.exp(-1j * hbar * np.fft.fftfreq(N, dx)**2 / (2 * m) * dt)
V = np.exp(-1j * V * dt / hbar)

# Time evolution function using the split-step Fourier method
def evolve(psi_x, V, K, steps):
    for _ in range(steps):
        psi_x = np.fft.ifft(np.fft.fft(psi_x) * K)  # Kinetic propagation in momentum space
        psi_x *= V  # Potential propagation in real space
    return psi_x

# Simulate over one period
steps = 1000 #int(T / dt)
psi_x_final = evolve(psi_x, V, K, steps)

# Plotting the initial and final wave functions
plt.figure(figsize=(10, 6))
plt.plot(x, np.abs(psi_x)**2, label='Initial', color='blue')
plt.plot(x, np.abs(psi_x_final)**2, label='After one period', color='red')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.title('Gaussian Wave Packet in a Simple Harmonic Oscillator')
plt.legend()
plt.show()