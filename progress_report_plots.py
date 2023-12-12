from plot_single import PlotSingle
from rydberg_hamiltonian_1d import RydbergHamiltonian1D
import numpy as np


''' Blockade Demonstration Plot'''

a_list = np.linspace(0, 10, 50)

for a in a_list:

    h_m = RydbergHamiltonian1D(2, a=a).hamiltonian_matrix([0])

