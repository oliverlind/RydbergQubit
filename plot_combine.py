import sys
import pandas as pd
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import numpy as np
import time
from scipy.linalg import expm

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import matplotlib.cm as cm

import data_analysis
import detuning_regimes
import rabi_regimes
from adiabatic_evolution import AdiabaticEvolution
from rydberg_hamiltonian_1d import RydbergHamiltonian1D
import data_analysis as da
import config.config as cf
from config.config import plotcolors
import ploting_tools
from matplotlib.colors import Normalize
from plot_single import PlotSingle



mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams["text.latex.preamble"]  = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['axes.linewidth'] = 1.0


class CombinedPlots(PlotSingle):

    def __init__(self, n, t, dt, δ_start, δ_end, detuning_type=None, single_addressing_list=None,
                 initial_state_list=None, rabi_regime='constant'):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list, rabi_regime=rabi_regime)


    def sweep_colourbar(self, type='rydberg', title=None):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12.5, 8), gridspec_kw={'width_ratios': [9, 1], 'height_ratios': [1.7, 1]})

        self.colour_bar(type=type, ax=axs[0, 0], cb_ax=axs[:, 1])
        self.rabi_and_detuning_shape(ax=axs[1, 0])

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')


        # Set x axis label
        plt.xlabel('Time ($\mu$s)')

        plt.show()

    def ordered_state_colourbars(self, a_list, initial_state_list, single_addressing_list, save_pdf=False):

        fig, axs = plt.subplots(len(a_list), 3,  figsize=(8, 4),  sharex='col', gridspec_kw={'width_ratios': [6, 3, 1], 'height_ratios': [1, 1]})


        for i in range(0, len(a_list)):

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, a=a_list[i], detuning_type=None, single_addressing_list=single_addressing_list,
                     initial_state_list=initial_state_list, rabi_regime='pulse start')

            if i == 0:
                cb_ax = axs[:, 2]
                cb = True
            else:
                cb_ax = None
                cb = False

            singleplot.colour_bar(ax=axs[i, 0], cb=cb, end_ax=axs[i, 1], cb_ax=cb_ax)


        axs[-1, 0].set_xlabel('Time ($\mu$s)')
        axs[-1, 1].set_xlabel(r'⟨$n_{i}$⟩')


        for ax in axs[:, 2]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)


        plt.show()

    def colorbar_state_fidelity(self,  states_to_test, type='rydberg', save_pdf=False):

        if type == 'rydberg':
            rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        else:
            sys.exit()

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 3.5),
                                gridspec_kw={'width_ratios': [9, 1], 'height_ratios': [1.7, 1]})

        self.colour_bar(data=rydberg_fidelity_data,type=type, ax=axs[0, 0], cb_ax=axs[:, 1])
        self.state_fidelity(states_to_test, q_states=states, ax=axs[1, 0])

        axs[1,0].set_ylabel(r'⟨$Z_{2}$|$\Psi$⟩')

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        # Set x axis label
        plt.xlabel('Time ($\mu$s)')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def quantum_mutual_informations(self):

        fig, axs = plt.subplots(self.n-1,1, sharex='col', figsize=(8, 4))

        states = self.time_evolve(states_list=True)

        for j in range(2, self.n+1):

            self.quantum_mutual_information(1, j, states=states, ax=axs[j-2])
            axs[j-2].set_ylim(0,1)

        plt.show()

    def eigenstate_fidelities(self):

        eigenvalues, eigenvectors, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                       eigenstate_fidelities=True)

        fig, axs = plt.subplots(1, 3, sharex=True, figsize=(8, 3.5),
                                gridspec_kw={'width_ratios': [5, 3, 0.7]})

        self.eigenenergies_lineplot_with_eigenstate_fidelities(eigenvalues=eigenvalues, eigenstate_probs=eigenstate_probs, ax=axs[0], cb=False)
        self.colour_bar(data=eigenstate_probs, ax=axs[1], type='eigen energies', cb_ax=axs[2])

        axs[1].set_xlim(2, self.t)


        axs[2].axis('off')

        # Set x axis label
        plt.xlabel('Time ($\mu$s)')

        plt.show()


if __name__ == "__main__":
    t = 5
    dt = 0.01
    n = 3
    δ_start = -30 * 2 * np.pi
    δ_end = 30 * 2 * np.pi

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['linear flat'] * 3
    three2 = ['quench']

    five = ['linear flat'] * 5
    five2 = ['quench'] * 5
    five3 = ['quench'] + ['linear flat'] * 4

    seven = ['linear flat'] * 7
    seven2 = ['quench'] * 7

    plotter = CombinedPlots(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=three2,
                         initial_state_list=[0, 0, 0], rabi_regime='constant'
                         )

    plotter.eigenstate_fidelities()

    plotter.quantum_mutual_informations()

    plotter.colorbar_state_fidelity([[1, 0, 1, 0, 1, 0, 1]], save_pdf=True)

    #plotter.eigenvalue_lineplot(show=True)

    #plotter.ordered_state_colourbars([5.48, 3.16], [0, 0, 0, 0, 0, 0, 0], seven, save_pdf=True)
