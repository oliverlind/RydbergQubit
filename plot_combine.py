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


    def sweep_colourbar(self, types, type='rydberg', title=None, save_pdf=False):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 2.2), gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [1.7, 1]})

        self.colour_bar(type=type, ax=axs[0, 0], cb_ax=axs[:, 1])
        self.detuning_shape(types, ax=axs[1, 0])

        #axs[1, 0].legend(loc='upper right')
        axs[1, 0].set_xlabel('Time ($\mu$s)')

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

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

        axs[0, 0].set_ylabel(r'$Z_{2}$ ($a$ = 5.48$\mu$m)')
        axs[1, 0].set_ylabel(r'$Z_{3}$ ($a$ = 3.16$\mu$m)')


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

    def quantum_mutual_informations(self, save_pdf=False):

        fig, axs = plt.subplots(self.n-1,1, sharex='col', figsize=(4, 4))

        states = self.time_evolve(states_list=True)

        for j in range(2, self.n+1):

            self.quantum_mutual_information(1, j, states=states, ax=axs[self.n-j])
            axs[self.n-j].set_ylim(0, 1.1)

            if j != 2:
                axs[self.n - j].set_ylabel(f'I(1,{j})', rotation=0, labelpad=10, ha='right')
                # axs[self.n-j].tick_params(axis='y', which='both', left=False, right=False)
                # axs[self.n-j].set_yticklabels(['   '])

            else:
                axs[self.n - j].set_ylabel(f'I(1,{j})', rotation=0, labelpad=10, ha='right')
                axs[self.n - j].set_xlabel(r'Time after quench ($\mu$s)')


        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def rydberg_correlation_cbs(self, i=None, save_pdf=False):

        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        # Get correlation data
        g_list = [[] for _ in range(self.steps)]

        for j in range(0, self.steps):
            g = self.rydberg_rydberg_density_corr_function(states[j], i=i)
            g_list[j] = g

        plotting_g_data = np.array(g_list)
        print(plotting_g_data)
        plotting_g_data = plotting_g_data.T

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 2.2),
                                gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [1, 1]})

        self.colour_bar(type='rydberg', data=rydberg_fidelity_data, ax=axs[0, 0], cb_ax=axs[0, 1])
        self.colour_bar(type='correlation', data=plotting_g_data, ax=axs[1, 0], cb_ax=axs[1, 1])

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def eigenstate_fidelities(self, save_pdf=False):

        eigenvalues, eigenvectors, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                       eigenstate_fidelities=True)

        fig, axs = plt.subplots(1, 3, figsize=(8, 2.7),
                                gridspec_kw={'width_ratios': [5, 1.5, 0.2]})

        self.eigenenergies_lineplot_with_eigenstate_fidelities(eigenvalues=eigenvalues, eigenstate_probs=eigenstate_probs, ax=axs[0], cb=True, cb_ax=axs[2])
        #self.colour_bar(data=eigenstate_probs, ax=axs[1], type='eigen energies', cb_ax=axs[2])
        #ploting_tools.end_eigenenergies_barchart(12, eigenstate_probs, axs[1])

        self.eigenenergies_lineplot_with_eigenstate_fidelities(eigenvalues=eigenvalues,
                                                               eigenstate_probs=eigenstate_probs, ax=axs[1], cb=False)

        axs[0].set_ylim(-400, 600)
        axs[1].set_ylim(-30, 420)
        axs[1].set_xlim(4.1, 5)

        axs[0].tick_params(axis='y', which='both',right=False)
        axs[1].tick_params(axis='y', which='both', right=False)
        axs[1].tick_params(axis='x', which='both', top=False, bottom=False)
        axs[1].set_xticklabels([''])
        axs[1].set_xlabel(r'⟨$\Psi_{\lambda}$|$\Psi(t>t_{quench})$⟩', labelpad=12)
        axs[0].set_ylabel(r'$E_{\lambda}$ (MHz)')


        axs[2].axis('off')

        plt.subplots_adjust(wspace=0.3)

        # Set x axis label
        axs[0].set_xlabel('Time ($\mu$s)')
        #axs[0].set_xlabel(r'⟨$\Psi_{\lambda}$|$\Psi$⟩')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def two_atom_eigenstates(self, save_pdf=False):
        eigenvalues, eigenvectors, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                       eigenstate_fidelities=True)

        fig, axs = plt.subplots(2, 3, figsize=(8, 4.5), sharex=True, sharey=True,
                                gridspec_kw={'width_ratios': [1,1,0.2]})

        labels = [r'|00⟩ Fidelity', r'|$\Psi^{-}$⟩ Fidelity', r'|$\Psi^{+}$⟩ Fidelity', r'|rr⟩ Fidelity']
        states_to_test = [[0,0], 'psi minus', 'psi plus', [1,1]]

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[0], eigenvalues=eigenvalues, eigenvectors=eigenvectors, detuning=True, ax=axs[0,0], cb_label=None)
        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[1], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[0,1],
                                                          cb_label=None)

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[2], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[1,0],
                                                          cb_label=None, reverse=False)

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[3], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[1,1],
                                                          cb_label='State Fidelity', cb_ax=axs[:,2])

        axs[0, 1].set_ylabel('')
        axs[1, 1].set_ylabel('')

        axs[1, 0].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')
        axs[1, 1].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')

        plt.ylim(-60, 55)

        for ax in axs[:, 2]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)


        plt.show()

    def cb_entanglement_entropy(self, atom_i=None, save_pdf=False):

        if atom_i is not None:
            rydberg_fidelity_data, states, ee = self.time_evolve(rydberg_fidelity=True, entanglement_entropy=True, states_list=True)
        else:
            rydberg_fidelity_data, ee = self.time_evolve(rydberg_fidelity=True, entanglement_entropy=True)
            states = None

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 2.2))

        self.colour_bar(type='rydberg', data=rydberg_fidelity_data, ax=axs[0], cb=False)
        self.plot_half_sys_entanglement_entropy(ax=axs[1], atom_i=atom_i, entanglement_entropy=ee, states=states)

        plt.subplots_adjust(hspace=0)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def correlation_averages(self, qsteps_list,  initial_state, single_addressing_list, save_pdf=False):

        averages_list = [[] for _ in range(len(qsteps_list))]
        averages_dict = {}

        for j, q_step in enumerate(qsteps_list):
            single_addressing_list[0] = int(q_step)

            print(single_addressing_list)

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                   single_addressing_list=single_addressing_list,
                                   initial_state_list=initial_state, rabi_regime='constant')

            states = singleplot.time_evolve(states_list=True)

            g_r_list = [[] for _ in range(self.steps)]

            for i in range(0, self.steps):
                g_r = self.rydberg_rydberg_density_corr_function(states[i], i=1)
                g_r_list[i] = g_r

            data = np.array(g_r_list)
            data = data.T

            averages = []

            for i in range(0, self.n-1):
                av = np.abs(np.average(data[i][q_step:q_step+398]))
                averages += [av]

            averages_dict[q_step] = averages
            averages_list[j] = averages

        print(averages_dict)

        averages_list = np.array(averages_list)
        averages_list = averages_list.T

        qsteps_list = np.array(qsteps_list)

        print(averages_list)

        #speeds = np.ones(len(qsteps_list)) / np.array(qsteps_list) * self.dt

        #print(speeds)

        for k in range(0, self.n-1):

            plt.scatter(qsteps_list*self.dt, averages_list[k], label=f'|G(1, {k+2})|')
            plt.plot(qsteps_list*self.dt, averages_list[k])

        plt.xlabel(r'$\Delta$$t_{quench}$ ($\mu$s)')
        plt.ylabel('|⟨G(1, i)⟩|')

        plt.legend()

        plt.show()


if __name__ == "__main__":
    t = 6
    dt = 0.01
    n = 7
    δ_start = 30 * 2 * np.pi
    δ_end = 30 * 2 * np.pi

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['linear flat'] * 3
    three2 = ['quench'] + ['linear flat'] * 2
    three3 = ['quench'] * 3

    five = ['linear flat'] * 5
    five2 = ['quench'] * 5
    five3 = ['quench'] + ['linear flat'] * 4

    seven = ['linear flat'] * 7
    seven2 = ['quench'] * 7
    seven3 = [175] +['linear flat'] * 6
    seven4 = ['short quench'] + ['linear flat'] * 6

    plotter = CombinedPlots(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=seven3,
                         initial_state_list=[1, 0, 1, 0, 1, 0, 1], rabi_regime='constant'
                         )

    plotter.correlation_averages(np.arange(1,200, 20), [1, 0, 1, 0, 1, 0, 1], seven4)

    #plotter.cb_entanglement_entropy(atom_i=1)

    plotter.rydberg_correlation_cbs(i=1)

    #plotter.colorbar_state_fidelity([[1, 0, 1, 0, 1, 0, 1]], save_pdf=True)

    # plotter.two_atom_eigenstates(save_pdf=True)
    #
    # plotter.sweep_colourbar(three3, save_pdf=True)
    #
    #plotter.eigenstate_fidelities(save_pdf=True)
    #
    plotter.quantum_mutual_informations(save_pdf=True)
    #

    #
    # #plotter.eigenvalue_lineplot(show=True)

    #plotter.ordered_state_colourbars([5.48, 3.16], [0, 0, 0, 0, 0, 0, 0], seven, save_pdf=True)
