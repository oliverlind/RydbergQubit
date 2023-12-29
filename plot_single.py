import sys
import pandas as pd
from scipy.fft import fft, fftfreq
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

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams["text.latex.preamble"] = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
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


class PlotSingle(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, a=5.48, detuning_type=None, single_addressing_list=None,
                 initial_state_list=None, rabi_regime='constant'):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list,
                         rabi_regime=rabi_regime, a=a)

    def colour_bar(self, type='rydberg', data=None, title=None, show=False, ax=None, cb=True, cb_ax=None, end_ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if data is None:
            if type == 'rydberg':
                data = self.time_evolve(rydberg_fidelity=True)
                print(np.shape(data))

        ploting_tools.set_up_color_bar(self.n, data, self.times, ax=ax, type=type, colorbar=cb, cb_ax=cb_ax)


        if end_ax is not None:
            ploting_tools.end_colorbar_barchart(self.n, data, ax=end_ax)



        if show:
            plt.show()

    def rabi_and_detuning_shape(self, ax=None, show=False):

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(11, 6))
        else:
            ax1 = ax

        detuning_1 = detuning_regimes.linear_detuning_flat(self.δ_start, self.δ_end, self.steps)
        rabi = rabi_regimes.pulse_start(self.steps)

        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel(r'$\Delta$ (2$\pi$ x Mhz)', color='r')
        ax1.plot(self.times, detuning_1, color='r', label='Atom 1, 4')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.set_ylim(min(self.δ_start, self.δ_end) - 20, max(self.δ_start, self.δ_end) + 60)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel(r'$\Omega$ \ $\Omega_{0}$', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.times, rabi, color=color)
        ax2.fill_between(self.times, rabi, 0, color='blue', alpha=.1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.1)

        if show:
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

    def state_fidelity(self, states_to_test, q_states=None, ax=None, sum_probs=False, colors_num=0):

        if ax is None:
            q_states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
            ax.set_xlabel(r'Time ($\mu$s)')
        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        ploting_tools.plot_state_fidelities(q_states, states_to_test, self.times, self.steps, ax=ax, sum_probs=sum_probs
                                            , colors_num=colors_num)

    def state_fidelity_ft(self, states_to_test, interval=None, colors_num=0, title=None, ax=None, q_states=None):

        if q_states is None:
            q_states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.title(f'Moving First Atom 10a per μs (Δ = {self.δ_end})', fontsize=18)
            plt.xlim(0, self.t)
            ax = plt.gca()
            ax.set_xlabel('Time (μs)')

        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        ax.set_ylabel('Probability')
        plt.ylim(0, 1)

        num_of_test_states = len(states_to_test)
        state_fidelities = [[] for _ in range(num_of_test_states)]

        if interval is None:
            interval = [0, self.t]

        for i in range(0, num_of_test_states):

            state_to_test = states_to_test[i]
            label = ploting_tools.state_label(state_to_test)

            v_state_to_test = self.initial_state(state_to_test)

            for j in range(0, self.steps):
                state_fidelity = data_analysis.state_prob(v_state_to_test, q_states[j])
                state_fidelities[i] += [state_fidelity]

            cn = colors_num + i

            ax.plot(self.times, state_fidelities[i], label=f'{label}', color=plotcolors[cn])

            fourier_transform = np.abs(fft(state_fidelities[i][200:]))  # Taken absolute value to get the magnitude of
            # each phase component

    def eigenenergies_lineplot(self, eigenvalues=None, detuning=False, ax=None):

        if eigenvalues is None:
            eigenvalues, eigenvectors = self.time_evolve(eigen_list=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            ax.set_xlabel(r'Time ($\mu$s)')

        if detuning:
            ploting_tools.plot_eigenenergies(self.n, self.detunning[0], eigenvalues, ax, range(0, self.dimension))

        else:
            ploting_tools.plot_eigenenergies(self.n, self.times, eigenvalues, ax, range(0, self.dimension))

    def eigenenergies_lineplot_with_eigenstate_fidelities(self, eigenvalues=None, eigenstate_probs=None, expectation_energies=None, eigenstate_fidelities=None, ax=None):

        if ax is None:
            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                                                 eigenstate_fidelities=True,
                                                                                                 expec_energy=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            ax.set_xlabel(r'Time ($\mu$s)')

        ploting_tools.plot_eigenenergies_fidelities_line(self.n, self.times, eigenvalues, eigenstate_probs,
                                                         expectation_energies, ax, range(0, self.dimension))

    def eigenenergies_lineplot_with_state_fidelities(self, state_to_test, eigenvalues=None, eigenvectors=None, detuning=False, ax=None,
                                                     cb_label=r'|$\Psi^{+}$⟩ Fidelity', save_pdf=False, show=False):

        if ax is None:

            eigenvalues, eigenvectors = self.time_evolve(eigen_list=True)
            fig, ax = plt.subplots(figsize=(4, 3))

            if detuning:
                ax.set_xlabel(r'$\Delta$ (MHz)')
            else:
                ax.set_xlabel(r'Time ($\mu$s)')

        if detuning:
            ploting_tools.plot_eigenenergies_state_fidelities_line(self.n, self.detunning[0] / (2 * np.pi), eigenvalues,
                                                                   eigenvectors,
                                                                   state_to_test, ax, range(0, self.dimension),
                                                                   cb_label=cb_label)

        else:
            ploting_tools.plot_eigenenergies_state_fidelities_line(self.n, self.times, eigenvalues, eigenvectors,
                                                                   state_to_test, ax, range(0, self.dimension),
                                                                   cb_label=cb_label)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.tight_layout()
            plt.show()

    def eigenvalues_distance(self, save_pdf=False, show=False):
        fig, ax = plt.subplots(figsize=(4, 3))

        ax.set_xlabel(r'$a$ ($\mu$m)', fontsize=12)
        ax.set_ylabel(r'$E_{\lambda}$ (MHz)')

        a_list = np.linspace(5.5, 15, 500)

        eigenvalues_list = []

        for a in a_list:
            h_m = RydbergHamiltonian1D(2, a=a).hamiltonian_matrix([0])

            eigenvalues, eigenvector = np.linalg.eigh(h_m)

            eigenvalues_list += [eigenvalues]

        labels = ['|00⟩', r'|$\Psi^{-}$⟩', r'|$\Psi^{+}$⟩', '|rr⟩']

        eigenvalues_list = np.array(eigenvalues_list)

        for i in reversed(range(0, 4)):
            ax.plot(a_list, eigenvalues_list[:, i]/self.two_pi, label=labels[i], linewidth=2, zorder=1)

        ax.set_xlim(min(a_list), max(a_list))

        ax.spines['right'].set_position(('data', max(a_list)))


        pale_blue = (0.7, 0.8, 1.0)

        plt.axhline(y=self.Rabi/self.two_pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhline(y=-self.Rabi/self.two_pi, color='black', linestyle='--', linewidth=1, alpha=0.5)
        plt.axhspan(ymin=-self.Rabi/self.two_pi, ymax=self.Rabi/self.two_pi, color='grey', alpha=0.2)

        plt.axvline(x=self.r_b, color=pale_blue, linestyle='--', linewidth=2, alpha=0.8)

        print(self.Rabi)

        plt.text(self.r_b + 0.35, 125/self.two_pi, r'$R_{B}$', ha='center', va='center', rotation=-90, fontsize=12)

        # plt.text(5.25, 315, r'$|rr⟩$', ha='center', va='center', fontsize=18)  # Displaying 1/2

        plt.axvspan(xmin=5.5, xmax=self.r_b, color=pale_blue, alpha=0.2)

        plt.subplots_adjust(right=0.8)
        plt.legend(loc='upper right', fontsize=12)

        # plt.arrow(13, -25+23.7, 0, -22.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)

        # Create an inset axes in the top right corner
        #axins = inset_axes(plt.gca(), width="45%", height="40%", loc='upper right')

        # Specify the region to zoom in (adjust these values accordingly)
        x1, x2, y1, y2 = 11.2, 15, -35, 40  # Define the zoomed-in region

        # # Set the limits for the inset axes
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        #
        # for i in reversed(range(0, self.dimension)):
        #     axins.plot(a_list, eigenvalues_list[:, i])
        #
        # axins.arrow(13.25, -25, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.arrow(13.25, 0.5, 0, -23.2, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.text(13.4, -13, r'$\Omega$', ha='center', va='center', fontsize=16)
        #
        # axins.arrow(13.5, 0, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.arrow(13.5, 23.7, 0, -22.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.text(13.65, 13, r'$\Omega$', ha='center', va='center', fontsize=16)
        #
        # # Customize the appearance of tick labels in the inset axes
        # axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.7)
        # plt.draw()

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    t = 5
    dt = 0.01
    n = 2
    δ_start = -190
    δ_end = 300

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['quench'] * 3

    five = ['linear'] * 5
    five1 = ['linear flat'] * 5
    five2 = ['quench'] * 5

    plotter = PlotSingle(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=two3,
                         initial_state_list=[0, 0],
                         )

    plotter.colour_bar(show=True)

    #plotter.eigenvalues_distance(show=True, save_pdf=True)

    plotter.eigenenergies_lineplot_with_state_fidelities('psi plus', detuning=True, save_pdf=True, show=True)

    plotter.state_fidelity([[0, 0]])

    plotter.eigenenergies_lineplot(detuning=True)
