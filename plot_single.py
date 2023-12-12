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

mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams["text.latex.preamble"]  = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
# mpl.rcParams["text.usetex"] = True
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

    def colour_bar(self, type='RF', title=None, ax=None, show=False):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if type == 'RF':
            data = self.time_evolve(rydberg_fidelity=True)

        ploting_tools.set_up_color_bar(self.n, data, self.times, ax=ax, type=type)

        if show:
            plt.show()

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

    def eigenvalue_lineplot(self, eigenvalues=None, ax=None, title=None, show=False):

        if eigenvalues is None:

            eigenvalues, eigenvectors = self.time_evolve(eigen_list=True)

            fig, ax = plt.subplots(figsize=(12, 6.5))

            ax.set_xlabel('Time (μs)')

        eigenvalues = np.array(eigenvalues)
        labels = cf.energy_eigenvalue_labels(self.n)
        ax.set_ylabel('Energy')

        # Plot eigenvalues
        for i in range(0, self.dimension):
            ax.plot(self.times, eigenvalues[:, i], label=labels[i])

        ax.legend(loc='upper right')

        if show:
            plt.show()

    def eigenvalues_distance(self):
        fig, ax = plt.subplots(figsize=(12, 6.5))

        ax.set_xlabel('Distance between Atoms (μm)', fontsize=12)
        ax.set_ylabel(f'Energy Eigenvalue {"($ħ^{-1}$)"}', fontsize=12)

        a_list = np.linspace(5.5, 15, 100)

        eigenvalues_list = []

        for a in a_list:
            h_m = RydbergHamiltonian1D(2, a=a).hamiltonian_matrix([0])

            eigenvalues, eigenvector = np.linalg.eigh(h_m)

            eigenvalues_list += [eigenvalues]

        labels = ['|00⟩', '|$Ψ^{-}$⟩', '|$Ψ^{+}$⟩', '|rr⟩']


        eigenvalues_list = np.array(eigenvalues_list)

        for i in reversed(range(0, 4)):
            ax.plot(a_list, eigenvalues_list[:, i], label=labels[i], linewidth=3, zorder=1)

        ax.set_xlim(min(a_list), max(a_list))

        ax.spines['right'].set_position(('data', max(a_list)))

        pale_blue = (0.7, 0.8, 1.0)
        # plt.axvline(x=self.r_b, color=pale_blue, linestyle='-',linewidth=10, alpha=0.5)
        plt.axvline(x=self.r_b, color=pale_blue, linestyle='--', linewidth=2, alpha=0.8)

        print(self.Rabi)

        plt.text(self.r_b+0.15, 125, 'Blockade Radius $R_{B}$', ha='center', va='center', rotation=-90, fontsize=12)

        plt.text(5.25, 315, r'$|rr⟩$', ha='center', va='center', fontsize=18)  # Displaying 1/2

        plt.axvspan(xmin=5.5, xmax=self.r_b, color=pale_blue, alpha=0.2)


        # plt.arrow(13, -25+23.7, 0, -22.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)

        # Create an inset axes in the top right corner
        axins = inset_axes(plt.gca(), width="45%", height="40%", loc='upper right')

        # Specify the region to zoom in (adjust these values accordingly)
        x1, x2, y1, y2 = 11.2, 15, -35, 40  # Define the zoomed-in region


        # Set the limits for the inset axes
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        for i in reversed(range(0, self.dimension)):
            axins.plot(a_list, eigenvalues_list[:, i])

        axins.arrow(13.25, -25, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        axins.text(13.4, -13, r'$Ω$', ha='center', va='center', fontsize=16)

        axins.arrow(13.5, 0, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        axins.text(13.65, 13, r'$Ω$', ha='center', va='center', fontsize=16)


        # Customize the appearance of tick labels in the inset axes
        axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

        #mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.7)
        #plt.draw()

        ax.legend(fontsize=16)

        plt.savefig('output_plot.svg', format='svg')

        plt.show()


if __name__ == "__main__":
    t = 2
    dt = 0.01
    n = 2
    δ_start = -200
    δ_end = 300

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    plotter = PlotSingle(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=two3,
                         initial_state_list=[0, 0],
                         )

    #plotter.eigenvalue_lineplot(show=True)

    plotter.eigenvalues_distance()








