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


class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, no_int=False, detuning_type=None, single_addressing_list=None,
                 initial_state_list=None, rabi_regime='constant'):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list,
                         rabi_regime=rabi_regime)
        if no_int:
            self.C_6 = 0

    def plot_colour_bar(self, show=False):

        # self.linear_detunning()
        # self.linear_detunning_quench()

        rydberg_fidelity_data = self.time_evolve(rydberg_fidelity=True)

        # Labels for the bars
        labels = [f'Atom {i + 1}' for i in range(self.n)]

        # Create a horizontal bar with changing colors for each data set
        fig, ax = plt.subplots(figsize=(10, 5))

        cmap = plt.get_cmap('viridis')  # Choose a colormap for data sets

        for i, ind_data in enumerate(rydberg_fidelity_data):
            for j, value in enumerate(ind_data):
                color = cmap(value)  # Map value to color using the colormap
                ax.barh(i, 1, left=self.times[j], height=1, color=color, align='center')

        # Set the y-axis ticks and labels
        ax.set_yticks(np.arange(self.n))
        ax.set_yticklabels(labels)

        # Fill whole figure
        ax.set_xlim(0, self.t)  # Set the x-axis limits
        ax.set_ylim(-0.5, self.n - 0.5)  # Set the y-axis limits
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.tick_params(which='minor', size=4)

        # Label figure
        ax.set_xlabel('Time (s)')
        # plt.title(
        # f'{"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm')  # Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
        # plt.title(f'Rabi Oscillations: No Interaction (V=0)')

        # Make room for colour bar
        fig.subplots_adjust(right=0.8)

        # Adjust colour bar
        cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])  # Adjust the [x, y, width, height] values
        bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation="vertical")
        bar.set_label("Rydberg Probabilty")  # label colour bar

        # Add a description below the plot
        # description = f' {"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm'
        plt.text(2.5, -2, description, fontsize=12, ha='center')

        if show:
            plt.tight_layout
            plt.xlabel('Time (s)')
            plt.show()

    def plot_line_rydberg_prob(self):

        rydberg_fidelity_list = self.time_evolve(rydberg_fidelity=True)

        plt.figure(figsize=(9, 4))

        plt.title(
            f'Two Atom System: Linear increase and global quench ( {"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)')  # Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
        plt.ylabel('Rydberg Probability')
        plt.ylim(0, 1)
        plt.xlabel('Time (μs)')

        for i in range(self.n):
            n_i = rydberg_fidelity_list[i]
            plt.plot(self.times, n_i, label=f'Atom {i + 1}')

        plt.legend(loc='upper right')

        plt.show()

    def plot_line_bell_pos_sup_prob(self, bell_fidelity_types=['psi plus']):
        bell_fidelity_data = self.time_evolve(bell_fidelity_types=bell_fidelity_types)

        plt.figure(figsize=(9, 4))

        plt.ylabel(f'|{"$Ψ^{-}$"}⟩ Probability')
        plt.ylim(0, 1)
        plt.xlabel('Time (μs)')

        for bell_type in bell_fidelity_types:
            for i in range(self.n - 1):
                bf = bell_fidelity_data[bell_type][i]
                print(bf)
                plt.plot(self.times, bf, label=f'Atom {i + 1} and {i + 2}')

        plt.legend(loc='upper left')

        plt.show()

    def energy_eigenvalues(self, twinx=False, probabilities=False):
        eigenvalues = []
        eigenvalue_probs = []
        dict = {}
        # self.linear_step_detunning()

        if probabilities:
            ψ = self.ground_state()
            j = self.row_basis_vectors(2 ** (self.n - 1))

            for k in range(0, self.steps):
                h_m = self.hamiltonian_matrix(self.detunning[k])
                ψ = np.dot(expm(-1j * h_m * self.dt), ψ)

                eigenvalue, eigenvector = np.linalg.eigh(h_m)

                ps = []
                for i in range(self.dimension):
                    v = eigenvector[:, i]
                    p = abs(np.dot(v, ψ)[0]) ** 2
                    ps += [p]

                eigenvalues += [eigenvalue]
                eigenvalue_probs += [ps]

            print(eigenvalues[-1])
            print(eigenvector)
            print(eigenvalue_probs[-1])


        else:

            for k in range(0, self.steps):
                h_m = self.hamiltonian_matrix(self.detunning[k])
                eigenvalue = np.linalg.eigvals(h_m)
                eigenvalue = np.sort(eigenvalue)
                eigenvalues += [eigenvalue]

            eigenvalues = np.array(eigenvalues)
            print(eigenvalues)

            if self.n == 1:
                fig, ax1 = plt.subplots()

                plt.title(f'Single Atom Linear Detuning')
                for i in range(0, self.dimension):
                    if i == 0:
                        ax1.plot(self.detunning, eigenvalues[:, i], label=f'{"$E_{0}$"}')
                    else:
                        ax1.plot(self.detunning, eigenvalues[:, i], label=f'{"$E_{1}$"}')

                ax1.set_xlabel('Δ (2πxMHz)')
                ax1.set_ylabel('Energy Eigenvalue')
                ax1.legend()

                if twinx:
                    # Create a twin Axes
                    ax2 = ax1.twiny()

                    # Define the time values for the top x-axis
                    time_values = self.times  # Replace with your actual time values

                    ax2.set_xticks(time_values)
                    ax2.set_xlabel('Time (s)')

                plt.show()

            if n == 2:
                fig, ax = plt.subplots(figsize=(10, 6))

                legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$"]

                for i in range(0, self.dimension):
                    ax.plot(self.detunning, eigenvalues[:, i], label=f'{legend_labels[i]}')

                plt.title(
                    f'Two Atom System: Linear detunning increase ({"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)')
                plt.xlabel('Δ (MHz)')
                plt.ylabel(f'Energy Eigenvalue {"($ħ^{-1}$)"}')

                handles, labels = plt.gca().get_legend_handles_labels()
                plt.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=10)

                # Create an inset axes in the top right corner
                axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')

                # Specify the region to zoom in (adjust these values accordingly)
                x1, x2, y1, y2 = 70, 120, -50, 50  # Define the zoomed-in region

                # Set the limits for the inset axes
                axins.set_xlim(x1, x2)
                axins.set_ylim(y1, y2)

                # Customize the appearance of tick labels in the inset axes
                axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

                # axins.set_xticks([])
                # axins.set_yticks([])

                # Plot the zoomed-in data in the inset axes
                for i in range(0, self.dimension):
                    axins.plot(self.detunning, eigenvalues[:, i], label=f'{i}')

                # Add a border around the inset axes
                axins.set_facecolor('white')

                # Create a dotted rectangle to highlight the region of interest
                # rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, linestyle='--', edgecolor='red')
                # plt.gca().add_patch(rect)

                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
                plt.draw()

                # plt.legend()

                plt.show()

    def energy_eigenstates(self, num_of_plots=17, showtime=False, show_detunning_plot=False,
                           insert_axins=False):

        legend_labels = cf.energy_eigenvalue_labels(self.n)
        eigenvalues = []
        expectation_values = []
        density_matrices = {'QS': {}}

        qs_density_matrices = self.time_evolve(density_matrix=True)

        plot_step = self.steps // (num_of_plots - 1)
        step = np.arange(0, self.steps + 1, plot_step)
        step[-1] = step[-1] - 1
        print(step)

        # Create a figure and axes
        fig, axes = plt.subplots(nrows=self.dimension + 1, ncols=num_of_plots, figsize=(12, 5))

        for i in range(0, num_of_plots):
            # Detunning value for each colour map plot
            δ = self.detunning[:, step[i]]
            print(δ)
            # Get quantum state density matrix
            qs_density_matrix = qs_density_matrices[step[i]]
            abs_matrix = np.abs(qs_density_matrix)
            phase_matrix = np.abs(np.angle(qs_density_matrix))

            density_matrices['QS']['Abs'] = abs_matrix
            density_matrices['QS']['Phase'] = phase_matrix

            # Get eigenstate density matrices
            h_m = self.hamiltonian_matrix(δ)
            eigenvalue, eigenvector = np.linalg.eigh(h_m)

            for j in range(0, self.dimension):
                v = eigenvector[:, j]
                v = v.reshape(-1, 1)
                es_dm = np.dot(v, v.conj().T)
                abs_matrix = np.abs(es_dm)
                phase_matrix = np.abs(np.angle(es_dm))

                density_matrices[f'E{j}'] = {}
                density_matrices[f'E{j}']['Abs'] = abs_matrix
                density_matrices[f'E{j}']['Phase'] = phase_matrix

            # Create a colormap (you can choose a different colormap if desired)
            cmap = plt.get_cmap('RdYlGn')

            # Set the extent of the heatmap to match the dimensions of the matrix
            extent = [0, self.dimension, 0, self.dimension]

            # Display the qs matrix as a heatmap
            ax = axes[self.dimension, i]

            ax.imshow(density_matrices['QS']['Phase'], cmap=cmap, extent=extent, alpha=density_matrices['QS']['Abs'],
                      vmin=0, vmax=np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')

            # Labelling
            if len(δ) > 1 or showtime:
                ax.set_xlabel(f'{round(self.times[step[i]], 1)}μs')
            else:
                ax.set_xlabel(f'Δ={round(δ[0])}')

            if i == 0:
                ax.set_ylabel('QS', rotation='horizontal', labelpad=40, fontsize=16, verticalalignment='center')

            # Display eigenstates matrices as heatmap
            for j in range(0, self.dimension):
                ax = axes[j, i]

                plot_list = np.arange(self.dimension - 1, -1, -1)
                p = plot_list[j]

                ax.imshow(density_matrices[f'E{p}']['Phase'], cmap=cmap, extent=extent,
                          alpha=density_matrices[f'E{p}']['Abs'], vmin=0, vmax=np.pi)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel('')

                # Labelling
                if i == 0:
                    ax.set_ylabel(legend_labels[p], rotation='horizontal', labelpad=40, fontsize=16,
                                  verticalalignment='center')

        plt.show()

        # Eigenvalue Line Plot

        # Get eigen and expectation values energy
        for k in range(0, self.steps):
            δ = self.detunning[:, k]
            h_m = self.hamiltonian_matrix(δ)
            density_matrix = qs_density_matrices[k]

            expectation_value = da.expectation_value(density_matrix, h_m)

            eigenvalue, eigenvector = np.linalg.eigh(h_m)

            eigenvalues += [eigenvalue]
            expectation_values += [expectation_value]

        eigenvalues = np.array(eigenvalues)

        fig, ax = plt.subplots(figsize=(12, 6.5))

        # Plot expectation
        if showtime or len(δ) > 1:
            ax.plot(self.times, expectation_values, label=legend_labels[-1])
            ax.set_xlabel('Time (μs)', fontsize=14)

        else:
            δ = self.detunning[0]
            ax.plot(δ, expectation_values, label=legend_labels[-1])
            ax.set_xlabel('Δ (Mhz)', fontsize=14)

        # Plot eigenvalues
        for i in range(0, self.dimension):
            if showtime or len(δ) > 1:
                ax.plot(self.times, eigenvalues[:, i], label=legend_labels[i])
            else:
                ax.plot(self.detunning, eigenvalues[:, i], label=legend_labels[i])

        plt.title(
            f'Three Atom System: Linear increase -60 to 60 ({"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)',
            fontsize=16)

        ax.set_ylabel(f'Energy Eigenvalue', fontsize=14)

        # reverse order of legend
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=10)

        if insert_axins:

            # Create an inset axes in the top right corner
            axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')

            # Specify the region to zoom in (adjust these values accordingly)
            x1, x2, y1, y2 = 70, 120, -50, 50  # Define the zoomed-in region

            # Set the limits for the inset axes
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Customize the appearance of tick labels in the inset axes
            axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

            # axins.set_xticks([])
            # axins.set_yticks([])

            # Plot the zoomed-in data in the inset axes
            for i in range(0, self.dimension):
                axins.plot(self.detunning, eigenvalues[:, i], label=f'{i}')

            # Add a border around the inset axes
            axins.set_facecolor('white')

            # Create a dotted rectangle to highlight the region of interest
            # rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, linestyle='--', edgecolor='red')
            # plt.gca().add_patch(rect)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            plt.draw()

            plt.tight_layout

        plt.show()

        if show_detunning_plot:
            plt.plot(self.times, self.detunning[0, :])
            plt.plot(self.times, self.detunning[1, :])
            plt.plot(self.times, self.detunning[2, :])
            plt.show()

    def eigenvalue_lineplot(self, eigenvalues=None, eigenvectors=None, expectation_energies=None, ax=None, eigen=True,
                            states=None, show=False):

        if ax is None:
            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs, states = self.time_evolve(
                expec_energy=True, eigen_list=True, eigenstate_fidelities=True, states_list=True)
            fig, ax = plt.subplots(figsize=(12, 6.5), label='Expec')
            plt.title(f'Adiabatic detunning sweep (Δ = {self.δ_start} to {self.δ_end}) followed by first atom quench')
            ax.set_xlabel('Time (μs)')

        eigenvalues = np.array(eigenvalues)

        labels = cf.energy_eigenvalue_labels(self.n)

        # ax.plot(self.times, expectation_energies, label=f'{labels[-1]}')
        ax.set_ylabel('Energy')

        # Plot eigenvalues
        if eigen:
            for i in range(0, self.dimension):
                ax.plot(self.times, eigenvalues[:, i], label=labels[i])

        plt.legend(loc='upper right')

        if show:
            plt.show()

    def eigenstate_fidelity_colorbar(self):
        eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(expec_energy=True,
                                                                                             eigen_list=True,
                                                                                             eigenstate_fidelities=True,
                                                                                             )
        fig, ax = plt.subplots()

        ploting_tools.set_up_color_bar(self.dimension, eigenstate_probs, self.times, ax=ax, type='eigen energies')

        plt.show()

    def entanglement_entropy(self, atom=None, show=False, ax=None, states=None, FT=False):

        if show:
            states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
        elif ax and states is not None:
            ax = ax
        else:
            sys.exit()

        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.tick_params(which='minor', size=4)
        ax.spines['left'].set_position('zero')
        ax.set_ylabel('Von Neumann Entropy')

        if atom is None:
            for j in range(1, self.n + 1):
                vne_list = []
                for i in range(0, self.steps):
                    rdm = self.reduced_density_matrix(states[i], j)
                    vne = da.von_nuemann_entropy(rdm)
                    vne_list += [vne]

                ax.plot(self.times, vne_list, label=f'Atom {j}')

        elif atom == 'sum':
            vne_lists = [[] for _ in range(self.n)]
            for j in range(1, self.n + 1):
                for i in range(0, self.steps):
                    rdm = self.reduced_density_matrix(states[i], j)
                    vne = da.von_nuemann_entropy(rdm)
                    vne_lists[j - 1] += [vne]

            vne_lists_array = np.array(vne_lists)

            vne_sum = np.sum(vne_lists_array, axis=0)

            if FT:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8))

                ax1.set_ylabel('Von Neumann Entropy Sum')
                ax1.plot(self.times, vne_sum)

                yf = fft(vne_sum)
                xf = fftfreq(self.steps, self.dt)[0:self.steps // 2]  # Range 0 to 1/2dt

                ax2.set_xlabel('Frequency (MHz)')
                ax2.plot(xf, 2.0 / self.steps * np.abs(yf[0:self.steps // 2]))

            else:

                ax.set_ylabel('Von Neumann Entropy Sum')
                ax.plot(self.times, vne_sum)


        else:
            vne_list = []
            for i in range(0, self.steps):
                rdm = self.reduced_density_matrix(states[i], atom)
                vne = da.von_nuemann_entropy(rdm)
                vne_list += [vne]

            ax.plot(self.times, vne_list, label=f'Atom {atom}')

        plt.legend(loc='upper left')

        if show:
            plt.xlabel(r'Time ($\mus)')
            plt.show()

    def entanglement_entropy_colorbar(self, atom=None, show=False, ax=None, states=None):

        if show:
            states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(12.5, 8))
            plt.xlim(0, self.t)
            ax = plt.gca()
        elif ax and states is not None:
            ax = ax
        else:
            sys.exit()

        vne_list = [[] for _ in range(self.n)]

        if atom is None:
            for j in range(1, self.n + 1):
                for i in range(0, self.steps):
                    rdm = self.reduced_density_matrix(states[i], j)
                    vne = da.von_nuemann_entropy(rdm)
                    vne_list[j - 1] += [vne]

        ploting_tools.set_up_color_bar(self.n, vne_list, self.times, ax=ax, type='vne', color='inferno', colorbar=True)

        if show:

            plt.show()

    def relative_entanglement_entropy(self, j, k, show=False, ax=None, states=None):

        quantum_relative_entropys = []

        if show:
            states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
        elif ax and states is not None:
            ax = ax
        else:
            sys.exit()

        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.tick_params(which='minor', size=4)
        ax.spines['left'].set_position('zero')

        for i in range(0, self.steps):
            rdm_1 = self.reduced_density_matrix(states[i], j)
            rdm_2 = self.reduced_density_matrix(states[i], k)

            quantum_relative_entropy = da.quantum_relative_entropy(rdm_1, rdm_2)

            quantum_relative_entropys += [quantum_relative_entropy]

        ax.plot(self.times, quantum_relative_entropys, label=f'Atom {j} with respect to Atom {k}')
        ax.set_ylabel('Quantum Relative Entropy')
        plt.legend()

        if show:
            plt.ylabel('Quantum Relative Entropy')
            plt.xlabel('Time (μs)')
            plt.show()

    def lineplot_and_rf_colorbar(self, states_to_test=None):

        # rydberg_fidelity_data, eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(
        #     rydberg_fidelity=True, expec_energy=True, eigen_list=True, eigenstate_fidelities=True)
        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 8))

        fig.suptitle(f'({"$Δ_{initial}$"} = {self.δ_start} )', fontsize=18)
        # fig.suptitle(f'Moving first Atom away 1 a per μs (Δ = {self.δ_end})', fontsize=18)
        # fig.suptitle(f'Adiabatic detunning sweep (Δ = {self.δ_start} to {self.δ_end}) followed by first atom quench', fontsize=18)

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1, colorbar=False)

        self.entanglement_entropy(ax=ax2, states=states, atom='sum')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def state_fidelity_rf_colorbar(self, states_to_test=None):

        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 8))

        # fig.suptitle(f'Quench Atom 1 from Δ = {self.δ_start} to Δ = 5 instant', fontsize=18)  # , {"$t_{quench}$ = 7.5 μs"}

        # fig.suptitle(f'Adiabatic quench Atom 1 ({"$Δ_{initial}$"} = {self.δ_start} {"$t_{quench}$ = 7.5 μs"})',
        # fontsize=18)

        # fig.suptitle(f'Generating GHZ State ({"$Δ_{1,4 final}$"} = {self.δ_end-50} , {"$Δ_{2,3 final}$"} = {self.δ_end})', fontsize=18)

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1, colorbar=False)

        self.state_fidelity(states_to_test, q_states=states, ax=ax2, sum_probs=True)

        ax2.legend(loc='upper right')

        # self.state_fidelity(states_to_test[2:], q_states=states, ax=ax3, sum_probs=True, colors_num=4)
        #
        # ax3.legend(loc='upper right')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def bell_fidelity_colorbar(self):
        bell_fidelity_data = self.time_evolve(bell_fidelity=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data, self.times, ax, type='bell', color='twilight')

        plt.show()

    def rf_bf_colorbars(self):
        rydberg_fidelity_data, bell_fidelity_data = self.time_evolve(rydberg_fidelity=True, bell_fidelity=True)

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

        plt.title("Moving Atom 1, 5a per μs")

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1)

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data, self.times, ax2, type='bell', color='cool')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def bell_fidelity_colorbars(self):
        bell_fidelity_data = self.time_evolve(bell_fidelity_types=['psi plus', 'psi minus'])

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi plus'], self.times, ax1, type='psi plus',
                                       color='cool')

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi minus'], self.times, ax2, type='psi minus',
                                       color='cool')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def rydberg_bell_fidelity_colorbars(self):

        plt.title("Moving Atom 1, 5a per μs")

        rydberg_fidelity_data, bell_fidelity_data = self.time_evolve(rydberg_fidelity=True,
                                                                     bell_fidelity_types=['psi plus', 'psi minus',
                                                                                          'phi plus', 'phi minus'])

        # Create subplots with shared x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1)

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi plus'], self.times, ax2, type='psi plus',
                                       color='cool')

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi minus'], self.times, ax3, type='psi minus',
                                       color='cool')

        # ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['phi plus'], self.times, ax4, type='phi plus',
        #                                color='cool')
        #
        # ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['phi minus'], self.times, ax5, type='phi minus',
        #                                color='cool')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def rdm_heatmaps(self):
        density_matrices = self.time_evolve(reduced_density_matrix_pair=True)

        ploting_tools.colormap_density_matrices(density_matrices, self.dt, self.times)

    def two_atom_eigenstates(self, probabilities=False, show=False):

        if self.n != 2:
            print('n is not 2!')
            sys.exit()

        eigenvalues = []
        eigenvalue_probs = []
        dict = {}
        # self.linear_step_detunning()

        if probabilities:
            ψ = self.ground_state()
            j = self.row_basis_vectors(2 ** (self.n - 1))

            for k in range(0, self.steps):
                h_m = self.hamiltonian_matrix(self.detunning[k])
                ψ = np.dot(expm(-1j * h_m * self.dt), ψ)

                eigenvalue, eigenvector = np.linalg.eigh(h_m)

                ps = []
                for i in range(self.dimension):
                    v = eigenvector[:, i]
                    p = abs(np.dot(v, ψ)[0]) ** 2
                    ps += [p]

                eigenvalues += [eigenvalue]
                eigenvalue_probs += [ps]

            print(eigenvalues[-1])
            print(eigenvector)
            print(eigenvalue_probs[-1])

        for k in range(0, self.steps):
            h_m = self.hamiltonian_matrix(self.detunning[k])
            # eigenvalue = np.linalg.eigvals(h_m)
            # eigenvalue = np.sort(eigenvalue)
            # eigenvalues += [eigenvalue]
            eigenvalue, eigenvector = np.linalg.eigh(h_m)
            eigenvalues += [eigenvalue]

        eigenvalues = np.array(eigenvalues)

        if show:

            fig, ax = plt.subplots(figsize=(10, 6))

            legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$"]

            for i in range(0, self.dimension):
                ax.plot(self.detunning, eigenvalues[:, i], label=f'{legend_labels[i]}')

            plt.title(f'Two Atom System: Linear detunning increase ({"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)')
            plt.xlabel('Δ (MHz)')
            plt.ylabel(f'Energy Eigenvalue {"($ħ^{-1}$)"}')

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(reversed(handles), reversed(labels), loc='upper left', fontsize=10)

            # Create an inset axes in the top right corner
            axins = inset_axes(plt.gca(), width="30%", height="30%", loc='upper right')

            # Specify the region to zoom in (adjust these values accordingly)
            x1, x2, y1, y2 = 70, 120, -50, 50  # Define the zoomed-in region

            # Set the limits for the inset axes
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            # Customize the appearance of tick labels in the inset axes
            axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

            # axins.set_xticks([])
            # axins.set_yticks([])

            # Plot the zoomed-in data in the inset axes
            for i in range(0, self.dimension):
                axins.plot(self.detunning, eigenvalues[:, i], label=f'{i}')

            # Add a border around the inset axes
            axins.set_facecolor('white')

            # Create a dotted rectangle to highlight the region of interest
            # rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=True, linestyle='--', edgecolor='red')
            # plt.gca().add_patch(rect)

            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            plt.draw()

            # plt.legend()

            plt.show()

        def play(self):
            pass

    def state_fidelity(self, states_to_test, q_states=None, ax=None, show=False, sum_probs=False, colors_num=0):

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

        plt.ylim(0, 1)

        num_of_test_states = len(states_to_test)

        state_fidelities = [[] for _ in range(num_of_test_states)]

        for i in range(0, num_of_test_states):

            state_to_test = states_to_test[i]
            label = ploting_tools.state_label(state_to_test)

            v_state_to_test = self.initial_state(state_to_test)

            for j in range(0, self.steps):
                state_fidelity = data_analysis.state_prob(v_state_to_test, q_states[j])
                state_fidelities[i] += [state_fidelity]

            cn = colors_num + i

            ax.plot(self.times, state_fidelities[i], label=f'{label}', color=plotcolors[cn])

        if sum_probs:

            state_fidelities = np.array(state_fidelities)

            sum_fidelities = np.linspace(0, 0, self.steps)

            for i in range(0, num_of_test_states):
                sum_fidelities = sum_fidelities + state_fidelities[i]

            ax.plot(self.times, sum_fidelities, label='Sum', alpha=0.65, color='grey')
        #
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.tick_params(which='minor', size=4)
        ax.spines['left'].set_position('zero')
        ax.set_ylabel('Probability')

        ax.legend(loc='upper right')

        if show:
            plt.show()

    def cmap_energyeigenvalues(self, eigenvalues=None, ax=None, eigen=True,
                               show=False):

        if ax is None:
            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(expec_energy=True,
                                                                                                 eigen_list=True,
                                                                                                 eigenstate_fidelities=True)
            fig, ax = plt.subplots(figsize=(12, 6.5), label='Expec')
            plt.title(f'Slow quench Atom 1  (Δ = {self.δ_start} to Δ = 0 in t = 5 μs)')
            ax.set_xlabel('Time (μs)')

        eigenvalues = np.array(eigenvalues)
        ax.set_ylabel('Energy')

        # Plot eigenvalues

        # Create a ScalarMappable object
        normalize = plt.Normalize(vmin=0, vmax=1)  # Define the normalization range [0, 1]
        colormap = cm.viridis  # Choose a colormap ('viridis' in this case)
        scalar_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalar_map.set_array([])  # Set an empty array to allow the ScalarMappable to work

        if eigen:
            for i in range(0, 4):
                colors = eigenstate_probs[i]
                ax.scatter(self.times, eigenvalues[:, i], cmap='viridis', c=scalar_map.to_rgba(colors), s=1)

        if show:
            plt.show()

    def eigenstate_table(self, time=0):

        eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                                             expec_energy=True,
                                                                                             eigenstate_fidelities=True)
        index = int(time / self.dt)

        print(np.shape(eigenvalues))

        eigenvalue_df = pd.DataFrame(eigenvalues[index])

        eigenstate_probs = np.array(eigenstate_probs)
        eigenstate_probs = eigenstate_probs.T

        probs_df = pd.DataFrame(eigenstate_probs[index])

        eigenvectors = eigenvectors[index]

        # comp_basis_probs = np.multiply(eigenvectors, eigenvectors)

        eigenstate_df = pd.DataFrame(eigenvectors)

        df = pd.concat([eigenvalue_df.T, probs_df.T, eigenstate_df], axis=0)

        binary_strings = ploting_tools.ascending_binary_strings(self.n)

        df.index = ['Energy Eigenvalue', 'Probability'] + binary_strings
        path = 'Output csv tables/data.csv'

        df.to_csv(path, index=True)

    def eigenstate_basis(self, state_to_convert=[]):

        eigenvalues, eigenvectors, states = self.time_evolve(eigen_list=True,
                                                             states_list=True)

        index = int(time / self.dt)

        v = self.initial_state(state_to_convert)
        eigenvectors_matrix = eigenvectors[index]

        inv_eigenvectors_matrix = np.linalg.inv(eigenvectors_matrix)

        v = 2

    def rf_and_quench(self, single_addressing_list=None):

        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12.5, 8))

        fig.suptitle(f'Quench Atom 1 (Δ = {self.δ_start})', fontsize=18)  # , {"$t_{quench}$ = 7.5 μs"}

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1, colorbar=False)

        # self.entanglement_entropy_colorbar(ax=ax2, states=states)

        detuning_regimes.driving_quench(self.t, self.dt, self.δ_start, self.δ_end, self.steps, ax=ax2)

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def rabi_and_detuning_shape(self, ax=None, show=False):

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(11, 6))
        else:
            ax1 = ax

        detuning_1 = detuning_regimes.linear_detuning_flat(self.δ_start, self.δ_end, self.steps)
        detuning_2 = detuning_regimes.linear_detuning_flat(self.δ_start, self.δ_end - 50, self.steps)

        rabi = rabi_regimes.pulse_start(self.steps)

        ax1.set_xlabel('Time (μs)')
        ax1.set_ylabel('Δ (2π x Mhz)')
        ax1.plot(self.times, detuning_1, color='r', label='Atom 1, 4')
        ax1.plot(self.times, detuning_2, color='g', label='Atom 2, 3, 4, 5')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel(f'{"Ω / $Ω_{0}$"}', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.times, rabi, color=color)
        ax2.fill_between(self.times, rabi, 0, color='blue', alpha=.1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-1, 1.1)

        ax1.legend(loc="upper right")

        if show:
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

    def ghz_fidelity(self, density_matrices=None, ax=None, show=False):

        if density_matrices is None:
            qs_density_matrices = self.time_evolve(density_matrix=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            plt.title(f'Title')
            ax.set_xlabel('Time (μs)')
        else:
            qs_density_matrices = density_matrices
            ax = ax

        # Get GHZ fidelities

        ghz_fidelities = []

        v1 = (1 / np.sqrt(2)) * (self.initial_state([1, 0, 1, 0]) + self.initial_state([0, 1, 0, 1]))

        ghz_density_matrix = np.dot(v1, v1.conj().T)

        print(ghz_density_matrix)

        for matrix in qs_density_matrices:
            ghz_fidelity = data_analysis.expectation_value(matrix, ghz_density_matrix)

            ghz_fidelities += [ghz_fidelity]

        # Plot Values

        ax.set_ylabel('GHZ Fidelity')
        ax.plot(self.times, ghz_fidelities)

        if show:
            plt.show()

    def colorbars_ghz(self):

        rydberg_fidelity_data, density_matrices = self.time_evolve(rydberg_fidelity=True, density_matrix=True)

        # Create subplots with shared x-axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12.5, 8))

        fig.suptitle(
            f'Generating GHZ State ({"$Δ_{1,6 final}$"} = {self.δ_end - 50} , {"$Δ_{2,3,4,5 final}$"} = {self.δ_end})',
            fontsize=18)

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1, colorbar=False)

        self.ghz_fidelity(density_matrices=density_matrices, ax=ax2)

        self.rabi_and_detuning_shape(ax=ax3)

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Frequency')

        plt.show()

    def fourier_transform_state_fidelity(self, states_to_test, interval=None, colors_num=0):

        if interval is None:
            interval = [0, self.steps]

        q_states = self.time_evolve(states_list=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8))

        ax1.set_xlabel('Time (μs)')
        ax1.set_xlabel('Frequency (MHz)')

        num_of_test_states = len(states_to_test)

        state_fidelities = [[] for _ in range(num_of_test_states)]

        for i in range(0, num_of_test_states):

            state_to_test = states_to_test[i]
            label = ploting_tools.state_label(state_to_test)

            v_state_to_test = self.initial_state(state_to_test)

            for j in range(0, self.steps):
                state_fidelity = data_analysis.state_prob(v_state_to_test, q_states[j])
                state_fidelities[i] += [state_fidelity]

            cn = colors_num + i

            ax1.plot(self.times, state_fidelities[i], label=f'{label}', color=plotcolors[cn])

            ax1.legend()

            yf = fft(state_fidelities[i])

            xf = fftfreq(self.steps, self.dt)[0:self.steps // 2]  # Range 0 to 1/2dt

            ax2.plot(xf, 2.0 / self.steps * np.abs(yf[0:self.steps // 2]))
            plt.show()

            print(state_fidelities[i])

    def fourier_transorm_ee(self):

        vne_sum = self.entanglement_entropy(atom='sum')

        yf = fft(state_fidelities[i])

        xf = fftfreq(self.steps, self.dt)[0:self.steps // 2]  # Range 0 to 1/2dt

        ax2.plot(xf, 2.0 / self.steps * np.abs(yf[0:self.steps // 2]))
        plt.show()

    def eigenstate_projection(self, time=0):

        eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                                             expec_energy=True,
                                                                                             eigenstate_fidelities=True)
        index1 = 449
        index2 = int(time / self.dt)

        x = ploting_tools.energy_labels(self.dimension)

        eigenstate_probs = np.array(eigenstate_probs)
        eigenstate_probs = eigenstate_probs.T

        probs_intial = eigenstate_probs[index1]
        probs_after = eigenstate_probs[index2]

        initial_array = np.zeros(self.dimension)
        initial_array[0] = 1

        fig = plt.figure(figsize=(12.5, 8))

        plt.bar(x, probs_intial, color='grey', alpha=0.3, label='Before Quench')
        plt.bar(x, probs_after, color='blue', label='After Quench')

        # Removing ticks on the top and right sides
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelsize=8)  # Remove ticks on x-axis (bottom and top)
        plt.tick_params(axis='y', which='both', right=False)  # Remove ticks on y-axis (right side)

        # Adding labels and title
        plt.legend(loc='upper right')
        plt.xlabel('Energy Eigenstates')
        plt.ylabel('Probability')
        plt.title(f'Quench 1st Atom Δ = {self.δ_start}')

        plt.show()

    def drop(self):
        pass


if __name__ == "__main__":
    start_time = time.time()

    t = 1
    dt = 0.01
    n = 2
    δ_start = 0
    δ_end = 0

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear flat', 'linear flat']

    three = ['quench'] + ['linear flat'] * 2
    three2 = ['flat positive'] * 3
    three3 = ['linear flat'] * 3
    three4 = ['linear flat'] + ['flat start'] * 2

    four = ['quench', 'linear flat', 'linear flat', 'linear flat']
    four2 = ['flat positive'] * 4
    four3 = ['linear flat 2'] + ['linear flat'] * 2 + ['linear flat 2']

    five = ['linear flat'] * 5
    five2 = ['flat positive'] * 5
    five3 = ['quench'] + ['linear flat'] * 4
    five4 = ['quench'] * 5
    five5 = ['linear flat'] + ['flat start'] * 4
    five6 = ['quench'] + ['linear flat'] * 3 + ['quench']
    five7 = ['driving quench'] * 5
    five8 = ['rabi osc'] * 2 + ['quench'] + ['rabi osc'] * 2
    five9 = ['quench'] + ['linear flat 2'] + ['linear flat'] * 2 + ['linear flat 2']
    five10 = ['quench'] + ['linear flat 2'] * 4

    six = ['quench'] + 5 * ['linear flat']
    six2 = ['linear flat 2'] + ['linear flat'] * 4 + ['linear flat 2']

    seven = ['quench'] + 6 * ['linear flat']
    seven2 = ['quench'] + 5 * ['linear flat'] + ['quench']
    seven3 = ['flat zero'] + ['flat positive'] * 6
    seven4 = ['linear flat'] + ['flat start'] * 6
    seven5 = 2 * ['linear flat'] + ['quench'] + 4 * ['linear flat']
    seven6 = ['quench']*7

    nine = ['quench'] + 8 * ['linear flat']

    evol = Plot(n, t, dt, δ_start, δ_end, detuning_type=None,
                single_addressing_list=two3,
                initial_state_list=[0, 0]
                )

    evol.eigenstate_table(time=0.2)

    evol.plot_colour_bar(show=True)

    #evol.entanglement_entropy(show=True)
    #evol.eigenvalue_lineplot(show=True)

    #evol.eigenstate_fidelity_colorbar()

    #
    #evol.entanglement_entropy_colorbar(show=True)
    #
    evol.bell_fidelity_colorbars()
    #
    #evol.colorbars_ghz()
    #
    # evol.state_fidelity_rf_colorbar([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]])
    #
    #evol.state_fidelity_rf_colorbar([[1, 0, 1, 0, 0]])


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


