import sys

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from adiabatic_evolution import AdiabaticEvolution
import numpy as np
import time
from scipy.linalg import expm
import data_analysis as da
import config.config as cf
import ploting_tools

mpl.rcParams['font.size'] = 14
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
    def __init__(self, n, t, dt, δ_start, δ_end, no_int=False, detuning_type=None, single_addressing_list=None, initial_state_list=None):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list)
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
        ax.set_xlabel('Time (μs)')
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
        description = f' {"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm'
        plt.text(2.5, -2, description, fontsize=12, ha='center')

        if show:
            plt.tight_layout
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
            plt.plot(self.times, n_i, label=f'Atom {i+1}')

        plt.legend(loc='upper right')

        plt.show()

    def plot_line_bell_pos_sup_prob(self, bell_fidelity_types=['psi plus']):
        bell_fidelity_data = self.time_evolve(bell_fidelity_types=bell_fidelity_types)

        plt.figure(figsize=(9, 4))


        plt.ylabel(f'|{"$Ψ^{-}$"}⟩ Probability')
        plt.ylim(0, 1)
        plt.xlabel('Time (μs)')

        for bell_type in bell_fidelity_types:
            for i in range(self.n-1):
                bf = bell_fidelity_data[bell_type][i]
                print(bf)
                plt.plot(self.times, bf, label=f'Atom {i+1} and {i+2}')

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
            f'Two Atom System: Linear increase and global quench ({"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)',
            fontsize=16)

        ax.set_ylabel(f'Energy Eigenvalue {"($ħ^{-1}$)"}', fontsize=14)

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

    def entanglement_entropy(self, atom=None, show=False, ax=None, states=None):

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

        if atom is None:
            for j in range(1, self.n + 1):
                vne_list = []
                for i in range(0, self.steps):
                    rdm = self.reduced_density_matrix(states[i], j)
                    vne = da.von_nuemann_entropy(rdm)
                    vne_list += [vne]

                ax.plot(self.times, vne_list, label=f'Atom {j}')

        else:
            vne_list = []
            for i in range(0, self.steps):
                rdm = self.reduced_density_matrix(states[i], atom)
                vne = da.von_nuemann_entropy(rdm)
                vne_list += [vne]

            ax.plot(self.times, vne_list, label=f'Atom {atom}')

        ax.set_ylabel('Von Neumann Entropy')
        plt.legend()

        if show:
            plt.xlabel('Time (μs)')
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

    def entanglement_entropy_and_colorbar(self):
        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)
        quantum_relative_entropys = []

        # Create subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

        ploting_tools.set_up_color_bar(self.n, rydberg_fidelity_data, self.times, ax1)

        #self.entanglement_entropy(ax=ax2, states=states,atom=1)
        self.relative_entanglement_entropy(7, 1, states=states, ax=ax2)

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

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi plus'], self.times, ax1,  type='psi plus', color='cool')

        ploting_tools.set_up_color_bar(self.n, bell_fidelity_data['psi minus'], self.times, ax2, type='psi minus', color='cool')

        # Adjust spacing between subplots and remove vertical space
        plt.subplots_adjust(hspace=0)

        # Set x axis label
        plt.xlabel('Time (μs)')

        plt.show()

    def rydberg_bell_fidelity_colorbars(self):

        rydberg_fidelity_data, bell_fidelity_data = self.time_evolve(rydberg_fidelity=True, bell_fidelity_types=['psi plus', 'psi minus', 'phi plus', 'phi minus'])

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


if __name__ == "__main__":
    start_time = time.time()

    t = 2
    dt = 0.001
    n = 5
    δ_start = 200
    δ_end = 200

    two = ['quench', 'linear flat']
    three = ['quench'] + ['linear flat', 'linear flat']
    four = ['quench', 'linear flat', 'linear flat', 'linear flat']
    four2 = ['flat positive']*4
    five = ['quench', 'linear flat', 'linear flat', 'linear flat', 'linear flat']
    five2 = ['flat zero'] + ['flat positive']*4
    six = ['quench'] + 5 * ['linear flat']
    seven = ['quench'] + 6 * ['linear flat']
    nine = ['quench'] + 8 * ['linear flat']

    evol = Plot(n, t, dt, δ_start, δ_end, detuning_type='rabi osc',
                single_addressing_list=five2,
                initial_state_list=[1, 0, 1]
                )


    #evol.rdm_heatmaps()

    evol.rydberg_bell_fidelity_colorbars()

    evol.plot_line_bell_pos_sup_prob()

    evol.entanglement_entropy_and_colorbar()

    # evol.entanglement_entropy(show=True)

    # evol.energy_eigenstates(showtime=True)

    #
    # evol.plot_line_rydberg_prob()

    evol.relative_entanglement_entropy(2, 1, show=True)

    evol.plot_colour_bar(show=True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")
