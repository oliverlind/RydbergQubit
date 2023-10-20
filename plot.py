import sys

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from adiabatic_evolution import AdiabaticEvolution
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import unicodeit
import time
from scipy.linalg import expm
import math
import plotly.graph_objects as go
import data_analysis as da



class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, no_int=False, detuning_type=None):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type)
        if no_int:
            self.C_6 = 0

    def plot_colour_bar(self):

        #self.linear_detunning()
        #self.linear_detunning_quench()

        rydberg_fidelity_data = self.time_evolve(rydberg_fidelity=True)

        print(rydberg_fidelity_data)

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


        # Label figure
        ax.set_xlabel('Time (μs)')
        plt.title(f'Z2 Configuration ( {"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)') #Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
        #plt.title(f'Rabi Oscillations: No Interaction (V=0)')

        # Make room for colour bar
        fig.subplots_adjust(right=0.8)


        # Adjust colour bar
        cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])  # Adjust the [x, y, width, height] values
        bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation="vertical")
        bar.set_label("Rydberg Probabilty")  # label colour bar

        plt.show()

    def plot(self):

        rydberg_fidelity_list = self.time_evolve(rydberg_fidelity=True)

        plt.Figure()

        plt.figure(figsize=(9, 4))

        plt.title(f'Rabi Oscillations ( {"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)') #Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
        plt.ylabel('Rydberg Probability')
        plt.xlabel('Time (μs)')

        for i in range(self.n-1):
            n_i = rydberg_fidelity_list[i]
            if i == 0:
                plt.plot(self.times, n_i, label=f'Atom 1 and 3')
            else:
                plt.plot(self.times, n_i, label=f'Atom {i + 1}')

        plt.legend(loc='upper right')

        plt.show()

    def energy_eigenvalues(self,twinx=False, probabilities=False):
        eigenvalues = []
        eigenvalue_probs = []
        dict = {}
        #self.linear_step_detunning()

        if probabilities:
            ψ = self.ground_state()
            j = self.row_basis_vectors(2 ** (self.n - 1))

            for k in range(0, self.steps):
                h_m = self.hamiltonian_matrix(self.detunning[k])
                ψ = np.dot(expm(-1j * h_m * self.dt), ψ)

                eigenvalue, eigenvector = np.linalg.eigh(h_m)


                ps = []
                for i in range(self.dimension):
                    v = eigenvector[:,i]
                    p = abs(np.dot(v, ψ)[0])**2
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
                    time_values = self.times # Replace with your actual time values

                    ax2.set_xticks(time_values)
                    ax2.set_xlabel('Time (s)')

                plt.show()

            if n == 2:
                fig, ax = plt.subplots(figsize=(10, 6))

                legend_labels = ["$E_{0}$", "$E_{1}$", "$E_{2}$", "$E_{3}$"]

                for i in range(0, self.dimension):
                    ax.plot(self.detunning, eigenvalues[:, i], label=f'{legend_labels[i]}')

                plt.title(f'Two Atom System: Linear detunning increase ({"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)')
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

    def single_atom_eigenstates(self, quench=False, expect_values=False, show=False):

        if quench:
            self.linear_detunning_quench()

        eigenvalues = []
        expectation_values = []

        density_matrices = self.time_evolve(density_matrix=True)

        num_of_plots = 11

        int_list = np.array(self.detunning, dtype=int)
        zero_position = np.where(int_list == 0)
        zero_position = zero_position[0]
        zero_position = [len(zero_position)//2]

        plot_step = self.steps // (num_of_plots-1)
        step = np.arange(0,self.steps+1, plot_step)
        step[-1] = step[-1]-1
        print(step)

        # Create a figure and axes
        fig, axes = plt.subplots(nrows=1, ncols=11, figsize=(12, 3))

        for i, ax in enumerate(axes):
            δ = self.detunning[step[i]]
            h_m = self.hamiltonian_matrix(δ)
            density_matrix = density_matrices[step[i]]

            abs_matrix = np.abs(density_matrix)
            phase_matrix = np.abs(np.angle(density_matrix))


            # print(matrix)
            # print(phase_matrix)


            # Create a colormap (you can choose a different colormap if desired)
            cmap = plt.get_cmap('RdYlGn')

            # Set the extent of the heatmap to match the dimensions of the matrix
            extent = [0, 4, 0, 4]

            # Display the matrix as a heatmap
            im = ax.imshow(phase_matrix, cmap=cmap, extent=extent, alpha=abs_matrix, vmin=0, vmax=np.pi)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f'Δ={round(self.detunning[step[i]])}')  # Empty x-label
            ax.set_ylabel('')  # Empty y-label


        plt.show()



        for k in range(0, self.steps):
            h_m = self.hamiltonian_matrix(self.detunning[k])
            density_matrix = density_matrices[k]

            expectation_value = da.expectation_value(density_matrix, h_m)

            eigenvalue, eigenvector = np.linalg.eigh(h_m)
            eigenvalues += [eigenvalue]
            expectation_values += [expectation_value]

        eigenvalues = np.array(eigenvalues)

        # Create a horizontal bar with changing colors for each data set
        fig, ax = plt.subplots(figsize=(10, 5))

        plt.title(f'Single Atom Linear Detuning')
        ax.plot(self.times, expectation_values)

        for i in range(0, self.dimension):
            if i == 0:
                ax.plot(self.times, eigenvalues[:, i], label=f'{"$E_{0}$"}')
            else:
                ax.plot(self.times, eigenvalues[:, i], label=f'{"$E_{1}$"}')

        plt.show()



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

    t = 5.00
    dt = 0.01
    n = 2
    δ_start = -200
    δ_end = 200

    evol = Plot(n, t, dt, δ_start, δ_end)

    #evol.single_atom_eigenstates()

    evol.single_atom_eigenstates()


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")