from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adiabatic_evolution import AdiabaticEvolution
import numpy as np
import unicodeit
import time
from scipy.linalg import expm



class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, rabi_osc=False, no_int=False):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, rabi_osc=rabi_osc)
        if no_int:
            self.C_6 = 0

    def plot_colour_bar(self):

        self.linear_detunning()

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
        plt.title(f'Rydberg Probabilities: Linear Adiabatic Evolution ( {"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)') #Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
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
                    v = eigenvector[:,i]
                    p = abs(np.dot(v, ψ)[0])**2
                    ps += [p]


                # eigenvalue = np.sort(eigenvalue)
                eigenvalues += [eigenvalue]
                eigenvalue_probs += [ps]

            print(eigenvalue_probs)
            print(eigenvalues)

        else:

            for k in range(0, self.steps):
                h_m = self.hamiltonian_matrix(self.detunning[k])
                eigenvalue = np.linalg.eigvals(h_m)
                # eigenvalue = np.sort(eigenvalue)
                eigenvalues += [eigenvalue]

            eigenvalues = np.array(eigenvalues)
            print(eigenvalues)

            if self.n == 1:
                fig, ax1 = plt.subplots()

                plt.title(f'Single Atom Linear Detuning')
                for i in range(0, self.dimension):
                    if i == 0:
                        ax1.plot(self.detunning, eigenvalues[:, i], label=f'|g⟩')
                    else:
                        ax1.plot(self.detunning, eigenvalues[:, i], label=f'|r⟩')


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




if __name__ == "__main__":
    start_time = time.time()

    t = 4
    dt = 0.1
    n = 1
    δ_start = -20
    δ_end = 20

    evol = Plot(n, t, dt, δ_start, δ_end)

    evol.energy_eigenvalues()

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")