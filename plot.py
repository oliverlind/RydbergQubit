from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adiabatic_evolution import AdiabaticEvolution
import numpy as np
import unicodeit


class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, rabi_osc=False, no_int=False):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, rabi_osc=rabi_osc)
        if no_int:
            self.C_6 = 0

    def plot_colour_bar(self):

        rydberg_fidelity_data = self.time_evolve(rydberg_fidelity=True)

        print(self.r_b)

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

        print(unicodeit.replace("R_1"))

        # Label figure
        ax.set_xlabel('Time (μs)')
        plt.title(f'Rabi Oscillations: Nearest Neighbour Blockade ( {"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)') #Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
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

        plt.figure(figsize=(10, 5))

        plt.title(f'Rabi Oscillations: Nearest Neighbour Blockade ( {"$R_{b}$"}={round(self.r_b,2)}μm, a={self.a}μm)') #Ω={int(self.Rabi/(2*np.pi))}(2πxMHz),
        plt.ylabel('Rydberg Probability')
        plt.xlabel('Time')

        for i in range(self.n-1):
            n_i = rydberg_fidelity_list[i]
            if i == 0:
                plt.plot(self.times, n_i, label=f'Atom 1 and 3')
            else:
                plt.plot(self.times, n_i, label=f'Atom {i + 1}')

        plt.legend(loc='upper right')

        plt.show()


if __name__ == "__main__":
    t = 10
    dt = 0.005
    n = 3
    δ_start = -15 * 2 * np.pi
    δ_end = 15 * 2 * np.pi

    evol = Plot(n, t, dt, δ_start, δ_end, rabi_osc=True)

    evol.plot()
