from matplotlib import pyplot as plt

from adiabatic_evolution import AdiabaticEvolution
import numpy as np


class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, rabi_osc=False):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, rabi_osc=rabi_osc)

    def plot_colour_bar(self):

        rydberg_fidelity_data = self.time_evolve(rydberg_fidelity=True)

        # Labels for the bars
        labels = [f'Atom {i + 1}' for i in range(self.n)]

        # Create a horizontal bar with changing colors for each data set
        fig, ax = plt.subplots()

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

        cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.8])  # Adjust the [x, y, width, height] values
        #
        bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax)

        # Label the color bar
        bar.set_label("Rydberg Probabilty")

        plt.show()


if __name__ == "__main__":
    t = 20
    dt = 0.1
    n = 3
    δ_start = -10
    δ_end = 20

    evol = Plot(n, t, dt, δ_start, δ_end, rabi_osc=True)

    evol.plot_colour_bar()
