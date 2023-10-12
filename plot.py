from matplotlib import pyplot as plt

from adiabatic_evolution import AdiabaticEvolution
import numpy as np


class Plot(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end)

    def plot_colour_bar(self):

        rydberg_fidelity_data = self.time_evolve(rydberg_fidelity=True)

        # Labels for the bars
        labels = [f'Atom {i + 1}' for i in range(self.n)]

        # Create a horizontal bar with changing colors for each data set
        fig, ax = plt.subplots()

        cmap = plt.get_cmap('plasma')  # Choose a colormap for data sets

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

        cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.02])  # Adjust the [x, y, width, height] values
        #
        bar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, orientation='horizontal')

        plt.show()


if __name__ == "__main__":
    t = 50
    dt = 0.1
    n = 5
    δ_start = -10
    δ_end = 10

    evol = Plot(n, t, dt, δ_start, δ_end)

    evol.plot_colour_bar()
