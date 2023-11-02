import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors


def set_up_color_bar(n, data, times, ax, type='rydberg', color='viridis', colorbar=True):
    # Labels for the bars
    if type == 'rydberg':
        labels = [f'Atom {i + 1}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = "Rydberg Probability"

    elif type == 'psi plus':
        n = n-1
        labels = [f'Atom {i + 1}, {i+2}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=0.8)
        bar_label = f'|{"$Ψ^{+}$"}⟩ Probability'

    elif type == 'psi minus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=0.8)
        bar_label = f'|{"$Ψ^{-}$"}⟩ Probability'

    else:
        sys.exit()

    cmap = plt.get_cmap(color)  # Choose a colormap for data sets

    for i, ind_data in enumerate(data):
        for j, value in enumerate(ind_data):
            color = cmap(norm(value))  # Map value to color using the colormap
            ax.barh(i, 1, left=times[j], height=1, color=color, align='center')

    # Set the y-axis ticks and labels
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)

    # Fill whole figure
    ax.set_xlim(0, times[-1])  # Set the x-axis limits
    ax.set_ylim(-0.5, n - 0.5)  # Set the y-axis limits
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.tick_params(which='minor', size=4)

    if colorbar:
        # Add a colorbar to show the mapping of values to colors
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # fake up the array
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label(bar_label)

def colormap_density_matrices(density_matrices, steps, num_of_plots=17, showtime=False):

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=num_of_plots, figsize=(12, 3))

    plot_step = steps // (num_of_plots - 1)
    step = np.arange(0, steps + 1, plot_step)
    step[-1] = step[-1] - 1
    print(step)

    for i in range(0, num_of_plots):

        density_matrix = density_matrices[step[i]]
        abs_matrix = np.abs(density_matrix)
        phase_matrix = np.abs(np.angle(density_matrix))

        # Create a colormap (you can choose a different colormap if desired)
        cmap = plt.get_cmap('RdYlGn')

        # Set the extent of the heatmap to match the dimensions of the matrix
        extent = [0, 4, 0, 4]

        # Display the rdm matrix as a heatmap
        ax = axes[0, i]

        ax.imshow(density_matrices['QS']['Phase'], cmap=cmap, extent=extent, alpha=density_matrices['QS']['Abs'],
                  vmin=0, vmax=np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')

    plt.show()




