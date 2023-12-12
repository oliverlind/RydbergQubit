import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
import config.config as cf


def set_up_color_bar(n, data, times, ax, type='rydberg', color='viridis', colorbar=True):
    # Labels for the bars
    if type == 'rydberg':
        labels = [f'Atom {i + 1}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = "Rydberg Probability"

    elif type == 'vne':
        labels = [f'Atom {i + 1}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=np.log(n))
        bar_label = "Von Neumann Entropy"

    elif type == 'psi plus':
        n = n-1
        labels = [f'Atom {i + 1}, {i+2}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = f'|{"$Ψ^{+}$"}⟩ Probability'

    elif type == 'psi minus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = f'|{"$Ψ^{-}$"}⟩ Probability'

    elif type == 'phi plus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = f'|{"$Φ^{+}$"}⟩ Probability'

    elif type == 'phi minus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = f'|{"$Φ^{-}$"}⟩ Probability'

    elif type == 'eigen energies':
        labels = [f'E{i}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = f'Eigenstate Probability'

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
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.9)
        cbar.set_label(bar_label)

def colormap_density_matrices(density_matrices, dt, times, num_of_plots=25, showtime=False):

    num_of_pairs = len(density_matrices)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=num_of_pairs, ncols=num_of_plots, figsize=(12, 3))

    steps = int(times[-1]/dt)

    times_range = np.array([2.42, 2.95])
    steps_range = times_range/dt
    diff = steps_range[1] - steps_range[0]

    plot_step = diff // (num_of_plots - 1)
    step = np.arange(steps_range[0], steps_range[1]+1, plot_step).astype(int)
    step[-1] = step[-1] - 1

    for j in range(num_of_pairs):
        density_matrices_pair = density_matrices[j]

        for i in range(0, num_of_plots):

            density_matrix = density_matrices_pair[step[i]]
            abs_matrix = np.abs(density_matrix)
            phase_matrix = np.abs(np.angle(density_matrix))

            print(np.angle(density_matrix))

            # Create a colormap (you can choose a different colormap if desired)
            cmap = plt.get_cmap('viridis')

            # Set the extent of the heatmap to match the dimensions of the matrix
            extent = [0, 4, 0, 4]

            # Display the rdm matrix as a heatmap
            ax = axes[1-j, i]

            ax.imshow(phase_matrix, cmap=cmap, extent=extent, alpha=abs_matrix,
                      vmin=0, vmax=np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')

            ax.set_xlabel(f'{round(times[step[i]], 2)}', fontsize=9)

            if i == 0:
                ax.set_ylabel(f'Atom {j+1}, {j+2}', rotation='horizontal', labelpad=40, fontsize=16, verticalalignment='center')


    plt.show()

def state_label(state):

    label = ''

    for num in state:
        if num == 1:
            label += 'r'

        elif num == 0:
            label += '0'

        else:
            pass

    label = f'|{label}⟩'

    return label

def ascending_binary_strings(n):
    result = ['0', '1']

    while len(result[0]) < n:
        result = [s + '0' for s in result] + [s + '1' for s in result]

    return [s.zfill(n) for s in result]

def energy_labels(n):
    labels = [f'E{i}' for i in range(n)]
    return labels


if __name__ == "__main__":
    pass









