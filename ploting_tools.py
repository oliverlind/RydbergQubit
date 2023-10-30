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

    elif type == 'bell':
        n = n-1
        labels = [f'Atom {i + 1}, {i+2}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=0.6)
        bar_label = f'|{"$Ψ^{+}$"}⟩ Probability'

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


