import sys

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.colors as mcolors
import config.config as cf
import cmocean

import vector_manipluation_tools as vm
import data_analysis as da
import config.config as cf
from config.config import plotcolors

mpl.rcParams['font.size'] = 12
mpl.rcParams["text.latex.preamble"] = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
mpl.rcParams["text.usetex"] = True


def set_up_color_bar(n, data, times, ax, type='rydberg', color='viridis', colorbar=True, cb_ax=None, ctype=1):
    # Labels for the bars
    if type == 'rydberg':
        labels = [f'{i + 1}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = "Rydberg Probability"
        ax.set_ylabel('Atom Site')

    elif type == 'vne':
        labels = [f'Atom {i + 1}' for i in range(n)]
        color = 'inferno'
        norm = mcolors.Normalize(vmin=0, vmax=np.log(2))
        bar_label = "Von Neumann Entropy"

    elif type == 'psi plus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'|$\Psi^{+}$⟩Probability'

    elif type == 'psi minus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]

        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'|$\Psi^{-}$⟩Probability'

    elif type == 'phi plus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]
        color = 'viridis'
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'|$\Phi^{+}$⟩Probability'

    elif type == 'phi minus':
        n = n - 1
        labels = [f'Atom {i + 1}, {i + 2}' for i in range(n)]
        color = 'viridis'
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'|$\Phi^{-}$⟩ Probability'

    elif type == 'eigen energies':
        labels = [f'E{i}' for i in range(n)]
        color ='cmo.amp'
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'⟨$\Psi_{\lambda}$|$\Psi$⟩'

    elif type == 'correlation':
        labels = [f'g(1, {i+1})' for i in range(1, n+1)]
        color = 'PiYG'
        norm = mcolors.Normalize(vmin=-0.25, vmax=0.25)
        bar_label = r'label'

    elif type == 'correlation pairwise':
        labels = [f'g({i}, {i+1})' for i in range(1, n+1)]
        color = 'PiYG'
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = r'label'

    elif type == 'concurrence':
        labels = [f'C({i}, {i+1})' for i in range(1, n+1)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        color = 'inferno'
        bar_label = r'C(i, j)'

    elif type == 'concurrence 2':
        labels = [f'C({ctype}, {i+1})' for i in range(0, n+1)]
        labels.remove(f'C({ctype}, {ctype})')
        norm = mcolors.Normalize(vmin=0, vmax=1)
        color = 'inferno'
        bar_label = r'Concurrence'

    elif type == 'concurrence 3':
        labels = [f'C({i})' for i in reversed(range(1, n+1))]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        color = 'inferno'
        bar_label = r'Concurrence'

    elif type == 'pairwise purity':
        labels = [r'$\gamma$'+f'({i}, {i+1})' for i in range(1, n+1)]
        norm = mcolors.Normalize(vmin=0.25, vmax=1)
        color = 'inferno'
        bar_label = r'$\gamma(i, j)$'

    elif type == 'two atom blockade':
        labels = [f'' for i in range(n)]
        norm = mcolors.Normalize(vmin=0, vmax=1)
        bar_label = "Rydberg Probability"
        ax.set_ylabel('')


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
    ax.yaxis.set_tick_params(width=0)

    # Fill whole figure
    ax.set_xlim(0, times[-1])  # Set the x-axis limits
    ax.set_ylim(-0.5, n - 0.5)  # Set the y-axis limits
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # ax.tick_params(which='minor', size=4)

    if colorbar:
        if cb_ax is not None:
            cb_ax = cb_ax
            frac = 0.7

        else:
            cb_ax = ax
            frac = 0.1

        # Add a colorbar to show the mapping of values to colors
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # fake up the array
        cbar = plt.colorbar(sm, ax=cb_ax, orientation='vertical', shrink=0.9, fraction=frac)
        cbar.set_label(bar_label)


def end_colorbar_barchart(n, data, ax):

    atoms = np.arange(1, n+1, 1)

    data = np.array(data)

    data = data[:, -1]


    # Remove y-axis ticks
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_yticklabels([''])
    ax.set_xlim(0, 1)


    ax.barh(atoms, data)

def end_eigenenergies_barchart(n, data, ax):
    energies = np.arange(0, n, 1)
    labels = [f'E{i}' for i in range(n)]
    color = 'cmo.amp'

    data = np.array(data)

    data = data[:n, -1]

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(color)
    colors = [cmap(norm(value)) for value in data]

    # Remove y-axis ticks
    ax.tick_params(axis='y', which='both', left=False, right=False)
    ax.set_yticks(energies)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(data)+0.1)

    ax.barh(energies, data, color=colors)

def colormap_density_matrices(density_matrices, dt, times, num_of_plots=25, showtime=False):
    num_of_pairs = len(density_matrices)

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=num_of_pairs, ncols=num_of_plots, figsize=(12, 3))

    steps = int(times[-1] / dt)

    times_range = np.array([2.42, 2.95])
    steps_range = times_range / dt
    diff = steps_range[1] - steps_range[0]

    plot_step = diff // (num_of_plots - 1)
    step = np.arange(steps_range[0], steps_range[1] + 1, plot_step).astype(int)
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
            ax = axes[1 - j, i]

            ax.imshow(phase_matrix, cmap=cmap, extent=extent, alpha=abs_matrix,
                      vmin=0, vmax=np.pi)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('')

            ax.set_xlabel(f'{round(times[step[i]], 2)}', fontsize=9)

            if i == 0:
                ax.set_ylabel(f'Atom {j + 1}, {j + 2}', rotation='horizontal', labelpad=40, fontsize=16,
                              verticalalignment='center')

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


def plot_state_fidelities(q_states, states_to_test, times, steps, ax, colors_num=0, sum_probs=False):
    num_of_test_states = len(states_to_test)
    state_fidelities = [[] for _ in range(num_of_test_states)]

    for i in range(0, num_of_test_states):

        state_to_test = states_to_test[i]

        label = state_label(state_to_test)

        state_fidelities[i] = da.get_state_fidelities(q_states, state_to_test)

        cn = colors_num + i

        ax.plot(times, state_fidelities[i], label=f'{label}', color=plotcolors[cn])

    if sum_probs:

        state_fidelities = np.array(state_fidelities)
        sum_fidelities = np.linspace(0, 0, steps)

        for i in range(0, num_of_test_states):
            sum_fidelities = sum_fidelities + state_fidelities[i]

        ax.plot(times, sum_fidelities, label='Sum', alpha=0.65, color='grey')

    # Configure Plot
    ax.set_ylim(0, 1)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(which='minor', size=4)
    ax.spines['left'].set_position('zero')
    ax.set_ylabel('Probability')
    #ax.legend(loc='upper right')



def plot_eigenenergies(n, times, eigenvalues, ax, energy_range):
    eigenvalues = np.array(eigenvalues)
    labels = cf.energy_eigenvalue_labels(n)
    ax.set_ylabel('Energy')

    # Plot eigenvalues
    for i in energy_range:
        ax.plot(times, eigenvalues[:, i])

    #ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_eigenenergies_fidelities_line(n, times, eigenvalues, eigen_probs, expectation_energies, ax, energy_range, cb=True, cb_label='h', cb_ax=None):

    eigenvalues = np.array(eigenvalues)

    #ax.plot(times, expectation_energies)

    for i in reversed(energy_range):

        points = np.array([times, eigenvalues[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        print(segments)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='cmo.amp', norm=norm)

        # Set the values used for colormapping
        lc.set_array(eigen_probs[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)


    ax.set_xlim(0, times[-1])
    ax.set_ylim(min(eigenvalues[:, 0]) - 10, max(eigenvalues[:, energy_range[-1]] + 10))



    if cb:
        if cb_ax is not None:
            cb_ax = cb_ax
            frac = 0.7

        else:
            cb_ax = ax
            frac = 0.1

        plt.colorbar(line, ax=cb_ax, label=cb_label, orientation='vertical', shrink=0.9, fraction=frac)



def plot_eigenenergies_state_fidelities_line(n, times, eigenvalues, eigenvectors, state_to_test, ax, energy_range, reverse=True, cb_label='h', cb_ax=None):

    eigenvalues = np.array(eigenvalues)/(2*np.pi) #convert to frequency
    eigenstate_state_probs = [[] for _ in range(0, energy_range[-1] + 1)]

    if state_to_test == 'psi plus':
        v_state_to_test = (1/np.sqrt(2)) * (vm.initial_state([1, 0]) + vm.initial_state([0, 1]))
    elif state_to_test == 'psi minus':
        v_state_to_test = (1/np.sqrt(2)) * (vm.initial_state([1, 0]) - vm.initial_state([0, 1]))

    else:
        v_state_to_test = vm.initial_state(state_to_test)

    for eigenvector in eigenvectors:

        for i in range(0, energy_range[-1] + 1):
            v = eigenvector[:, i]
            v = v[:, np.newaxis]


            eigenstate_state_prob = da.state_prob(v_state_to_test, v)
            eigenstate_state_probs[i] += [eigenstate_state_prob]

    if reverse:
        plot_order = reversed(energy_range)

    else:
        plot_order = energy_range

    for i in plot_order:

        points = np.array([times, eigenvalues[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 1)
        lc = LineCollection(segments, cmap='cmo.amp', norm=norm)

        # Set the values used for colormapping
        lc.set_array(eigenstate_state_probs[i])
        lc.set_linewidth(2)
        line = ax.add_collection(lc)

    if cb_ax is not None:
        plt.colorbar(line, ax=cb_ax, label=cb_label, shrink=0.9, fraction=0.7)

    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(min(eigenvalues[:, 0]) - 10, max(eigenvalues[:, energy_range[-1]] + 10))

    ax.set_ylabel(r'$E$ / 2$\pi$ (MHz)')

def plot_domain_wall_density():
    pass










if __name__ == "__main__":
    pass
