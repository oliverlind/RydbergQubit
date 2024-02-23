import sys
import pandas as pd
from scipy.fft import fft, fftfreq
import numpy as np
import time
from scipy.linalg import expm

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import matplotlib.cm as cm

import data_analysis
import detuning_regimes
import rabi_regimes
from adiabatic_evolution import AdiabaticEvolution
from rydberg_hamiltonian_1d import RydbergHamiltonian1D
import data_analysis as da
import config.config as cf
from config.config import plotcolors
import ploting_tools
from matplotlib.colors import Normalize
import pandas as pd

mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams["text.latex.preamble"] = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
mpl.rcParams["text.usetex"] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['axes.linewidth'] = 1.0


class PlotSingle(AdiabaticEvolution):
    def __init__(self, n, t, dt, δ_start, δ_end, a=5.48, detuning_type=None, single_addressing_list=None,
                 initial_state_list=None, rabi_regime='constant'):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list,
                         rabi_regime=rabi_regime, a=a)

    '''ColourBar'''
    def colour_bar(self, type='rydberg', data=None, title=None, show=False, ax=None, cb=True, cb_ax=None, end_ax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        if data is None:
            if type == 'rydberg':
                data = self.time_evolve(rydberg_fidelity=True)
                n = self.n

            elif type == 'eigen energies':
                eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(expec_energy=True,
                                                                                                     eigen_list=True,
                                                                                                     eigenstate_fidelities=True,
                                                                                                     )
                data = eigenstate_probs
                n = self.dimension

            elif type == 'correlation':
                states = self.time_evolve(states_list=True)

                g_r_list = [[] for _ in range(self.steps)]

                for i in range(0, self.steps):
                    g_r = self.rydberg_rydberg_density_corr_function(states[i], i=1)
                    g_r_list[i] = g_r

                data = np.array(g_r_list)
                data = data.T

                n = self.n - 1

            else:
                sys.exit()


        else:
            if type == 'rydberg':
                n = self.n

            elif type == 'eigen energies':
                n = 10  # self.dimension

            elif type == 'correlation':
                n = self.n - 1

            else:
                sys.exit()


        ploting_tools.set_up_color_bar(n, data, self.times, ax=ax, type=type, colorbar=cb, cb_ax=cb_ax)

        if end_ax is not None:
            ploting_tools.end_colorbar_barchart(self.n, data, ax=end_ax)

        if show:
            plt.show()

        return data

    '''Rabi and Detunning shape'''
    def rabi_and_detuning_shape(self, ax=None, show=False):

        if ax is None:
            fig, ax1 = plt.subplots(figsize=(11, 6))
        else:
            ax1 = ax

        detuning_1 = detuning_regimes.linear_detuning_flat(self.δ_start, self.δ_end, self.steps)
        rabi = rabi_regimes.pulse_start(self.steps)

        ax1.set_xlabel('Time ($\mu$s)')
        ax1.set_ylabel(r'$\Delta$ (2$\pi$ x Mhz)', color='r')
        ax1.plot(self.times, detuning_1, color='r', label='Atom 1, 4')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.set_ylim(min(self.δ_start, self.δ_end) - 20, max(self.δ_start, self.δ_end) + 60)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel(r'$\Omega$ \ $\Omega_{0}$', color=color)  # we already handled the x-label with ax1
        ax2.plot(self.times, rabi, color=color)
        ax2.fill_between(self.times, rabi, 0, color='blue', alpha=.1)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.1)

        if show:
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()

    def detuning_shape(self, types, position=0.5, ax=None, show=False, save_pdf=False):

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Time ($\mu$s)')
        else:
            ax = ax

        for i, d_type in enumerate(types):

            if d_type == 'quench':
                color = 'b'
                d = detuning_regimes.linear_detuning_quench(self.δ_start, self.δ_end, self.steps)
                ax.plot(self.times, d / self.two_pi, color=color)
            elif d_type == 'linear flat':
                if i == 1:
                    color = 'r'
                    d = detuning_regimes.linear_detuning_flat(self.δ_start, self.δ_end, self.steps)
                    ax.plot(self.times, d / self.two_pi, color=color)
                else:
                    pass
            elif d_type == 'linear':
                color = 'r'
                d = np.linspace(self.δ_start, self.δ_end, self.steps)
                ax.plot(self.times, d / self.two_pi, color=color)

            elif type(d_type) == int:
                color = 'r'
                d = detuning_regimes.quench_ramped(δ_start, δ_end, self.steps, d_type, position=position)
                ax.plot(self.times, d / self.two_pi, color=color)

            else:
                pass

        ax.set_ylim(- 40, max(self.δ_start, self.δ_end) / self.two_pi + 10)  # min(self.δ_start, self.δ_end)/self.two_pi
        ax.set_ylabel(r'$\Delta$/$2\pi$ (MHz)')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.tight_layout()
            plt.show()


    '''State Fidelity Functions'''

    def state_fidelity(self, states_to_test, q_states=None, ax=None, sum_probs=False, show=False, colors_num=0):

        if ax is None:
            q_states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
            ax.set_xlabel(r'Time ($\mu$s)')
        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        ploting_tools.plot_state_fidelities(q_states, states_to_test, self.times, self.steps, ax=ax, sum_probs=sum_probs
                                            , colors_num=colors_num)

        if show:
            plt.show()

    def state_fidelity_ft(self, states_to_test, interval=None, colors_num=0, title=None, ax=None, q_states=None):

        if q_states is None:
            q_states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.title(f'Moving First Atom 10a per μs (Δ = {self.δ_end})', fontsize=18)
            plt.xlim(0, self.t)
            ax = plt.gca()
            ax.set_xlabel('Time (μs)')

        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        ax.set_ylabel('Probability')
        plt.ylim(0, 1)

        num_of_test_states = len(states_to_test)
        state_fidelities = [[] for _ in range(num_of_test_states)]

        if interval is None:
            interval = [0, self.t]

        for i in range(0, num_of_test_states):

            state_to_test = states_to_test[i]
            label = ploting_tools.state_label(state_to_test)

            v_state_to_test = self.initial_state(state_to_test)

            for j in range(0, self.steps):
                state_fidelity = data_analysis.state_prob(v_state_to_test, q_states[j])
                state_fidelities[i] += [state_fidelity]

            cn = colors_num + i

            ax.plot(self.times, state_fidelities[i], label=f'{label}', color=plotcolors[cn])

            fourier_transform = np.abs(fft(state_fidelities[i][200:]))  # Taken absolute value to get the magnitude of
            # each phase component


    '''Eigenenergies functions'''

    def eigenenergies_barchart(self, eigenvalues=None, eigenstate_probs=None,
                               expectation_energies=None, before=False, ax=None, save_pdf=False, show=False):

        if ax is None:
            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                                                 eigenstate_fidelities=True,
                                                                                                 expec_energy=True)
            fig, ax = plt.subplots(1, 1, figsize=(6, 2.2))

        ax.set_xlabel('Energy Eigenvalue')
        ax.set_ylabel(r'|⟨$\Psi_{\lambda}$|$\Psi(t>t_{quench})$⟩|$^{2}$')


        eigenstate_probs = np.array(eigenstate_probs)
        n = len(eigenvalues[-1])
        bound = np.max(eigenstate_probs[:, -1]) + 0.05
        initial_fidelity = eigenstate_probs[0, 0]
        print(initial_fidelity)


        if before:
            ax.bar(eigenvalues[0], [1] * n, color='grey', alpha=0.2, width=4)
            ax.bar(eigenvalues[0], eigenstate_probs[:, 0], width=4)

        else:
            ax.bar(eigenvalues[-1], [1]*n, color='grey', alpha=0.2, width=4)
            ax.bar(eigenvalues[-1], eigenstate_probs[:, -1], width=4)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.show()



    def eigenstate_fidelity_colorbar(self, show=False):
        eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(expec_energy=True,
                                                                                             eigen_list=True,
                                                                                             eigenstate_fidelities=True,
                                                                                             )
        fig, ax = plt.subplots()

        ploting_tools.set_up_color_bar(17, eigenstate_probs, self.times, ax=ax, type='eigen energies')

        if show:
            plt.show()

    def eigenenergies_lineplot(self, eigenvalues=None, detuning=False, ax=None):

        if eigenvalues is None:
            eigenvalues, eigenvectors = self.time_evolve(eigen_list=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            ax.set_xlabel(r'Time ($\mu$s)')

        if detuning:
            ploting_tools.plot_eigenenergies(self.n, self.detunning[0], eigenvalues, ax, range(0, self.dimension))

        else:
            ploting_tools.plot_eigenenergies(self.n, self.times, eigenvalues, ax, range(0, self.dimension))

    def eigenenergies_lineplot_with_eigenstate_fidelities(self, eigenvalues=None, eigenstate_probs=None,
                                                          expectation_energies=None, eigenstate_fidelities=None,
                                                          ax=None,
                                                          show=False, cb=True, cb_label=r'|$\Psi^{+}$⟩ Fidelity',
                                                          cb_ax=None):

        if ax is None:
            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                                                 eigenstate_fidelities=True,
                                                                                                 expec_energy=True)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlabel(r'Time ($\mu$s)')

        ploting_tools.plot_eigenenergies_fidelities_line(self.n, self.times, eigenvalues, eigenstate_probs,
                                                         expectation_energies, ax, range(0, self.dimension), cb=cb,
                                                         cb_label=r'⟨$\Psi_{\lambda}$|$\Psi$⟩', cb_ax=cb_ax)

        if show:
            plt.show()

    def eigenenergies_lineplot_with_state_fidelities(self, state_to_test, eigenvalues=None, eigenvectors=None,
                                                     detuning=False, ax=None,
                                                     cb_label=r'|$\Psi^{+}$⟩ Fidelity', save_pdf=False, reverse=True,
                                                     show=False, cb_ax=None):

        if ax is None:

            eigenvalues, eigenvectors = self.time_evolve(eigen_list=True)
            fig, ax = plt.subplots(figsize=(4, 3))

            if detuning:
                ax.set_xlabel(r'$\Delta$ (MHz)')
            else:
                ax.set_xlabel(r'Time ($\mu$s)')

        if detuning:
            ploting_tools.plot_eigenenergies_state_fidelities_line(self.n, self.detunning[0] / (2 * np.pi), eigenvalues,
                                                                   eigenvectors,
                                                                   state_to_test, ax, range(0, self.dimension),
                                                                   reverse=reverse,
                                                                   cb_label=cb_label, cb_ax=cb_ax)

        else:
            ploting_tools.plot_eigenenergies_state_fidelities_line(self.n, self.times, eigenvalues, eigenvectors,
                                                                   state_to_test, ax, range(0, self.dimension),
                                                                   reverse=reverse,
                                                                   cb_label=cb_label, cb_ax=cb_ax)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.tight_layout()
            plt.show()

    def eigenvalues_distance(self, save_pdf=False, show=False):
        fig, ax = plt.subplots(figsize=(8, 3.5))

        ax.set_xlabel(r'$a$ ($\mu$m)', fontsize=12)
        ax.set_ylabel(r'$E$ (MHz)')

        xmin = 4
        xmax = 13

        a_list = np.linspace(xmin, xmax, 1000)

        eigenvalues_list = []

        for a in a_list:
            h_m = RydbergHamiltonian1D(2, a=a).hamiltonian_matrix([0])

            eigenvalues, eigenvector = np.linalg.eigh(h_m)

            eigenvalues_list += [eigenvalues]

        labels = ['|00⟩', r'|$\Psi^{-}$⟩', r'|$\Psi^{+}$⟩', '|rr⟩']

        eigenvalues_list = np.array(eigenvalues_list)

        for i in reversed(range(0, 4)):
            ax.plot(a_list, eigenvalues_list[:, i] / self.two_pi, label=labels[i], linewidth=2, zorder=1)

        ax.set_xlim(min(a_list), max(a_list))

        ax.spines['right'].set_position(('data', max(a_list)))

        pale_blue = (0.7, 0.8, 1.0)

        plt.axhline(y=self.Rabi / self.two_pi, color='black', linestyle='--', linewidth=1, alpha=0.1)
        plt.axhline(y=-self.Rabi / self.two_pi, color='black', linestyle='--', linewidth=1, alpha=0.1)
        plt.axhspan(ymin=-self.Rabi / self.two_pi, ymax=self.Rabi / self.two_pi, color='grey', alpha=0.1)

        plt.axvline(x=self.r_b, color=pale_blue, linestyle='--', linewidth=2, alpha=0.8)

        print(self.r_b)

        plt.text(self.r_b + 0.35, 10, r'$R_{B}$', ha='center', va='center', rotation=-90, fontsize=12)

        # plt.text(5.25, 315, r'$|rr⟩$', ha='center', va='center', fontsize=18)  # Displaying 1/2

        plt.axvspan(xmin=xmin, xmax=self.r_b, color=pale_blue, alpha=0.2)

        plt.subplots_adjust(right=0.8)
        plt.legend(loc='upper right', fontsize=12)
        plt.ylim(-5, 15)

        # plt.arrow(13, -25+23.7, 0, -22.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)

        # Create an inset axes in the top right corner
        # axins = inset_axes(plt.gca(), width="45%", height="40%", loc='upper right')

        # Specify the region to zoom in (adjust these values accordingly)
        x1, x2, y1, y2 = 11.2, 15, -35, 40  # Define the zoomed-in region

        # # Set the limits for the inset axes
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        #
        # for i in reversed(range(0, self.dimension)):
        #     axins.plot(a_list, eigenvalues_list[:, i])
        #
        # axins.arrow(13.25, -25, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.arrow(13.25, 0.5, 0, -23.2, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.text(13.4, -13, r'$\Omega$', ha='center', va='center', fontsize=16)
        #
        # axins.arrow(13.5, 0, 0, 23.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.arrow(13.5, 23.7, 0, -22.7, head_width=0.1, head_length=2, ec='black', fc='black', linewidth=2, zorder=2)
        # axins.text(13.65, 13, r'$\Omega$', ha='center', va='center', fontsize=16)
        #
        # # Customize the appearance of tick labels in the inset axes
        # axins.tick_params(axis='both', labelsize=6)  # Adjust the labelsize as needed

        # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", alpha=0.7)
        # plt.draw()

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        if show:
            plt.tight_layout()
            plt.show()
            
    '''Expectation Energy'''
    
    def energy_std(self, save_pdf=False, show=False):
        
        expectation_energies, std_energies_list = self.time_evolve(expec_energy=True, std_energy_list=True)
        
        energy_spread_percentage = data_analysis.energy_spread(expectation_energies, std_energies_list)

        if show:
            plt.plot(self.times, energy_spread_percentage)

            plt.show()

    '''Quantum Correlation and entanglement functions'''

    def quantum_mutual_information(self, i, j, states=None, ax=None, show=False, data=False, purity=False, label=''):

        if ax is None:
            states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        qmi_list = []
        purity_list = []


        for state in states:
            rdm_i = self.reduced_density_matrix(state, i)
            rdm_j = self.reduced_density_matrix(state, j)
            rdm_ij = self.reduced_density_matrix_pair(state, i, j)

            qmi = da.q_mutual_info(rdm_i, rdm_j, rdm_ij)
            qmi_list += [qmi]

            if purity:
                p = np.trace(np.dot(rdm_ij, rdm_ij))
                purity_list += [p]

        ax.plot(self.times, qmi_list, label=label)
        #ax.set_ylim(0, 0.75)
        print(f'Atom {j} QMI_av', np.average(qmi_list))

        if purity:
            print(f'Atom {j} Purity_av', np.average(purity_list))

            ax.plot(self.times, purity_list)

        if show:
            plt.ylabel('Quantum Relative Entropy')
            plt.xlabel('Time (s)')
            plt.show()

        if data:
            return qmi_list

    def plot_half_sys_entanglement_entropy(self, ax=None, atom_i=None, entanglement_entropy=None, states=None, label='', save_pdf=False, show=False):

        if ax is None:
            states, entanglement_entropy = self.time_evolve(states_list=True, entanglement_entropy=True)

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_xlabel(r'Time ($\mu$s)')

            ax.plot(self.times, entanglement_entropy, color='blue', label='Half Chain')

            ax.set_ylabel('Half Chain VNE')

        else:
            ax.plot(self.times, entanglement_entropy, label=label)

        if atom_i is not None:
            vne_list = []
            for i in range(0, self.steps):
                rdm = self.reduced_density_matrix(states[i], atom_i)
                vne = da.von_nuemann_entropy(rdm)
                vne_list += [vne]

            ax.plot(self.times, vne_list, color='orange', label='Atom 1')

        ax.legend(loc='upper left')

        plt.axhline(y=np.log(4), color='blue', linestyle='--', linewidth=1, alpha=0.5)

        plt.axhline(y=np.log(2), color='orange', linestyle='--', linewidth=1, alpha=0.5)

        if show:
            plt.show()

    def plot_entanglement_entropy_single_atom(self, j, states=None, ax=None, show=False, data=False, label=''):

        if ax is None:
            states = self.time_evolve(states_list=True)
            fig = plt.figure(figsize=(10, 5))
            plt.xlim(0, self.t)
            ax = plt.gca()
        elif ax is not None:
            ax = ax
        else:
            sys.exit()

        vne_list = []

        for i in range(0, self.steps):
            rdm = self.reduced_density_matrix(states[i], j)
            vne = da.von_nuemann_entropy(rdm)
            vne_list += [vne]


        ax.plot(self.times, vne_list, label=label)
        ax.set_ylim((0,0.8))
        ax.axhline(y=np.log(2), color='grey', linestyle='--', linewidth=1, alpha=0.5)
        print(f'Atom {j} VNE_av', np.average(vne_list))


        if show:
            plt.ylabel('VNE')
            plt.xlabel('Time (s)')
            plt.show()

        if data:
            return vne_list

    def correlation(self, ax=None, states=None, corr_lengths=False, save_pdf=False, show=False):

        if ax is None:
            states = self.time_evolve(states_list=True)

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.set_xlabel(r'r')

        corr_lengths_list = []
        x = np.arange(0,self.t, 50*self.dt)

        for i in np.arange(0,self.steps,50):

            g_r = self.rydberg_rydberg_density_corr_function(states[i])

            corr_length = data_analysis.correlation_length(self.n, g_r)

            corr_lengths_list +=[corr_length]

            print(corr_length)

        r = np.arange(1, self.n, 1)

        ax.plot(x, corr_lengths_list)

        plt.show()

    '''Thermalization'''

    def thermalization_matrix_colour_maps(self, d):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

        h_m = self.hamiltonian_matrix([d])

        t_m, e = data_analysis.thermalization_matrix(h_m, eigenvectors_table=True)

        print(h_m)

        print(t_m)

        # Create a colormap (you can choose a different colormap if desired)
        cmap = plt.get_cmap('RdYlGn')

        # Set the extent of the heatmap to match the dimensions of the matrix
        extent = [0, self.dimension, 0, self.dimension]

        ax.imshow(t_m, cmap=cmap, extent=extent, vmin=-self.n, vmax=self.n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('')

        plt.show()






if __name__ == "__main__":
    t = 0.1
    dt = 0.01
    n = 7
    δ_start = 30 * 2 * np.pi
    δ_end = 30 * 2 * np.pi

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['quench'] * 3
    three2 = ['quench'] + ['linear flat'] * 2
    three3 = ['linear flat'] * 3
    three4 = [1] + ['linear flat'] * 2

    four = ['quench'] * 4

    five = ['linear'] * 5
    five1 = ['linear flat'] * 5
    five2 = [5] * 5
    five3 = [1] + ['linear flat'] * 4
    five4 = [[1, 25]] + ['linear flat'] * 4

    seven = ['linear'] * 7
    seven2 = [5] * 7
    seven3 = [10] + ['linear flat'] * 6


    nine = ['linear'] * 9

    plotter = PlotSingle(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=seven2,
                         initial_state_list=[1,0,1,0,1,0,1],
                         a=5.48
                         )

    #plotter.detuning_shape(types=seven2, show=True, save_pdf=True, position=0.125)

    plotter.eigenenergies_barchart(show=True, save_pdf=True)
    #plotter.eigenstate_fidelity_colorbar(show=True)
    plotter.colour_bar(show=True)

    plotter.thermalization_matrix_colour_maps(0)

    plotter.plot_half_sys_entanglement_entropy(atom_i=None, show=True)

    plotter.plot_entanglement_entropy_single_atom(3, show=True)

    #plotter.eigenenergies_lineplot_with_eigenstate_fidelities(show=True)

    plotter.energy_std()

    #plotter.correlation(show=True)

    data = plotter.colour_bar(type='correlation', show=True)

    print(np.average(data[0][10:]),np.average(data[1][10:]))

    #plotter.eigenvalues_distance(show=True, save_pdf=True)






    plotter.state_fidelity([[]])

    plotter.eigenenergies_lineplot(detuning=True)
