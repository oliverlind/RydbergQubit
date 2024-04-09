import sys
import pandas as pd
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import numpy as np
import time
from scipy.linalg import expm
import math

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.collections import LineCollection
from matplotlib.collections import PathCollection
from matplotlib.path import Path
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle


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
from plot_single import PlotSingle
from matplotlib.animation import FFMpegWriter



mpl.rcParams['animation.ffmpeg_path'] = '/opt/homebrew/bin/ffmpeg'
mpl.rcParams['font.size'] = 11
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


class CombinedPlots(PlotSingle):

    def __init__(self, n, t, dt, δ_start, δ_end, detuning_type=None, single_addressing_list=None,
                 initial_state_list=None, rabi_regime='constant'):
        super().__init__(n, t, dt, δ_start=δ_start, δ_end=δ_end, detuning_type=detuning_type,
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list,
                         rabi_regime=rabi_regime)

    '''Rydberg Colorbars'''

    def sweep_colourbar(self, types, type='rydberg', title=None, save_pdf=False):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 2.2),
                                gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [1.7, 1]})

        self.colour_bar(type=type, ax=axs[0, 0], cb_ax=axs[:, 1])
        self.detuning_shape(types, ax=axs[1, 0])

        # axs[1, 0].legend(loc='upper right')
        axs[1, 0].set_xlabel('Time ($\mu$s)')

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def ordered_state_colourbars(self, a_list, initial_state_list, single_addressing_list, save_pdf=False):

        fig, axs = plt.subplots(len(a_list)+1, 3, figsize=(8, 5), sharex='col',
                                gridspec_kw={'width_ratios': [6, 2, 1], 'height_ratios': [0.5, 1, 1]})


        sweep = self.detunning[0]/self.two_pi
        axs[0,0].plot(self.times, sweep)
        axs[0,0].set_ylabel(r'$\Delta$/2$\pi$ (MHz)')
        axs[0,0].set_ylim(-35,35)


        for i in range(1, len(a_list)+1):
            print(a_list[i-1])

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, a=a_list[i-1], detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state_list, rabi_regime='pulse start')

            if i == 1:
                cb_ax = axs[1:, 2]
                cb = True
            else:
                cb_ax = None
                cb = False

            singleplot.colour_bar(ax=axs[i, 0], cb=cb, end_ax=axs[i, 1], cb_ax=cb_ax)

        axs[-1, 0].set_xlabel('Time ($\mu$s)')
        axs[-1, 1].set_xlabel(r'⟨$n_{i}$⟩')

        axs[1, 0].set_ylabel(r'$Z_{2}$ ($a$ = 5.48$\mu$m)'+'\n'+ ''+ '\n'+'Atom site')
        axs[2, 0].set_ylabel(r'$Z_{3}$ ($a$ = 3.16$\mu$m)'+'\n'+ ''+ '\n'+'Atom site')

        for ax in axs[:, 2]:
            ax.axis('off')

        axs[0,1].axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def colorbars_vs_detunings(self, detunings,single_addressing_list, initial_state, cm_type='rydberg', save_pdf=False):

        fig, axs = plt.subplots(2, 3, figsize=(8, 3.5), sharex='col',
                                gridspec_kw={'width_ratios': [5, 5, 1], 'height_ratios': [9/8, 1]})

        for i, detuning in enumerate(detunings):
            print(detuning / self.two_pi)

            singleplot = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')
            #
            if i == 0:
                cb_ax = axs[:, 2]
                cb = True
            #
            # # elif i == 3:
            # #     cb_ax = axs[1, 2]
            # #     cb = True
            #
            else:
                cb_ax = None
                cb = False



            if i == 0:
                ax = axs[0, 0]
                cm_type = 'rydberg'

            if i == 1:
                ax = axs[0, 1]
                cm_type = 'rydberg'


            if i == 2:
                ax = axs[1, 0]
                # cm_type = 'concurrence'

            if i == 3:
                ax = axs[1, 1]
                # cm_type = 'concurrence'

            singleplot.colour_bar(ax=ax, cb=cb, cb_ax=cb_ax, type=cm_type)
            ax.tick_params(length=3)

            # if i == 2:
            #     ax.set_title(r'$\Delta_{int}$/2$\pi$ = $V_{i,i+1}$ =' + f' 31.9 (MHz)', fontsize=12)
            # else:
            #ax.set_title(r'$\Delta_{int}$/2$\pi$ ='+f' {round(detuning/self.two_pi,1)} (MHz)', fontsize=12)

            ax.axvline(x=0.3, color='r', linestyle='--', linewidth=1, alpha=0.6)
            #ax.get_yticklabels()[0].set_color('red')

            #ax.text(0.0005, 7, r'$\Delta$/2$\pi$ ='+f'{round(detuning/self.two_pi,1)}(MHz)', color='r')

        for ax in axs[:, 2]:
            ax.axis('off')


        axs[0, 1].set_ylabel('')
        axs[0, 1].set_yticklabels([])

        axs[1, 1].set_ylabel('')
        axs[1, 1].set_yticklabels([])

        axs[1, 0].set_xlabel(r'Time ($\mu$s)')
        axs[1, 1].set_xlabel(r'Time ($\mu$s)')
        plt.subplots_adjust(hspace=0.3)

        axs[0, 0].set_title(r'$\Delta_{t=0}$/2$\pi$ =' + f' 27.0 (MHz)', fontsize=12)
        axs[0, 1].set_title(r'$\Delta_{t=0}$/2$\pi$ =' + f' 30.0 (MHz)', fontsize=12)
        axs[1, 0].set_title(r'$\Delta_{t=0}$/2$\pi$ = $V_{NN}$/2$\pi$ =' + f' 31.9 (MHz)', fontsize=12)
        axs[1, 1].set_title(r'$\Delta_{t=0}$/2$\pi$ =' + f' 34.0 (MHz)', fontsize=12)



        if save_pdf:
            plt.savefig(f'Quick Save Plots/cbs.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()


    '''State Fidelity'''

    def colorbar_state_fidelity(self, states_to_test, type='rydberg', save_pdf=False):

        if type == 'rydberg':
            rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)
        else:
            sys.exit()

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 2.65),
                                gridspec_kw={'width_ratios': [17, 1], 'height_ratios': [1, 1.7]})

        # Sweep
        sweep = self.detunning[0] / self.two_pi
        print(np.shape(sweep))
        print(np.shape(self.times))
        sweep2 = self.detunning[1] / self.two_pi
        quench = 0.1*self.t

        axs[0, 0].set_yticks([24, 0])
        axs[0, 0].plot(self.times, sweep, color='b')
        axs[0, 0].plot(self.times, sweep2, color='g')
        axs[0, 0].set_ylabel(r'$\Delta_{i}$/2$\pi$ (MHz)', fontsize=10.5)


        axs[0, 0].set_ylim(-5, 37)
        # #axs[0, 0].axvspan(xmin=0, xmax=quench, color='green', alpha=0.1)
        axs[0, 0].axvspan(xmin=quench, xmax=self.t, color='red', alpha=0.1)
        axs[0, 0].text(2.55, 4, 'Atom 1', color='b')
        axs[0, 0].text(2.55, 28, 'Atom 2-9', color='g')
        axs[0, 0].text(quench + 0.03, 28, 'Quench', color='red')
        axs[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.2)
        axs[0,0].tick_params(top=False)

        axs[0,0].set_title(r'$\Delta_{t=0}$/2$\pi$ =' + f' 24.0 (MHz)', fontsize=12, pad=7)

        for ax in axs[:,0]:
            ax.tick_params(length=3)



        self.colour_bar(data=rydberg_fidelity_data, type=type, ax=axs[1, 0], cb_ax=axs[:, 1])
        axs[1, 0].axvline(x=quench, color='red', linestyle='--')
        axs[1,0].get_yticklabels()[0].set_color('red')

        # State Fidelities
        #self.state_fidelity(states_to_test, q_states=states, ax=axs[2, 0])
        #
        # axs[2, 0].set_ylabel(r'|⟨$Zeros$|$\Psi$⟩|$^2$')
        # axs[2, 0].set_xlabel(r'Time ($\mu$s)')
        # axs[2, 0].set_ylim(0, 1.1)
        # plt.xlim(0, 4.0)

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    '''Eigenstates'''

    def eigenspectrums(self, detunings, initial_state, single_addressing_list, save_pdf=False):

        fig, axs = plt.subplots(2, 1, figsize=(8, 3), sharex='col',
                                gridspec_kw={'height_ratios': [1,1]})

        for i, detuning in enumerate(detunings):
            print(detuning / self.two_pi)

            singleplot = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state[i], rabi_regime='constant')

            singleplot.eigenenergies_barchart(save_pdf=True, inset=True,ax=axs[i])
            axs[0].set_xlabel('')

            if i == 1:
                axs[i].set_ylabel(r'|⟨$\Psi_{\lambda}$|$\Psi(t)$⟩|$^{2}$')
                axs[i].set_xlabel(r'Energy Eigenvalue/$h$ (MHz)')

            rectangle = Rectangle((-11, 0.8), 5, 0.8, color='white', fill=True)
            axs[i].add_patch(rectangle)

            labels = ['(a)', '(b)']
            axs[i].text(-16.5, 0.84, labels[i], color='r', fontsize=16)

            # Multiple D

            # singleplot.eigenenergies_barchart(save_pdf=True, ax=axs[i])
            # axs[0].set_xlabel('')

            # if i == 2:
            #     axs[i].text(9, 0.42, r'$\Delta_{int}$/2$\pi$ = $V_{i,i+1}$ =' + f' {round(detuning / self.two_pi, 1)} (MHz)',
            #                 color='r', fontsize=12)
            # else:
            #     axs[i].text(31, 0.44, r'$\Delta_{int}$/2$\pi$ =' + f' {round(detuning / self.two_pi, 1)} (MHz)',
            #                 color='r', fontsize=12)
            #
            # rectangle = Rectangle((28, 0.38), 60, 0.8, color='white', fill=True)
            # axs[i].add_patch(rectangle)
            # axs[i].tick_params(right=False)
            # axs[i].tick_params(top=False)
            #
            # if i == 3:
            #     axs[i].set_ylabel(r'|⟨$\Psi_{\lambda}$|$\Psi(t)$⟩|$^{2}$')
            #     axs[i].set_xlabel(r'Energy Eigenvalue/$h$ (MHz)')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/eigenspectrums.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def eigenstate_fidelities(self, save_pdf=False):

        eigenvalues, eigenvectors, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                       eigenstate_fidelities=True)

        fig, axs = plt.subplots(1, 3, figsize=(8, 2.7),
                                gridspec_kw={'width_ratios': [5, 1.5, 0.2]})

        self.eigenenergies_lineplot_with_eigenstate_fidelities(eigenvalues=eigenvalues,
                                                               eigenstate_probs=eigenstate_probs, ax=axs[0], cb=True,
                                                               cb_ax=axs[2])
        # self.colour_bar(data=eigenstate_probs, ax=axs[1], type='eigen energies', cb_ax=axs[2])
        # ploting_tools.end_eigenenergies_barchart(12, eigenstate_probs, axs[1])

        self.eigenenergies_lineplot_with_eigenstate_fidelities(eigenvalues=eigenvalues,
                                                               eigenstate_probs=eigenstate_probs, ax=axs[1], cb=False)

        # axs[0].set_ylim(-400, 600)
        # axs[1].set_ylim(-30, 420)
        # axs[1].set_xlim(4.1, 5)

        axs[0].tick_params(axis='y', which='both', right=False)
        axs[1].tick_params(axis='y', which='both', right=False)
        axs[1].tick_params(axis='x', which='both', top=False, bottom=False)
        axs[1].set_xticklabels([''])
        axs[1].set_xlabel(r'⟨$\Psi_{\lambda}$|$\Psi(t>t_{quench})$⟩', labelpad=12)
        axs[0].set_ylabel(r'$E_{\lambda}$ (MHz)')

        axs[2].axis('off')

        plt.subplots_adjust(wspace=0.3)

        # Set x axis label
        axs[0].set_xlabel('Time ($\mu$s)')
        # axs[0].set_xlabel(r'⟨$\Psi_{\lambda}$|$\Psi$⟩')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def two_atom_eigenstates(self, save_pdf=False):
        eigenvalues, eigenvectors, eigenstate_probs = self.time_evolve(eigen_list=True,
                                                                       eigenstate_fidelities=True)

        fig, axs = plt.subplots(2, 3, figsize=(8, 4.5), sharex=True, sharey=True,
                                gridspec_kw={'width_ratios': [1, 1, 0.2]})

        labels = [r'|00⟩ Fidelity', r'|$\Psi^{-}$⟩ Fidelity', r'|$\Psi^{+}$⟩ Fidelity', r'|rr⟩ Fidelity']
        states_to_test = [[0, 0], 'psi minus', 'psi plus', [1, 1]]

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[0], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[0, 0],
                                                          cb_label=None)
        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[1], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[0, 1],
                                                          cb_label=None)

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[2], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[1, 0],
                                                          cb_label=None, reverse=False)

        self.eigenenergies_lineplot_with_state_fidelities(states_to_test[3], eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors, detuning=True, ax=axs[1, 1],
                                                          cb_label='State Fidelity', cb_ax=axs[:, 2])

        axs[0, 1].set_ylabel('')
        axs[1, 1].set_ylabel('')

        axs[1, 0].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')
        axs[1, 1].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')

        plt.ylim(-60, 55)

        for ax in axs[:, 2]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def multiple_eigenenergies_vs_detuning(self, qsteps_list, initial_detunings, save_pdf=False):

        fig, axs = plt.subplots(len(initial_detunings), 1, sharex=True,figsize=(8, 2.2))

        initial_state = [1 if i % 2 == 0 else 0 for i in range(self.n)]

        for i, int_detuning in enumerate(initial_detunings):
            qstep = qsteps_list[i]
            single_addressing_list = [qstep] + ['linear flat'] * (self.n-1)


            singleplot = PlotSingle(n, self.t, self.dt, int_detuning, int_detuning, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            eigenvalues, eigenvectors, expectation_energies, eigenstate_probs = singleplot.time_evolve(eigen_list=True,
                                                                                                 eigenstate_fidelities=True,
                                                                                                 expec_energy=True)

            singleplot.eigenenergies_barchart(eigenvalues=eigenvalues, eigenstate_probs=eigenstate_probs, expectation_energies=expectation_energies, ax=axs[i])

        plt.show()



    '''Energy_spread'''

    def energy_spread(self, n_list, qsteps_list,  save_pdf=False):

        plt.figure(figsize=(4, 6))

        for n in n_list:

            energy_spread_plot_list = []
            initial_state = [1 if i % 2 == 0 else 0 for i in range(n)]
            single_addressing_list = ['linear flat'] * n

            for j, q_step in enumerate(qsteps_list):
                single_addressing_list[0] = int(q_step)

                print(single_addressing_list)

                singleplot = PlotSingle(n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                        single_addressing_list=single_addressing_list,
                                        initial_state_list=initial_state, rabi_regime='constant')

                expectation_vals, std_vals = singleplot.time_evolve(expec_energy=True, std_energy_list=True)

                energy_spread_frac = data_analysis.energy_spread(expectation_vals, std_vals)

                energy_spread_plot_list += [energy_spread_frac[-1]] # add last term as energy and spread are constant after quench (H is constant)


            plt.scatter(qsteps_list * self.dt, energy_spread_plot_list, label=f'N={n}')
            plt.plot(qsteps_list * self.dt, energy_spread_plot_list, marker='o')

        plt.xlabel(r'$\Delta$$t_{quench}$ ($\mu$s)')
        plt.ylabel(r'$\sigma_{E} / \langle E \rangle$')
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    '''Entanglement Entropy and Correlation'''

    def quantum_mutual_informations(self, save_pdf=False):

        fig, axs = plt.subplots(self.n - 1, 1, sharex='col', figsize=(4, 4))

        states = self.time_evolve(states_list=True)

        for j in range(2, self.n + 1):

            self.quantum_mutual_information(1, j, states=states, ax=axs[self.n - j])
            axs[self.n - j].set_ylim(0, 1.1)

            if j != 2:
                axs[self.n - j].set_ylabel(f'I(1,{j})', rotation=0, labelpad=10, ha='right')
                # axs[self.n-j].tick_params(axis='y', which='both', left=False, right=False)
                # axs[self.n-j].set_yticklabels(['   '])

            else:
                axs[self.n - j].set_ylabel(f'I(1,{j})', rotation=0, labelpad=10, ha='right')
                axs[self.n - j].set_xlabel(r'Time after quench ($\mu$s)')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def area_law_test_animation(self, initial_states, single_addressing_list, times_list=None, save=False):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, np.log(2**(self.n/2)))
        ax.set_ylabel(r'$S_{EE}$')
        ax.set_xlabel('System A size')
        time_text = ax.text(0.8, 0.9, '', transform=ax.transAxes)


        if times_list is not None:
            steps = np.array(times_list) / self.dt
            steps = [int(item) for item in steps]

        states_combined = [[] for _ in range(len(initial_states))]

        for i, initial_state in enumerate(initial_states):

            singleplot = PlotSingle(n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            states = singleplot.time_evolve(states_list=True)

            states_combined[i] = states


        sites_list = np.arange(0, self.n + 1, 1)
        vne_list, = ax.plot([], [], 'ro-', label='|r0r0r0r⟩')  # Initialize an empty plot
        vne_list2, = ax.plot([], [], 'bo-', label=f'|{"0"*self.n}⟩')

        if len(initial_states) > 1:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
            plt.subplots_adjust(top=0.9)
        def init():
            vne_list.set_data([], [])
            return vne_list, vne_list2, time_text

        def update(step):

            for i, states in enumerate(states_combined):
                vne_values = [0]  # Initialize vne_list with the starting value
                for n_a in range(1, self.n):
                    rdm = self.reduced_density_matrix_from_left(self.n, n_a, states[step][:][:])
                    vne = data_analysis.von_nuemann_entropy(rdm)
                    vne_values.append(vne)
                vne_values.append(0)  # Append the ending value

                if i == 0:
                    vne_list.set_data(sites_list, vne_values)
                else:
                    vne_list2.set_data(sites_list, vne_values)

            time_text.set_text(f't = {round(step*self.dt, 2)}'+r'$\mu$s')

            return vne_list, vne_list2, time_text

        ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, repeat=False)

        plt.show()

        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        if save:
            path = 'Output Videos/'
            ani.save(path +'vne_animation.mp4', writer=writer)



    def area_law_test(self, initial_states, single_addressing_list, detuning, times_list, ax=None, save_pdf=False):

        if ax is None:
            fig, axs = plt.subplots(1, len(times_list), sharey=True, figsize=(4, 2.5))

        else:
            pass

        for ax in axs:
            ax.set_xlim(0, self.n)
            ax.set_ylim(0, 1.5)
            ax.set_xlabel(r'$N_{A}$')
            ax.set_xticks(np.arange(0,self.n+1,1))
            ax.tick_params(top=False)




        axs[0].set_ylabel(r'$S_{EE}(\rho_A)$')
        axs[0].set_title('t = 0.40 '+r'$\mu$s')
        axs[1].set_title('t = 0.80 ' + r'$\mu$s')
        axs[0].text(0.2, 1.32, '(b.i)', color='r', fontsize=16)
        axs[1].text(0.2, 1.32, '(b.ii)', color='r', fontsize=16)

        axs[0].text(2.1, 0.33, 'Area Law', color='tab:blue', fontsize=11)
        axs[1].text(1.65, 0.33, 'Volume Law', color='tab:blue', fontsize=11)

        steps = np.array(times_list) / self.dt
        steps = [int(item) for item in steps]


        for k, initial_state in enumerate(initial_states):

            singleplot = PlotSingle(n, self.t, self.dt, detuning[k], detuning[k], detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            states = singleplot.time_evolve(states_list=True)

            for i, step in enumerate(steps):
                vne_list = [0]*(self.n+1)
                sites_list = np.arange(0, self.n+1, 1)

                for n_a in range(1, self.n):

                    rdm = self.reduced_density_matrix_from_left(self.n, n_a, states[step])
                    vne = data_analysis.von_nuemann_entropy(rdm)

                    vne_list[n_a] = vne

                axs[i].scatter(sites_list, vne_list)
                axs[i].plot(sites_list, vne_list)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/area_law.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()



    def entanglement_propagation_barchart_animation(self, times_list, save=False):

        steps = np.array(times_list) / self.dt
        steps = [int(item) for item in steps]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.set_ylim(0, np.log(2.5))
        ax.set_ylabel(r'$S_{EE}$')
        ax.set_xlabel('Atom site')

        time_text = ax.text(0.8, 0.9, '', transform=ax.transAxes)


        states = self.time_evolve(states_list=True)

        vne_list = [0]*self.n

        bars = ax.bar(range(1, self.n + 1), vne_list, align='center', width=0.9)

        def init():
            for bar in bars:
                bar.set_height(0)
            return vne_list, time_text

        def update(step):
            for j in range(1, self.n + 1):
                rdm = self.reduced_density_matrix(states[step], j)
                vne = da.von_nuemann_entropy(rdm)
                vne_list[j - 1] = vne

            for bar, new_height in zip(bars, vne_list):  # Example: replace np.random.rand(self.n) with actual VNE values
                bar.set_height(new_height)

            time_text.set_text(f't = {round(step*self.dt, 2)}'+r'$\mu$s')


            return vne_list, time_text

        ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, repeat=False)

        plt.show()

        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        if save:
            path = 'Output Videos/'
            ani.save(path + 'bar_vne_animation.mp4', writer=writer)

    def rydberg_fidelity_barchart_animation(self, times_list, detunning=False, save=False):

        steps = np.array(times_list) / self.dt
        steps = [int(item) for item in steps]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_ylim(0, 1)
        ax.set_ylabel(r'Rydberg Fidelity')
        ax.set_xlabel('Atom Site')
        time_text = ax.text(0.51, 1.05, '', transform=ax.transAxes)
        d_text = ax.text(0.7, 1.05, '', transform=ax.transAxes)
        plt.subplots_adjust(top=0.9)

        rydberg_fidelity_list = self.time_evolve(rydberg_fidelity=True)
        rydberg_fidelity_list = np.array(rydberg_fidelity_list)
        rydberg_fidelity_list =rydberg_fidelity_list.T
        rydberg_fidelitys = [0]*self.n

        bars = ax.bar(range(1, self.n + 1), rydberg_fidelitys, align='center', width=0.9)

        def init():
            for bar in bars:
                bar.set_height(0)
            return rydberg_fidelitys, time_text, d_text

        def update(step):
            rydberg_fidelitys = rydberg_fidelity_list[step]

            for bar, new_height in zip(bars, rydberg_fidelitys):
                bar.set_height(new_height)

            time_text.set_text(f't = {round(step*self.dt, 2)}'+r'$\mu$s')

            if detunning:
                d_text.set_text(r'$\Delta_{1}$/2$\pi$' + f' = {round(self.detunning[0][step] / self.two_pi, 2)}' + '(MHz)')

            return rydberg_fidelitys, time_text, d_text

        ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, repeat=False)

        plt.show()

        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)

        if save:
            path = 'Output Videos/'
            ani.save(path + 'ryd_bar_vne_animation.mp4', writer=writer)




    def qmi_compare(self, n, atom_list, q_list, corr_type='QMI', save_pdf=False, save_df=False):

        data = pd.DataFrame()
        data['Time'] = self.times


        fig, axs = plt.subplots(len(atom_list), 1, figsize=(4, 3.5))
        plt.subplots_adjust(hspace=0)
        t_diff_data = pd.DataFrame()


        # for k, atom in enumerate(atom_list):

        initial_state = [1 if i % 2 == 0 else 0 for i in range(n)]

        background = ['linear flat'] * n
        single_addressing_lists = [background]

        for q in q_list:
            single_quench = [q] + ['linear flat'] * (n - 1)
            single_addressing_lists += [single_quench]

        for i, single_addressing_list in enumerate(single_addressing_lists):
            print(single_addressing_list)


            singleplot = PlotSingle(n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            states = singleplot.time_evolve(states_list=True)

            # if i > 0:
            #     speed = q_list[i - 1] * self.dt
            #     speed = r'$\Delta$$t_{q}$' + f'= {speed}' + '($\mu$s)'
            #
            # else:
            speed = ''

            for k, atom in enumerate(atom_list):

                if corr_type == 'QMI':

                    data[f'I(1,{atom}) t_q = {single_addressing_list[0]}'] = singleplot.quantum_mutual_information(1, atom,
                                                                                                              states=states,
                                                                                                              ax=axs,
                                                                                                              label=speed,
                                                                                                              data=True,
                                                                                                              purity=False)

                    axs.set_ylabel(f'I(1, {atom})')
                    axs.legend()

                elif corr_type == 'VNE':
                     data[f'VNE atom{atom} t_q = {single_addressing_list[0]}'] = singleplot.plot_entanglement_entropy_single_atom(atom, states=states, ax=axs[k], label=speed, data=True)


                     if i > 0:
                         bg_av = np.average(data[f'VNE atom{atom} t_q = linear flat'])
                         bg_sd = np.std(data[f'VNE atom{atom} t_q = linear flat'])
                         criteria = bg_av + (3 * bg_sd)
                         #axs[k].axhline(y=criteria, color='blue', linestyle='--', linewidth=1, alpha=0.5)

                         print(criteria)
                         start = data[data[f'VNE atom{atom} t_q = {single_addressing_list[0]}'] > criteria]
                         start2 = data[data[f'VNE atom{atom} t_q = {single_addressing_list[0]}'] > 1.1*data[f'VNE atom{atom} t_q = linear flat']]
                         start.reset_index(drop=True, inplace=True)
                         start2.reset_index(drop=True, inplace=True)

                         if not start.empty:
                            start = start['Time'][0]
                            print(start)
                         else:
                            start = 0

                         if not start2.empty:
                            start2 = start2['Time'][0]
                            print(start2)
                         else:
                            start2 = 0

                         t_diff_data[f'Atom {atom}'] = [start, start2]
                         axs[k].axvline(x=start, color='red', linestyle='--', linewidth=1, alpha=0.5)
                         axs[k].axvline(x=start2, color='green', linestyle='--', linewidth=1, alpha=0.5)

                     axs[k].set_ylabel(r'$S_{EE}$($\rho_{'+f'{atom}'+'}$)', rotation=0, labelpad=10, ha='right')




        t_diff_data = t_diff_data.T
        t_diff_data = t_diff_data.iloc[::-1]
        print(t_diff_data)
        axs[-1].set_xlabel(r'Time ($\mu$s)')
        axs[-1].tick_params(top=False)
        axs[-1].set_yticks([0, np.log(2)])
        axs[-1].set_yticklabels(['0.00', r'$ln(2)$'])
        axs[-1].yaxis.tick_right()

        for i in range(0, self.n-1):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        # if corr_type == 'QMI':
        #     ax.set_ylabel(f'I(1, {atom})')
        # elif corr_type == 'VNE':
        #     ax.set_ylabel(r'S($\rho_{7}$)')

        #legend = ax.legend(loc='upper left')
        # legend.set_title(r'$\Delta$$t_{quench}$ ($\mu$s)')

        if save_df:
            path = 'Output CSV tables/qmi_data.csv'
            path2 = f'2Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital={self.δ_start},t_q={q_list[0]*0.001},dt=0.001.csv'
            data.to_csv(path, index=True)
            t_diff_data.to_csv(path2, index=True)


        if save_pdf:
            plt.savefig(f'Plotting Data/Propagation Speed EE/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        #plt.tight_layout()
        plt.show()

    def rydberg_correlation_cbs(self, i=None, save_pdf=False):

        rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)

        # Get correlation data
        g_list = [[] for _ in range(self.steps)]

        for j in range(0, self.steps):
            g = self.rydberg_rydberg_density_corr_function(states[j], i=i)
            g_list[j] = g

        plotting_g_data = np.array(g_list)
        print(plotting_g_data)
        plotting_g_data = plotting_g_data.T

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(8, 2.2),
                                gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [1, 1]})

        self.colour_bar(type='rydberg', data=rydberg_fidelity_data, ax=axs[0, 0], cb_ax=axs[0, 1])
        self.colour_bar(type='correlation', data=plotting_g_data, ax=axs[1, 0], cb_ax=axs[1, 1])

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def cb_entanglement_entropy(self, atom_i=None, save_pdf=False):

        if atom_i is not None:
            rydberg_fidelity_data, states, ee = self.time_evolve(rydberg_fidelity=True, entanglement_entropy=True,
                                                                 states_list=True)
        else:
            rydberg_fidelity_data, ee = self.time_evolve(rydberg_fidelity=True, entanglement_entropy=True)
            states = None

        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 2.2))

        self.colour_bar(type='rydberg', data=rydberg_fidelity_data, ax=axs[0], cb=False)
        self.plot_half_sys_entanglement_entropy(ax=axs[1], atom_i=atom_i, entanglement_entropy=ee, states=states)

        plt.subplots_adjust(hspace=0)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def correlation_averages(self, qsteps_list, initial_state, single_addressing_list, save_pdf=False):

        averages_list = [[] for _ in range(len(qsteps_list))]
        averages_dict = {}

        for j, q_step in enumerate(qsteps_list):
            single_addressing_list[0] = int(q_step)

            print(single_addressing_list)

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            states = singleplot.time_evolve(states_list=True)

            g_r_list = [[] for _ in range(self.steps)]

            for i in range(0, self.steps):
                g_r = self.rydberg_rydberg_density_corr_function(states[i], i=1)
                g_r_list[i] = g_r

            data = np.array(g_r_list)
            data = data.T

            averages = []

            for i in range(0, self.n - 1):
                av = np.abs(np.average(data[i][q_step:q_step + 398]))
                averages += [av]

            averages_dict[q_step] = averages
            averages_list[j] = averages

        print(averages_dict)

        averages_list = np.array(averages_list)
        averages_list = averages_list.T

        qsteps_list = np.array(qsteps_list)

        print(averages_list)

        # speeds = np.ones(len(qsteps_list)) / np.array(qsteps_list) * self.dt

        # print(speeds)

        for k in range(0, self.n - 1):
            plt.scatter(qsteps_list * self.dt, averages_list[k], label=f'|G(1, {k + 2})|')
            plt.plot(qsteps_list * self.dt, averages_list[k])

        plt.xlabel(r'$\Delta$$t_{quench}$ ($\mu$s)')
        plt.ylabel('|⟨G(1, i)⟩|')

        plt.legend()

        plt.show()

    def propagation_speed(self, correlation_type='qmi', save_df=False):

        data = pd.DataFrame()
        data['Time'] = self.times

        if correlation_type == 'qmi':
            for j in range(2, self.n + 1):
                data[f'I(1, {j})'] = self.quantum_mutual_information(1, j, data=True)

        print(data)

        print(np.std(data['I(1, 5)']))

        if save_df:
            path = 'Output CSV tables/qmi_data.csv'
            data.to_csv(path, index=True)

    def plot_data(self):

        data = {
            1: [0.13489283051526726, 0.10380234775952514, 0.08040943845775576, 0.06553260627426315, 0.04057111284473093,
                0.02444809915997938],
            21: [0.11023006102623722, 0.09178485138055804, 0.06984160351299987, 0.05934002849246761,
                 0.031205558822378392,
                 0.018287089053959617],
            41: [0.0920279083546462, 0.07840842369274088, 0.057779742326798694, 0.04910861043768696,
                 0.023799060243895557,
                 0.012873887057077581],
            61: [0.08123914209812547, 0.06914794130178653, 0.04979353497461435, 0.041964564808964995,
                 0.01971498711298161,
                 0.00992853963756731],
            81: [0.07005979572903284, 0.05907882148131936, 0.04132772613297933, 0.033931476603844254,
                 0.015109432213584219,
                 0.007100981934281265],
            101: [0.0683400798924331, 0.05756333851706075, 0.040022406064073125, 0.0326811350759612,
                  0.014954944182175628,
                  0.007293985042401345],
            121: [0.06452441202093445, 0.05258587578043243, 0.03763660069310551, 0.03136757858614926,
                  0.013323908914943028,
                  0.005706200670566734],
            141: [0.06165852144773278, 0.050344233085087835, 0.03390752252584309, 0.027089587365593405,
                  0.01080320482039607, 0.00455087747837037],
            161: [0.06049884434406294, 0.0477200580345399, 0.03263154642033303, 0.026348435027717675,
                  0.010519853053357158,
                  0.004301806796079201],
            181: [0.05941969891070888, 0.04718727122455187, 0.03172270879634774, 0.02506887687918867,
                  0.009989879346662225,
                  0.003975630549848664]}

        q_steps = list(data.keys())
        values = list(data.values())

        q_steps = np.array(q_steps)
        values = np.array(values)
        values = values.T

        speed = self.δ_end / self.two_pi * np.ones_like(q_steps) / (self.dt * q_steps)

        print(speed
              )

        for i in range(0, 6):
            plt.scatter(speed[3:], values[i][3:])

        plt.xscale('log')
        plt.show()

        print(values)

        print(speed)

    def multiple_entanglement_entropy(self, initial_states, single_addressing_list, int_detuning, save_pdf=False, atom_i=None):

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 2.5))

        labels = [r'$Z_{2}$', 'Zeros']

        for i, initial_state in enumerate(initial_states):

            singleplot = PlotSingle(self.n, self.t, self.dt, int_detuning[i], int_detuning[i], detuning_type=None,
                                   single_addressing_list=single_addressing_list,
                                   initial_state_list=initial_state, rabi_regime='constant')

            states, ee = singleplot.time_evolve(states_list=True, entanglement_entropy=True)

            label = labels[i] #ploting_tools.state_label(initial_state)

            self.plot_half_sys_entanglement_entropy(ax=ax, atom_i=atom_i, entanglement_entropy=ee, states=states, label=label)

            ax.text(0, 1.32, '(a)', color='r', fontsize=16)
            #ax.axhline(y=np.log(16), color='grey', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=0.4, color='#39FF14', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=0.8, color='#FF007F', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_ylim(0, 1.5)
            ax.tick_params(top=False)
            ax.tick_params(right=False)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/multi_EE.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()





    ''' Investigating thermalization'''

    def sum_rydbergs(self, save_pdf=False):
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 2.7))

        density_matrices = self.time_evolve(density_matrix=True)

        rydberg_sum_2 = data_analysis.rydberg_number_expectations(density_matrices)

        ax.plot(self.times, rydberg_sum_2)
        ax.set_ylabel('⟨Number of Rydberg excitations⟩')
        ax.set_xlabel(r'Time ($\mu$s)')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/ryd_sum.pdf', format='pdf', bbox_inches='tight', dpi=700)




        plt.show()

    ''' Detuning variation'''

    def gs_vs_detuning(self, detunings, state, rabi_list=[4*2 * np.pi], save_pdf=False, inset=False):


        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(4, 2.2))

        #gs = gridspec.GridSpec(2, 2, width_ratios=[0.05, 2.15, 1], height_ratios=[1, 1], hspace=0.3)
        #ax1 = fig.add_subplot(gs[0, 0])
        #ax2 = fig.add_subplot(gs[1, 0])
        #ax3 = fig.add_subplot(gs[:, 1])

        if inset:

            inset_ax = inset_axes(ax, width="25%", height="25%", loc='upper right')
            inset_ax.tick_params(top=False)
            inset_ax.tick_params(right=False)
            inset_ax.tick_params(axis='both', labelsize=10)
            inset_ax.axvline(x=31.9, color='r', linestyle='--', linewidth=1, alpha=0.3)

        single_addressing_list = ['linear'] * self.n
        intial_gs_fidelity_list = [0]*len(detunings)


        for rabi in rabi_list:

            for i, detuning in enumerate(detunings):
                print(detuning/self.two_pi)

                singleplot = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                       single_addressing_list=single_addressing_list,
                                       initial_state_list=state, rabi_regime='constant', Rabi=rabi, a=5.48)

                #eigenstate_probs = singleplot.time_evolve(eigenstate_fidelities=True)

                eigenvalues, eigenvectors, eigenstate_probs = singleplot.time_evolve(
                    eigen_list=True,
                    eigenstate_fidelities=True)


                intial_gs_fidelity_list[i] = eigenstate_probs[0][0]

            # Plot
            ax.scatter(detunings / self.two_pi, intial_gs_fidelity_list, s=12, marker='x')
            ax.plot(detunings/self.two_pi, intial_gs_fidelity_list)

            if inset:
                inset_ax.scatter(detunings / self.two_pi, intial_gs_fidelity_list, s=5, marker='x', label='Inset Graph')
                inset_ax.plot(detunings / self.two_pi, intial_gs_fidelity_list, label='Inset Graph')

                inset_ax.set_xlim(20,45)
                inset_ax.set_ylim(0.95, 1.005)
                inset_ax.set_xticks([20,45])

        ax.axvline(x=31.9, color='r', linestyle='--', linewidth=1, alpha=0.3)
        ax.set_ylim(0,1.1)
        ax.set_xlabel(r'$\Delta$/2$\pi$ (MHz)')
        ax.set_ylabel(r'|⟨$Z_{2}$|$\Psi_{0}$⟩|$^{2}$')
        ax.tick_params(right=False)
        ax.tick_params(top=False)


        if save_pdf:
            plt.savefig(f'Quick Save Plots/gs_vs_detuning.pdf', format='pdf', bbox_inches='tight', dpi=700)


        plt.show()

    def remove_background_colourbar(self, detuning, single_addressing_list, int_state):

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 3))

        singleplot = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                single_addressing_list=single_addressing_list,
                                initial_state_list=int_state, rabi_regime='constant', Rabi=8 * 2 * np.pi)

        singleplot2 = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                 single_addressing_list=['linear'] * self.n,
                                 initial_state_list=int_state, rabi_regime='constant', Rabi=8 * 2 * np.pi)


        quench_data = singleplot.colour_bar(type='pairwise purity')
        print(quench_data)
        background_data = singleplot2.colour_bar(type='pairwise purity')
        print(background_data)

        data = 1 + quench_data - background_data

        ploting_tools.set_up_color_bar(self.n, data, self.times, ax=ax, type='pairwise purity')

        plt.show()

    def two_atom_sweep(self):
        pass

    def two_atom_sweeps_eigenenergies(self, save_pdf=False):

        fig, axs = plt.subplots(1, 3, sharey=True, figsize=(8, 1.5), gridspec_kw={'width_ratios': [5, 5, 1]})

        detuning = 24 * 2 * np.pi

        singleplot = PlotSingle(2, 1.00, 0.01, -detuning, detuning, detuning_type=None,
                                single_addressing_list=['linear']*2,
                                initial_state_list=[0,0], rabi_regime='constant', Rabi=4 * 2 * np.pi)

        singleplot2 = PlotSingle(2, 0.1, 0.0005, -detuning, detuning, detuning_type=None,
                                 single_addressing_list=['linear'] * 2,
                                 initial_state_list=[0,0], rabi_regime='constant', Rabi=4 * 2 * np.pi)


        singleplot.eigenenergies_lineplot_with_eigenstate_fidelities(ax=axs[0], cb=False)
        singleplot2.eigenenergies_lineplot_with_eigenstate_fidelities(ax=axs[1], cb=True, cb_ax=axs[2])

        # axs[0].set_title(r'$t_{sweep}$ = 1.00 $\mu$s', pad=10)
        # axs[1].set_title(r'$t_{sweep}$ = 0.10 $\mu$s', pad=10)

        axs[0].tick_params(top=False)
        #axs[0].tick_params(right=False)
        #axs[1].tick_params(right=False)

        axs[0].set_ylabel(r'$E$/$h$ (MHz)')
        axs[0].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')
        axs[1].set_xlabel(r'$\Delta$/2$\pi$ (MHz)')

        axs[0].text(-23, 30,'Adiabatic', color='#000080')
        axs[1].text(-23, 30, 'Diabatic', color='#000080')
        axs[0].text(6, 30, r'$t_{swp}$' + '=' + '1.00 $\mu$s')
        axs[1].text(6, 30, r'$t_{swp}$' + '=' +  r'0.10 $\mu$s')

        axs[2].axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/two_atom_sweep.pdf', format='pdf', bbox_inches='tight', dpi=700)

        plt.show()

    def NN_compare(self, detunings, initial_state, single_addressing_list, save_pdf):

        fig, axs = plt.subplots(len(detunings), 1, sharex=True, figsize=(8,3.7))


        for i, detuning in enumerate(detunings):
            print(detuning)

            singleplot = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant')

            singleplotNN = PlotSingle(self.n, self.t, self.dt, detuning, detuning, detuning_type=None,
                                    single_addressing_list=single_addressing_list,
                                    initial_state_list=initial_state, rabi_regime='constant', NN=True)

            data = singleplot.time_evolve(rydberg_fidelity=True)[-1]

            NN_data = singleplotNN.time_evolve(rydberg_fidelity=True)[-1]

            axs[i].plot(self.times, NN_data, label=r'Local NN', color='coral')
            axs[i].plot(self.times, data, label=r'$1/R^{6}$', color='#000080')


            axs[i].axhline(y=1, linestyle='--', color='grey')
            axs[i].axvline(x=np.argmin(data) * self.dt, linestyle='--', color='forestgreen', alpha=0.4)
            #axs[i].axvline(x=np.argmin(NN_data) * self.dt, linestyle='--', color='limegreen')


            axs[i].set_ylim(0, 1.2)
            ax2 = axs[i].twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(r'$\Delta_{t=0}$/2$\pi$'+f' = {round(detuning/self.two_pi,1)}'+' (MHz)', rotation=0, labelpad=5, ha='left', va='center',  fontsize=12)

            if i == 4:
                ax2.set_ylabel(r'$\Delta_{t=0}$/2$\pi$' + f' = {round(detuning / self.two_pi, 1)}' + ' (MHz)',
                               rotation=0, labelpad=5, ha='left', va='center', fontsize=12, color='red')

        for i, ax in enumerate(axs):
            ax.set_yticks([])
            ax.tick_params(length=2)

        axs[-1].set_yticks([0,1])
        axs[-1].set_ylabel(r'⟨$n_{9}$⟩')
        axs[-1].set_xlabel('Time after quench ($\mu$s)')
        axs[-1].set_xlim(0,1.2)
        #axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.8), ncol=2)
        axs[0].legend(loc='upper left', ncol=2)



        plt.subplots_adjust(hspace=0)
        #plt.subplots_adjust(left=0.8)

        if save_pdf:
            plt.savefig(f'Quick Save Plots/NN_compare.pdf', format='pdf', bbox_inches='tight', dpi=700)



        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    t = 0.9
    dt = 0.01
    n = 7
    δ_start = 31.85 * 2 * np.pi
    δ_end = 31.85 * 2 * np.pi

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['linear flat'] * 3
    three2 = ['quench'] + ['linear flat'] * 2
    three3 = ['quench'] * 3

    five = ['quench flat'] * 5
    five2 = [0] * 5
    five3 = [1] + ['linear flat'] * 4
    five4 = [1] + ['linear flat'] * 3 + [1]
    five5 = ['linear flat'] * 2 + [1] + ['linear flat'] * 2

    seven = ['quench flat'] * 7
    seven2 = ['quench'] * 7
    seven3 = ['quench'] + ['linear flat'] * 6
    seven4 = ['linear flat'] * 3 + [1] + ['linear flat'] * 3

    nine = ['linear']
    nine2 = ['linear flat'] * 4 + ['quench'] + ['linear flat'] * 4
    nine3 = ['quench'] + ['linear flat'] * 8
    nine4 = ['quench'] + ['linear flat'] * 7 + ['quench']

    Z2 = [1 if i % 2 == 0 else 0 for i in range(n)]
    Zero = [0] * n
    Ones = [1] * n
    Z3 = [1, 0, 0, 1, 0, 0, 1]

    plotter = CombinedPlots(n, t, dt, δ_start, δ_end, detuning_type=None,
                            single_addressing_list=nine3,
                            initial_state_list=Z2, rabi_regime='constant',
                            )

    plotter.two_atom_sweeps_eigenenergies(save_pdf=True)

    plotter.area_law_test([Z2, Zero], seven2, np.array([-24, 24]) * 2 * np.pi, [0.4, 0.8], save_pdf=True)

    #plotter.multiple_entanglement_entropy([Z2, Zero], seven2, np.array([-24, 24]) * 2 * np.pi, save_pdf=True)

    plotter.NN_compare(np.array([21, 24, 27, 30, 31.9, 34, 37]) * 2 * np.pi, Z2, nine3, save_pdf=True)

    #plotter.colorbar_state_fidelity([Zero], save_pdf=True)
    plotter.colorbars_vs_detunings(np.array([27, 30, 31.85, 34]) * 2 * np.pi, nine3, Z2, cm_type='rydberg',
                                    save_pdf=True)

    plotter.qmi_compare(9, [9, 8, 7, 6, 5, 4, 3, 2, 1], [1], corr_type='VNE', save_pdf=True)



    plotter.gs_vs_detuning(np.arange(-5, 110, 2.5) * 2 * np.pi, Z2, rabi_list=[0, 4 * 2 * np.pi], save_pdf=True,
                           inset=True)














    #plotter.eigenspectrums(np.array([24,-24])*2*np.pi, [Z2, Zero], seven2, save_pdf=True)



    #plotter.remove_background_colourbar(31.9*2*np.pi, nine3, Z2)

    #np.array([10, 12, 15, 21, 24, 27, 30, 31.9, 34, 37, 40, 45, 50, 55, 60])




    #plotter.blockade_plots()



    plotter.ordered_state_colourbars([5.48, 3.16], Zero, seven, save_pdf=True)

    #plotter.sum_rydbergs(save_pdf=True)

    #plotter.rydberg_fidelity_barchart_animation(np.arange(0,1.0, 0.01), detunning=True, save=True)

    #plotter.entanglement_propagation_barchart_animation(np.arange(0,6, 0.01), save=True)

    plotter.area_law_test_animation([Z2], single_addressing_list=nine3, times_list=np.arange(0,1, 0.01), save=True)

    # plotter.multiple_eigenenergies_vs_detuning([10, 8],[30 * 2 * np.pi, 24 * 2 * np.pi])
    #

    #
    # #plotter.sum_rydbergs()
    #
    #plotter.cb_entanglement_entropy(atom_i=None, save_pdf=True)
    #





    #plotter.energy_spread([3, 5, 7], np.arange(1, 200, 20))

    # plotter.propagation_speed(save_df=True)

    # plotter.plot_data()

    # plotter.correlation_averages(np.arange(1, 200, 20), [1, 0, 1, 0, 1, 0, 1], seven4)



    # plotter.rydberg_correlation_cbs(i=1)



    # plotter.two_atom_eigenstates(save_pdf=True)
    #
    # plotter.sweep_colourbar(three3, save_pdf=True)
    #
    # plotter.eigenstate_fidelities(save_pdf=True)
    # #
    # plotter.quantum_mutual_informations(save_pdf=True)
    #

    #
    # #plotter.eigenvalue_lineplot(show=True)


