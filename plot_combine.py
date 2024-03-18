import sys
import pandas as pd
from scipy.fft import fft, fftfreq, rfft, rfftfreq
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
from matplotlib.animation import FuncAnimation


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

    '''State Fidelity'''

    def colorbar_state_fidelity(self, states_to_test, type='rydberg', save_pdf=False):

        if type == 'rydberg':
            rydberg_fidelity_data, states = self.time_evolve(rydberg_fidelity=True, states_list=True)
        else:
            sys.exit()

        fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 4.2),
                                gridspec_kw={'width_ratios': [13, 1], 'height_ratios': [0.9, 1.7, 0.9]})

        sweep = self.detunning[0] / self.two_pi
        quench = self.t*4.5/7 - 0.01
        axs[0, 0].plot(self.times, sweep)
        axs[0, 0].set_ylabel(r'$\Delta$/2$\pi$ (MHz)  ')
        axs[0, 0].set_ylim(-39, 39)
        axs[0, 0].axvspan(xmin=0, xmax=quench, color='green', alpha=0.1)
        axs[0, 0].axvspan(xmin=quench, xmax=self.t, color='red', alpha=0.1)
        axs[0, 0].text(0.2, 19, 'Sweep', color='green')
        axs[0, 0].text(6.1, 19, 'Quench', color='red')
        axs[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.2)

        self.colour_bar(data=rydberg_fidelity_data, type=type, ax=axs[1, 0], cb_ax=axs[:, 1])
        self.state_fidelity(states_to_test, q_states=states, ax=axs[2, 0])

        axs[2, 0].set_ylabel(r'⟨$Z_{2}$|$\Psi$⟩')
        axs[2, 0].set_xlabel(r'Time ($\mu$s)')
        axs[2, 0].set_ylim(0, 1.1)

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=600)

        plt.show()

    '''Eigenstates'''

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

        plt.figure(figsize=(4,6))

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



    def area_law_test(self):

        pass



        # else:
        #     steps = [-1]
        #
        # for initial_state in initial_states:
        #
        #     singleplot = PlotSingle(n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
        #                             single_addressing_list=single_addressing_list,
        #                             initial_state_list=initial_state, rabi_regime='constant')
        #
        #     states = singleplot.time_evolve(states_list=True)
        #
        #
        #
        #     for step in steps:
        #
        #         vne_list = [0]
        #         sites_list = np.arange(0, self.n+1, 1)
        #
        #         for n_a in range(1, self.n):
        #
        #             rdm = self.reduced_density_matrix_from_left(self.n, n_a, states[step])
        #             vne = data_analysis.von_nuemann_entropy(rdm)
        #
        #             vne_list += [vne]
        #
        #         vne_list += [0]
        #
        #         ax.scatter(sites_list, vne_list)
        #         ax.plot(sites_list, vne_list)
        #
        # plt.show()



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


        fig, axs = plt.subplots(len(atom_list), 1, figsize=(8, 4))
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
                         start = data[data[f'VNE atom{atom} t_q = {single_addressing_list[0]}'] > 1.2*data[f'VNE atom{atom} t_q = linear flat']]
                         start.reset_index(drop=True, inplace=True)
                         if not start.empty:
                            start = start['Time'][0]
                            print(start)
                         else:
                            start = 0

                         t_diff_data[f'Atom {atom}'] = [start]
                         axs[k].axvline(x=start, color='red', linestyle='--', linewidth=1, alpha=0.5)

                     axs[k].set_ylabel(r'$S_{EE}$($\rho_{'+f'{atom}'+'}$)', rotation=0, labelpad=10, ha='right')




        t_diff_data = t_diff_data.T
        t_diff_data = t_diff_data.iloc[::-1]
        print(t_diff_data)
        axs[-1].set_xlabel(r'Time ($\mu$s)')
        axs[-1].tick_params(top=False)
        axs[-1].set_yticks([0, np.log(2)])
        axs[-1].set_yticklabels(['0.00', r'$ln(2)$'])
        axs[-1].yaxis.tick_right()

        for i in range(0,self.n-2):
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

    def multiple_entanglement_entropy(self, initial_states, single_addressing_list, atom_i=None):

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 2.2))

        for initial_state in initial_states:

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, detuning_type=None,
                                   single_addressing_list=single_addressing_list,
                                   initial_state_list=initial_state, rabi_regime='constant')

            states, ee = singleplot.time_evolve(states_list=True, entanglement_entropy=True)

            label = ploting_tools.state_label(initial_state)

            self.plot_half_sys_entanglement_entropy(ax=ax, atom_i=atom_i, entanglement_entropy=ee, states=states, label=label)

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




if __name__ == "__main__":
    t = 7
    dt = 0.01
    n = 7
    δ_start = -24 * 2 * np.pi
    δ_end = 24 * 2 * np.pi

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
    seven3 = [1] + ['linear flat'] * 6
    seven4 = ['linear flat'] * 3 + [1] + ['linear flat'] * 3

    nine = ['quench']
    nine2 = ['linear flat'] * 4 + [1] + ['linear flat'] * 4
    nine3 = [1] + ['linear flat'] * 8

    Z2 = [1 if i % 2 == 0 else 0 for i in range(n)]
    Zero = [0] * n

    plotter = CombinedPlots(n, t, dt, δ_start, δ_end, detuning_type=None,
                            single_addressing_list=seven,
                            initial_state_list=Zero, rabi_regime='constant'
                            )

    plotter.colorbar_state_fidelity([Z2], save_pdf=True)

    plotter.ordered_state_colourbars([5.48, 3.16], Zero, seven, save_pdf=True)

    #plotter.sum_rydbergs(save_pdf=True)

    #plotter.rydberg_fidelity_barchart_animation(np.arange(0,1.0, 0.01), detunning=True, save=True)

    #plotter.entanglement_propagation_barchart_animation(np.arange(0,6, 0.01), save=True)

    plotter.area_law_test_animation([Z2], single_addressing_list=nine3, times_list=np.arange(0,1, 0.01), save=True)

    # plotter.multiple_eigenenergies_vs_detuning([10, 8],[30 * 2 * np.pi, 24 * 2 * np.pi])
    #
    #plotter.multiple_entanglement_entropy([[1,0,0,1,1,0,1],[0,0,0,0,0,0,0]], single_addressing_list=seven2)
    #
    # #plotter.sum_rydbergs()
    #
    #plotter.cb_entanglement_entropy(atom_i=None, save_pdf=True)
    #
    #plotter.qmi_compare(9, [9, 8, 7, 6, 5, 4, 3, 2], [1], corr_type='VNE', save_pdf=True)




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


