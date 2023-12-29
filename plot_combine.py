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



mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams["text.latex.preamble"]  = r" \usepackage[T1]{fontenc} \usepackage[charter,cal=cmcal]{mathdesign}"
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
                         single_addressing_list=single_addressing_list, initial_state_list=initial_state_list, rabi_regime=rabi_regime)


    def sweep_colourbar(self, type='rydberg', title=None):

        fig, axs = plt.subplots(2, 2, sharex=True, figsize=(12.5, 8), gridspec_kw={'width_ratios': [9, 1], 'height_ratios': [1.7, 1]})

        self.colour_bar(type=type, ax=axs[0, 0], cb_ax=axs[:, 1])
        self.rabi_and_detuning_shape(ax=axs[1, 0])

        plt.subplots_adjust(hspace=0)

        for ax in axs[:, 1]:
            ax.axis('off')


        # Set x axis label
        plt.xlabel('Time ($\mu$s)')

        plt.show()

    def ordered_state_colourbars(self, a_list, initial_state_list, single_addressing_list, save_pdf=False):

        fig, axs = plt.subplots(len(a_list), 3,  figsize=(8, 4),  sharex='col', gridspec_kw={'width_ratios': [6, 3, 1], 'height_ratios': [1, 1]})


        for i in range(0, len(a_list)):

            singleplot = PlotSingle(self.n, self.t, self.dt, self.δ_start, self.δ_end, a=a_list[i], detuning_type=None, single_addressing_list=single_addressing_list,
                     initial_state_list=initial_state_list, rabi_regime='pulse start')

            if i == 0:
                cb_ax = axs[:, 2]
                cb = True
            else:
                cb_ax = None
                cb = False

            singleplot.colour_bar(ax=axs[i, 0], cb=cb, end_ax=axs[i, 1], cb_ax=cb_ax)


        axs[-1, 0].set_xlabel('Time ($\mu$s)')
        axs[-1, 1].set_xlabel(r'⟨$n_{i}$⟩')


        for ax in axs[:, 2]:
            ax.axis('off')

        if save_pdf:
            plt.savefig(f'Quick Save Plots/output.pdf', format='pdf', bbox_inches='tight', dpi=700)


        plt.show()






if __name__ == "__main__":
    t = 4
    dt = 0.01
    n = 7
    δ_start = -30 * 2 * np.pi
    δ_end = 30 * 2 * np.pi

    two = ['quench', 'quench']
    two2 = ['quench', 'linear flat']
    two3 = ['linear', 'linear']

    three = ['linear flat'] * 3

    five = ['linear flat'] * 5

    seven = ['linear flat'] * 7

    plotter = CombinedPlots(n, t, dt, δ_start, δ_end, detuning_type=None,
                         single_addressing_list=seven,
                         initial_state_list=[0, 0, 0, 0, 0, 0, 0], rabi_regime='pulse start'
                         )

    #plotter.eigenvalue_lineplot(show=True)

    plotter.ordered_state_colourbars([5.48, 3.16], [0, 0, 0, 0, 0, 0, 0], seven, save_pdf=True)
