import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import linregress

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

def propagation_speed(data_list):
    fig, ax = plt.subplots()

    for data in data_list:
        l = np.array(data['l'])
        t_diff = np.array(data['t_diff'])

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(t_diff, l)

        x_intercept = -intercept/slope

        print(slope)
        print(x_intercept)

        # Create scatter plot
        ax.scatter(t_diff[:], l[:], label='Data')
        ax.plot(t_diff[:], l[:], label='Data')

        # Plot the line of best fit
        #ax.plot(t_diff, slope * t_diff + intercept, color='red', label='Line of Best Fit')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add labels and legend
    plt.ylabel('Sites travelled')
    plt.xlabel('t*')
    #plt.legend()

    # Show plot
    plt.show()

def propagation_speed2(data_list, legend_list, threshold_type='Time 3sd', save_pdf=False):
    fig, axs = plt.subplots(1,2, sharey=True, figsize=(8,2))

    markers = ['o','s','^','d','*','v']
    x_errors_asymmetric = [[0.005]*8, [0.005]*8]


    for i, data in enumerate(data_list):
        l = np.array(data['Site'])
        t_diff = np.array(data[threshold_type])
        t_diff2 = np.array(data['Time 1.1'])

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = linregress(t_diff[1:], l[1:])

        x_intercept = -intercept/slope

        #print(slope)
        print(x_intercept)

        # Create scatter plot

        axs[1].scatter(t_diff[1:], l[1:], marker=markers[i], label=legend_list[i], s=20)
        axs[1].plot(t_diff[1:], l[1:])

        axs[0].scatter(t_diff2[1:], l[1:], marker=markers[i], s=20)
        axs[0].plot(t_diff2[1:], l[1:])

    axs[0].set_title(r'$S_{EE}(\rho_{quench})$ > 1.1 x $S_{EE}(\rho_{const}$)', pad=10, fontsize=11)
    axs[1].set_title(r'$S_{EE}(\rho_{quench})$ >  $ \langle S_{EE}(\rho_{const}) \rangle_{t<0.6 \mu s}$ + 3$\sigma_{const}$', pad=10, fontsize=11)
    axs[0].grid(True)
    axs[1].grid(True)

    fig.subplots_adjust(right=0.8)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.08, 1.03), borderaxespad=0., title=r'$\Delta_{t=0}$/2$\pi$ (MHz)')


        # Plot the line of best fit
        #ax.plot(t_diff, slope * t_diff + intercept, color='red', label='Line of Best Fit')

    for ax in axs:
        ax.set_ylim(bottom=0)
        ax.tick_params(top=False)
        ax.set_yticks(np.arange(0,9,1))
        ax.set_xticks(np.arange(0, 0.61, 0.1))
        ax.tick_params(axis='both', which='major', length=3)
        ax.set_xlabel(r'$t$* ($\mu$s)')

    axs[0].set_xlim(0, 0.5)
    axs[1].set_xlim(0, 0.60)
    axs[0].set_ylabel('Sites travelled')

    plt.subplots_adjust(wspace=0.15)


    # Add labels and legend

    # plt.xlabel('t*')
    #plt.legend()

    if save_pdf:
        plt.savefig(f'Quick Save Plots/EE_velocity_plot.pdf', format='pdf', bbox_inches='tight', dpi=700)

    # Show plot
    plt.show()




if __name__ == "__main__":
    # paths = [
    #         'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=30,t_q=0.01,dt=0.001.xlsx',
    #         'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=27,t_q=0.009,dt=0.001.xlsx',
    #         'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=24,t_q=0.008,dt=0.001.xlsx',
    #         'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=21,t_q=0.007,dt=0.001.xlsx'
    #          ]

    paths = [
        'Plotting Data/Propagation Speed EE/9 Atom/D=21.xlsx',
        'Plotting Data/Propagation Speed EE/9 Atom/D=24.xlsx',
        'Plotting Data/Propagation Speed EE/9 Atom/D=27.xlsx',
        'Plotting Data/Propagation Speed EE/9 Atom/D=30.xlsx',
        'Plotting Data/Propagation Speed EE/9 Atom/D=31.9.xlsx',
        'Plotting Data/Propagation Speed EE/9 Atom/D=34.xlsx'
    ]

    data_list = []
    legend_list = ['34.0', r'31.9 ($V_{NN}$)','30.0','27.0', '24.0','21.0']

    for path in reversed(paths):
        df = pd.read_excel(path)
        data_list += [df]

    propagation_speed2(data_list, legend_list, save_pdf=True)

