import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import linregress

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




if __name__ == "__main__":
    paths = [
            'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=30,t_q=0.01,dt=0.001.xlsx',
            'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=27,t_q=0.009,dt=0.001.xlsx',
            'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=24,t_q=0.008,dt=0.001.xlsx',
            'Plotting Data/Propagation Speed EE/1.5 separation data/7 Atom D_inital=21,t_q=0.007,dt=0.001.xlsx'
             ]

    data_list = []

    for path in reversed(paths):
        df = pd.read_excel(path)
        data_list += [df]

    propagation_speed(data_list)

