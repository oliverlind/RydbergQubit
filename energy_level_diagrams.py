import numpy as np
import matplotlib.pyplot as plt
from energydiagram import ED

Rabi = 4*2 * np.pi


diagram = ED()
diagram.add_level(0,'Separated Reactants')
diagram.add_level(4*2 * np.pi, r'|rr⟩',top_text='4*2 * np.pi') #Using 'last'  or 'l' it will be together with the previous level
diagram.add_level(-4*2 * np.pi,r'|00⟩','last',)


diagram.add_link(0,1)
diagram.add_link(0,2)



diagram.plot(show_IDs=False)

plt.ylabel(f'Energy Eigenvalue {"($ħ^{-1}$)"}', fontsize=12)
plt.show()