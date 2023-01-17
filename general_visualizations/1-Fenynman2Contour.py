import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc

import matplotlib.cm as cm
from matplotlib.colors import Normalize



data = ff.Feynman2.generate_df()
xlist = np.linspace(1, 3.0, 100)
ylist = np.linspace(1, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)

sigma = X
theta = Y
Z = ff.Feynman2.calculate(sigma, theta)

F2Constraints = [ x for x in fc.constraints if x['EquationName'] == 'Feynman2'][0]
F2order1sigma = [ y for y in F2Constraints['Constraints'] if ((y['name'] == 'sigma') and (y['order_derivative'] == 1)) ][0]
f2derived = eval(F2order1sigma['derivative_lambda'])
Z2 = f2derived([sigma,theta])

F2Constraints = [ x for x in fc.constraints if x['EquationName'] == 'Feynman2'][0]
F2order1theta = [ y for y in F2Constraints['Constraints'] if ((y['name'] == 'theta') and (y['order_derivative'] == 1)) ][0]
f2derived = eval(F2order1theta['derivative_lambda'])
Z3 = f2derived([sigma,theta])

print([np.min(Z),np.min(Z2), np.min(Z3)])
print([np.max(Z),np.max(Z2), np.max(Z3)])
min  = np.min([np.min(Z),np.min(Z2), np.min(Z3)])
max  = np.max([np.max(Z),np.max(Z2), np.max(Z3)])

norm = Normalize(vmin=min, vmax=max )
cmap = cm.get_cmap('coolwarm')

fig, ax = plt.subplots(1,3, figsize=(6,2.2),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.09, bottom=0.22, right=0.87, top=0.85, wspace=0.1, hspace=0.1)

cp = ax[0].contourf(X,Y,Z, cmap= cmap , norm= norm)
ax[0].set_xlabel("$\sigma$")
ax[0].set_ylabel("$\\theta$")


cp = ax[1].contourf(X,Y,Z2, cmap = cmap, norm= norm)
ax[1].set_xlabel("$\sigma$")

cp = ax[2].contourf(X,Y,Z3, cmap = cmap, norm = norm)
ax[2].set_xlabel("$\sigma$")


ax[0].text(1.5, 3, '$f(\sigma,\\theta)$' ,   fontsize = 10, va='bottom', ha='left')
ax[1].text(1.8, 3.01, '$\\frac{\partial f}{\partial \sigma}$' ,   fontsize = 14, va='bottom', ha='left')
ax[2].text(1.8, 3.01, '$\\frac{\partial f}{\partial \\theta}$',  fontsize = 14, va='bottom', ha='left')

ax[0].yaxis.set_ticks(np.arange(1, 4, 1))

cbar_ax = fig.add_axes([.89, .22, .025, .63])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),cax = cbar_ax)

plt.savefig('figures/experimental_setup/Feynman2_contour_derviative.png', dpi = 600)
plt.show()