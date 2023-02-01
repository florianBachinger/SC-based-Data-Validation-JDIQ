import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

import matplotlib.cm as cm
from matplotlib.colors import Normalize


#--------------------- plot the interpolated contour ---------------------
def PlotContour(zi, x, y, axis , yLabelAddition, text, printTitle=False, plotYLabel = False):
  grid_x, grid_y = np.mgrid[1:3:200j, 1:3:200j]
  grid_z0 = griddata((y,x), zi, (grid_x, grid_y), method='linear')


  axis.contourf(grid_x, grid_y, grid_z0, cmap= cmap , norm= norm)
  axis.set_xlabel("$\sigma$")
  if plotYLabel:
    axis.set_ylabel(yLabelAddition+"\n$\\theta$")
    axis.yaxis.set_ticks(np.arange(1, 4, 1))
  if printTitle:
    axis.text(2, 3, text ,   fontsize = 10, va='bottom', ha='center')


#--------------------- define figure size ---------------------
fig, ax = plt.subplots(1,3, figsize=(6,2.2),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.09, bottom=0.22, right=0.87, top=0.85, wspace=0.1, hspace=0.1)

# --------------------- read Feynman2 with noise and error ---------------------
borderpoints = [(1.0,1.0),(1.0,3.0),(3.0,1.0),(3.0,3.0)]
border = pd.DataFrame(data = borderpoints , columns= ['sigma','theta' ])
border['target_with_noise'] = np.array([ ff.Feynman2.calculate(sigma,theta) for (sigma, theta) in borderpoints],dtype=float)
border['error_function'] = np.array([0,0,0,0],dtype=float)
border['target_with_error'] = np.array([ ff.Feynman2.calculate(sigma,theta) for (sigma, theta) in borderpoints],dtype=float)

df1 = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_Square.csv')

df1 = df1.append(border)
x = df1['theta']
y = df1['sigma']
z1 = df1['target_with_noise']
z2 = df1['error_function'] * 1.5 * np.std( df1['target'])
z3 = df1['target_with_error']

max = np.max(list(zip(z1,z2,z3)))
norm = Normalize( vmin=-max, vmax=max)
cmap = cm.get_cmap('coolwarm')

# --------------------- plot Feynman2 with noise and error ---------------------

PlotContour(z1,x,y,ax[0],"", "random noise\nwith $\zeta = 0.01$",True, True)
PlotContour(z2,x,y,ax[1],"",  "error function\nwith $\psi = 0.125$",True, False)
PlotContour(z3,x,y,ax[2],"",  "erroneous dataset\nwith $\phi = 1.5$",True, False)

ax[1].add_patch(Rectangle((1.09,1.76),0.75,0.75,
                    edgecolor= sns.color_palette("coolwarm", n_colors=10).as_hex()[9],
                    facecolor='none',
                    lw=1))

ax[0].scatter(y,x, s = 9, marker = "x",linewidth = 1, c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.5)
ax[1].scatter(y,x, s = 9, marker = "x",linewidth = 1, c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.5)
ax[2].scatter(y,x, s = 9, marker = "x",linewidth = 1, c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.5)

cbar_ax = fig.add_axes([.89, .22, .025, .63])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),cax = cbar_ax)
cbar_ax.set_xlim(0,max)
cbar_ax.set_ylim(0,max)
plt.savefig('figures/experimental_setup/Feynman2_data_generation_error.png', dpi = 600)
plt.show()
