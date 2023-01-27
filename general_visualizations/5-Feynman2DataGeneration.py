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

  norm = Normalize( vmin=-np.max(zi), vmax=np.max(zi))
  cmap = cm.get_cmap('coolwarm')
  axis.contourf(grid_x, grid_y, grid_z0, cmap= cmap , norm= norm)
  axis.set_xlabel("$\sigma$")
  if plotYLabel:
    axis.set_ylabel(yLabelAddition+"\n$\\theta$")
    axis.yaxis.set_ticks(np.arange(1, 4, 1))
  if printTitle:
    axis.text(2, 3, text ,   fontsize = 10, va='bottom', ha='center')


#--------------------- define figure size ---------------------
fig, ax = plt.subplots(1,4, figsize=(6,2),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1], 'width_ratios': [1,1,1,1]})
plt.subplots_adjust(left=0.09, bottom=0.24, right=0.98, top=0.82, wspace=0.1, hspace=0.1)

# --------------------- generate Feynman2 without noise ---------------------
data = ff.Feynman2.generate_df()
xlist = np.linspace(1, 3.0, 100)
ylist = np.linspace(1, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)

sigma = X
theta = Y
Z = ff.Feynman2.calculate(sigma, theta)

# --------------------- plot Feynman2 without noise ---------------------
norm = Normalize( vmin=-np.max(Z), vmax=np.max(Z))
cmap = cm.get_cmap('coolwarm')
ax[0].contourf(X,Y,Z, cmap= cmap , norm= norm)
cp = ax[0].contourf(X,Y,Z, cmap= cmap , norm= norm)
ax[0].set_xlabel("$\sigma$")
ax[0].set_ylabel("$\\theta$")
ax[0].text(2, 3, "equation output" ,   fontsize = 10, va='bottom', ha='center')

# --------------------- read Feynman2 with noise and error ---------------------

borderpoints = [(1.0,1.0),(1.0,3.0),(3.0,1.0),(3.0,3.0)]
border = pd.DataFrame(data = borderpoints , columns= ['sigma','theta' ])
border['target'] = np.array([ ff.Feynman2.calculate(sigma,theta) for (sigma, theta) in borderpoints],dtype=float)
border['target_with_noise'] = np.array([ ff.Feynman2.calculate(sigma,theta) for (sigma, theta) in borderpoints],dtype=float)
border['error_function'] = np.array([0,0,0,0],dtype=float)
border['target_with_error'] = np.array([ ff.Feynman2.calculate(sigma,theta) for (sigma, theta) in borderpoints],dtype=float)
df1 = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_Square.csv')

df1 = df1.append(border)
x = df1['theta']
y = df1['sigma']

z1 = df1['target']
z2 = df1['target_with_noise']
z3 = df1['error_function']
z4 = df1['target_with_error']

# --------------------- plot uniform random input ---------------------

ax[1].scatter(y,x, s = 3, marker = "x", c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.4)
ax[1].text(2, 3, "input space\nuniform random" ,   fontsize = 10, va='bottom', ha='center')
ax[1].set_xlabel("$\sigma$")

# --------------------- plot linear approximation ---------------------

PlotContour(z1,x,y,ax[2],"", "linear\napproximation",True, False)
ax[2].scatter(y,x, s = 3, marker = "x", c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.4)


# --------------------- plot Feynman2 with noise and error ---------------------

PlotContour(z2,x,y,ax[3],"", "linear approximation\nwith $\zeta = 0.01$",True, False)
ax[3].scatter(y,x, s = 3, marker = "x", c =  sns.color_palette("coolwarm", n_colors=10).as_hex()[9], alpha =.4)


plt.savefig('figures/experimental_setup/Feynman2_data_generation.png', dpi = 600)
plt.show()
