import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc
from scipy.interpolate import griddata

import matplotlib.cm as cm
from matplotlib.colors import Normalize

# df = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_None.csv')
df1 = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_Square.csv')
df2 = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_Spike.csv')
df3 = pd.read_csv('data/multivariate/4_datasets_with_error/Feynman2_s200_n0.01_es1.5_ew0.125_Normal.csv')



fig, ax = plt.subplots(3,3, figsize=(4,4),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1,1,1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.14, bottom=0.12, right=0.99, top=0.95, wspace=0.1, hspace=0.1)



def plotRow(zi,x,y, axis,yLabelAddition,text, printTitle=False, plotYLabel = False):
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
    axis.text(1.5, 3, text ,   fontsize = 10, va='bottom', ha='left')


def plotDataFrame(df, axisrow, yLabelAddition, printTitle=False):
  x = df['theta']
  y = df['sigma']
  z1 = df['target_with_noise']
  z2 = df['error_function']
  z3 = df['target_with_error']

  plotRow(z1,x,y,axisrow[0],yLabelAddition, '$f(\sigma,\\theta)$', printTitle, True)
  plotRow(z2,x,y,axisrow[1],yLabelAddition, '$err(\sigma,\\theta)$', printTitle, False)
  plotRow(z3,x,y,axisrow[2],yLabelAddition, '$f_{err}(\sigma,\\theta)$', printTitle, False)

plotDataFrame(df1, ax[0],'Square', True)
plotDataFrame(df2, ax[1],'Spike', False)
plotDataFrame(df3, ax[2],'Normal', False)

plt.savefig('figures/experimental_setup/Feynman2_errorFunctions.png', dpi = 600)
plt.show()
