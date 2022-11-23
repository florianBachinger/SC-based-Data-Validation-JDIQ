# to make plot interactive
from operator import invert
import numpy as np
from scipy.integrate import simpson
from numpy import trapz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
import matplotlib.cm as cm
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)


df = pd.read_csv('data/univariate/8-AUC_comparison.csv')
df = df[df['DataSize'] == 50]
  

# importing required libraries
from matplotlib import cm;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib.colors import Normalize

plt.interactive(True)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

norm = Normalize(vmin=np.min(df['AUC']), vmax=np.max(df['AUC']))
cmap = sns.color_palette("magma", as_cmap=True)
cbar_ax = fig.add_axes([.86, .4, .02, .40])

X = np.unique(df['ErrorWidthPercentage'])
Y = np.unique(df['NoiseLevelPercentage'])
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)
Z = Z +1

for error_scaling in np.unique(df['ErrorScaling']):
  print(f'scaling {error_scaling}')
  filtered = df[df['ErrorScaling'] == error_scaling]
  pivot = filtered.pivot_table(index='NoiseLevelPercentage', columns='ErrorWidthPercentage',
                           values='AUC', aggfunc=np.average)
  print(pivot)
  surf = ax.plot_surface(X, Y, Z*error_scaling,facecolors = cmap(norm(np.array(pivot))), cmap = cmap,         antialiased = True, alpha= None);

range = np.linspace(0,1,100)

ax.set_xlabel(f'error function width in %')
ax.set_ylabel(f'noise level in percentage of $\sigma$')
ax.set_zlabel(f'error height in z * $\sigma$')

fig.colorbar(surf, cax=cbar_ax)
plt.show(block=True)