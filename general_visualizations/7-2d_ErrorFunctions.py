import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc
import shared_packages.SyntheticError.DataTransformation as sedt
from scipy.interpolate import griddata
import random

import matplotlib.cm as cm
from matplotlib.colors import Normalize

random.seed(31415)
np.random.seed(31415)


#--------------------- define figure size ---------------------
fig, ax = plt.subplots(1,3, figsize=(5,1.5),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.09, bottom=0.15, right=0.99, top=0.97, wspace=0.1, hspace=0.1)



dimension_data = [{
  'name': 'x',
  'low': 0,
  'high' : 1,
  'width' : 1,
  'center' : 0.5
  },{
  'name': 'y',
  'low': 0,
  'high' : 1,
  'width' : 1,
  'center' : 0.5
  } ]

dimSize = 100

xlist = np.linspace(0,1, dimSize)
ylist = np.linspace(0,1, dimSize)
X, Y = np.meshgrid(xlist, ylist)

x = X.flatten()
y = Y.flatten()
z = [0] * len(x)

affected_space_width = sedt.Multivariate.CalculateAffectedSpace(len(x),dimension_data,0.2)
shifted_space = sedt.Multivariate.ShiftAffectedSpace( dimension_data, affected_space_width)


def drawError(ax, functionName):
  data_error = pd.DataFrame(list(zip(x,y,z)), columns= ["x","y","z"])
  data_error = sedt.Multivariate.ApplyErrorFunction(error_function_name = functionName
                                            ,data = data_error
                                            ,input_target_name = "z"
                                            ,output_target_name = "z_e"
                                            ,error_function_variable_name = 'ef'
                                            ,affected_space = shifted_space
                                            ,error_value = 1
                                            ,returnErrorFunction=True)

  Z = np.array(data_error['ef']).reshape((-1, dimSize))

  norm = Normalize(vmin=-np.max(Z) , vmax=np.max(Z) )
  cmap = cm.get_cmap('coolwarm')

  ax.set_xticks([0,1])
  ax.set_xticklabels(['0','1'])
  ax.set_yticks([0,1])
  ax.set_yticklabels(['0','1'])
  cp = ax.contourf(X,Y,Z, cmap= cmap , norm= norm)

drawError(ax[0],'Spike')
drawError(ax[1],'Square')
drawError(ax[2],'Normal')

plt.savefig('figures/experimental_setup/2d_errorFunctions.png', dpi = 600)
plt.show()
