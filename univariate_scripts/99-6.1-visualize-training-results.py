import numpy as np
import pandas as pd
import shared_packages.Feynman.Functions as ff
import shared_packages.SyntheticError.DataTransformation as ef
import shared_packages.Feynman.Constraints as fc
import matplotlib.pyplot as plt

import glob
import os

datafolder = 'data/univariate/5_validation_model_results'
figfolder = 'figures/univariate/6.1-training_results'

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

for file in glob.glob(f'{datafolder}/*.csv'):
  data = pd.read_csv(file)

  xvariable = data['varied_variable_name']
  x = data[xvariable]

  fig, ax = plt.subplots()

  ax.plot(x, data['target'], label='original function',linewidth=0.5,alpha=0.5, c = 'green')
  ax.plot(x, data['target_with_noise'], label='original function with noise',linewidth=0.5,alpha=0.5, c = 'blue')
  ax.plot(x, data['target_with_error_without_noise'], label='original function with error',linewidth=0.5,alpha=0.5, c = 'orange')
  ax.plot(x, data['target_with_error'], label='target',linewidth=0.5,alpha=0.5, c = 'red')
  ax.plot(x, data['Predicted'], label='predicted',linewidth=0.5,alpha=0.5, c = 'purple')
  
  legend_without_duplicate_labels(ax)

  filename =   os.path.basename(file)
  plt.savefig(f'{figfolder}/{filename}.png', dpi=300)
  plt.clf()
  plt.close()