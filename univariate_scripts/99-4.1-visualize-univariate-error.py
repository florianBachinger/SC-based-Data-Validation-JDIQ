import numpy as np
import pandas as pd
import shared_packages.Feynman.Functions as ff
import shared_packages.SyntheticError.DataTransformation as ef
import shared_packages.Feynman.Constraints as fc
import matplotlib.pyplot as plt

datafolder = 'data/univariate/4_datasets_with_error'
figfolder = 'figures/univariate/4.1_datasets_with_error'
df = pd.read_csv(f'{datafolder}/_info.csv')

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

for (id, row) in df.iterrows():
  data = pd.read_csv(f'{datafolder}/{row.FileName}.csv')
  xvariable = data['varied_variable_name']
  x = data[xvariable]

  fig, ax = plt.subplots()

  ax.plot(x, data['target'], label='original function',linewidth=0.5,alpha=0.5, c = 'green')
  ax.plot(x, data['target_with_noise'], label='original function with noise',linewidth=0.5,alpha=0.5, c = 'blue')
  ax.plot(x, data['target_with_error_without_noise'], label='original function with error',linewidth=0.5,alpha=0.5, c = 'orange')
  ax.plot(x, data['target_with_error'], label='target',linewidth=0.5,alpha=0.5, c = 'red')
  
  legend_without_duplicate_labels(ax)

  plt.savefig(f'{figfolder}/{row.FileName}.png', dpi=200)
  plt.clf()
  plt.close()