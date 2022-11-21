import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt

datafolder = 'data/2.1.1-univariate_error'
figfolder = 'fig/2.1.2-univariate_error'
df = pd.read_csv(f'{datafolder}/info/overview.csv')
for (id, row) in df.iterrows():
  data = pd.read_csv(f'{datafolder}/{row.FileName}.csv')
  x = range(len(data))

  plt.plot(x, data['target'], label='target',alpha=0.5)
  plt.plot(x, data['target_with_noise'], label = 'target noise',alpha=0.5)
  plt.plot(x, data['target_with_error_without_noise'], label = 'target error',alpha=0.5)
  plt.plot(x, data['target_with_error'], label='target noise error',alpha=0.5)

  plt.savefig(f'{figfolder}/{row.FileName}.png')
  plt.clf()
  plt.close()