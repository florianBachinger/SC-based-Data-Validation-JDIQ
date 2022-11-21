import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt

import glob
import os

datafolder = 'data/3.1.1-univariate_error_results'
figfolder = 'fig/3.1.1-univariate_error'

for file in glob.glob(f'{datafolder}/*.csv'):
  data = pd.read_csv(file)

  x = range(len(data))

  plt.plot(x, data['target_with_error_without_noise'], label = 'target error',alpha=0.5)
  plt.plot(x, data['target_with_error'], label='target noise error',alpha=0.5)
  plt.plot(x, data['Predicted'], label='target noise error',alpha=0.5)

  filename =   os.path.basename(file)

  plt.savefig(f'{figfolder}/{filename}.png')
  plt.clf()
  plt.close()