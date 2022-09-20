import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt

import glob
import os

datafolder = 'data/3.1.1-univariate_error_results'
figfolder = 'fig/3.1.1-univariate_error_scatter'

for file in glob.glob(f'{datafolder}/*.csv'):
  data = pd.read_csv(file)

  x = range(len(data))

  plt.scatter( data['Predicted'], data['target_with_error'])

  filename =   os.path.basename(file)

  plt.savefig(f'{figfolder}/{filename}.png')
  plt.clf()
  plt.close()