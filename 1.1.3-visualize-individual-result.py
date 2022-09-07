import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import os 

folder = 'C:/Users/P41874/Desktop/results'

for f in os.scandir(folder):
    if f.is_file():
      df = pd.read_csv(f.path)

      
      plt.scatter(x=df['target_with_noise'], y= df['Predicted'])
      plt.show()
      plt.clf()
