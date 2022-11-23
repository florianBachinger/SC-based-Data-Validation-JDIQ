import numpy as np
import pandas as pd
import shared_packages.Feynman.Functions as ff
import shared_packages.SyntheticError.DataTransformation as ef
import shared_packages.Feynman.Constraints as fc
import matplotlib.pyplot as plt
from ast import literal_eval
from datetime import datetime

data = pd.read_csv('3.2.3_training_error.csv')

i = 0
border_count = 0
length = 1000

df= pd.DataFrame(columns=['DataSize','ErrorWidthPercentage','NoiseLevelPercentage','ErrorScaling','Border','true positive rate','false positive rate'])

for border in np.linspace(np.min(data['RMSE']), np.max(data['RMSE']), length):
  print(f'{datetime.now()} [{border_count}/{length}] {border}')
  border_count = border_count + 1
  for data_size in np.unique(data['DataSize']):
    for error_width_percentage in np.unique(data['ErrorWidthPercentage']):
      for noise_level_percentage in np.unique(data['NoiseLevelPercentage']):
        for error_scaling_sigma in np.unique(data['ErrorScaling']):

          filtered = data[(
            (data['DataSize'] == data_size) &
            (data['ErrorWidthPercentage'] == error_width_percentage) &
            (data['NoiseLevelPercentage'] == noise_level_percentage) &
            (data['ErrorScaling'] == error_scaling_sigma) )]
          
          if(len(filtered) == 0):
            continue
          print(len(filtered))
          
          knownInvalidCount = np.sum(filtered['ConstraintsViolated']==True)
          knownValidCount = np.sum(filtered['ConstraintsViolated']==False)

          truePositives = np.sum(((filtered['ConstraintsViolated']==True) & (filtered['RMSE']>=border)))
          falsePositives = np.sum(((filtered['ConstraintsViolated']==False) & (filtered['RMSE']>=border)))
          trueNegatives = np.sum(((filtered['ConstraintsViolated']==False) & (filtered['RMSE']<border)))
          falseNegatives = np.sum(((filtered['ConstraintsViolated']==True) & (filtered['RMSE']<border)))

          truePositivesRate = (truePositives) / (truePositives + falseNegatives)
          falsePositivesRate = (falsePositives) / (falsePositives + trueNegatives)

          df.loc[i] =[data_size, error_width_percentage, noise_level_percentage, error_scaling_sigma, border, truePositivesRate, falsePositivesRate]
          i = i +1

df.to_csv('data/multivariate/7-rocCurveResults.csv', index = False)