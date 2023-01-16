import numpy as np
import pandas as pd
import shared_packages.Feynman.Functions as ff
import shared_packages.SyntheticError.DataTransformation as ef
import shared_packages.Feynman.Constraints as fc
import matplotlib.pyplot as plt

import glob
import os

datafolder = 'data/multivariate/5_validation_model_results'
overview = pd.read_csv(f'data/multivariate/4_datasets_with_error/_info.csv')
results = pd.read_csv('data/multivariate/5_validation_model_results/_results.csv')

def Scale(val, minimum, maximum):
  range = maximum - minimum
  return (val - minimum)/range;

def RMSE(target, estimate):
  mse = np.average((target-estimate) * (target-estimate))
  return np.sqrt(mse)

df = pd.DataFrame(columns=['filename','equation_name','errorfunction','RMSE','ConstraintsViolated','DataSize','ErrorWidthPercentage','NoiseLevelPercentage','ErrorScaling'])
i = 0
files = glob.glob(f'{datafolder}/*.csv')
for file in files:
  print(f'[{i}/{len(files)}] {file}')
  if( "\_results.csv" in file ):
    continue

  data = pd.read_csv(file)
  data_length = len(data)
  target =data['target_with_error']
  prediction =data['Predicted']
  max = np.max([target,prediction])
  min = np.min([target,prediction])

  scaled_target = Scale(target,min,max)
  scaled_prediction = Scale(prediction,min,max)


  rmse = RMSE(scaled_target,scaled_prediction)
  
  result_filename_withExtension =  os.path.basename(file)
  result_filename = os.path.splitext(result_filename_withExtension)[0]

  filename = result_filename.replace('_d5_i3_l1E-05_a0,5','')

  equation_name = data['equation_name'][0]
  errorfunction = filename.split('_')[2]

  filtered = overview[overview["FileName"] == filename]

  if(len(filtered)!=1):
     raise 'error'
  ConstraintsViolated = filtered['ConstraintsViolated'].values[0]
  DataSize = filtered['DataSize'].values[0]
  ErrorWidthPercentage = filtered['ErrorWidthPercentage'].values[0]
  NoiseLevelPercentage = filtered['NoiseLevelPercentage'].values[0]
  ErrorScaling = filtered['ErrorScaling'].values[0]
  df.loc[i] =[equation_name,filename,errorfunction,rmse, ConstraintsViolated,DataSize,ErrorWidthPercentage,NoiseLevelPercentage,ErrorScaling]
  i = i +1
  
df.to_csv('data/multivariate/6-training_error.csv')


