import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt

import glob
import os

windowsize = 200
stepwidth = 100

datafolder = 'data/3.1.1-univariate_error_results'
overview = pd.read_csv(f'data/2.1.1-univariate_error/info/overview.csv')
results = pd.read_csv('3.1.1-result_univariate_error.csv')

def Scale(val, minimum, maximum):
  range = maximum - minimum
  return (val - minimum)/range;

def RMSE(target, estimate):
  mse = np.average((target-estimate) * (target-estimate))
  return np.sqrt(mse)

df = pd.DataFrame(columns=['filename','equation_name','varied_variable_name','errorfunction','RMSE','ConstraintsViolated'])
i = 0
for file in glob.glob(f'{datafolder}/*.csv'):
  data = pd.read_csv(file)
  data_length = len(data)
  target =data['target_with_error']
  prediction =data['Predicted']
  max = np.max([target,prediction])
  min = np.min([target,prediction])

  scaled_target = Scale(target,min,max)
  scaled_prediction = Scale(prediction,min,max)

  RMSE_areas = []

  for index in range(int(data_length/stepwidth)):
    start = index * stepwidth
    stop = ((index + 1) * stepwidth)+windowsize
    if stop > data_length:
      stop = data_length - 1 
    if(start!=stop):
      target_selection = scaled_target[start:stop]
      prediction_selection = scaled_prediction[start:stop]
      RMSE_areas.append(RMSE(target_selection,prediction_selection))
  
  filename =  os.path.basename(file)
  equation_name = data['equation_name'][0]
  varied_variable_name = data['varied_variable_name'][0]
  if '_' in varied_variable_name:
    errorfunction = filename.split('_')[3]
  else:
    errorfunction = filename.split('_')[2]

  filtered = overview[((overview['EquationName'] ==equation_name) & (overview['Variable'] ==varied_variable_name)
            & (overview['ErrorFunction'] ==errorfunction))]

  if(len(filtered)!=1):
     raise 'error'
  ConstraintsViolated = filtered['ConstraintsViolated'].values[0]
  df.loc[i] =[equation_name,filename,varied_variable_name,errorfunction,str(RMSE_areas), ConstraintsViolated]
  i = i +1
  
df.to_csv('3.1.1_training_error.csv')


