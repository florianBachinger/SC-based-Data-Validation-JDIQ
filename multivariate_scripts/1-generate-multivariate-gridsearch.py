from tkinter import E
import numpy as np
import pandas as pd
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc
import random
from os.path import exists

random.seed(31415)
np.random.seed(31415)

target_name = 'target_with_noise'
instances = ['I.6.2',
'I.9.18',
'I.15.3x',
'I.30.5',
'I.32.17',
'I.41.16',
'I.48.2',
'II.6.15a',
'II.11.27',
'II.11.28',
'II.35.21',
# 'III.9.52', # no constraints available
'III.10.19',]

Degrees = [3,4,5,6,7]
Lambdas = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1]
Alphas = [0,0.5,1]
MaxInteractions = [2,3]

size = 2000
foldername = 'data/multivariate/1_gridsearch'
info_file = f'{foldername}/_info.csv';
result_file = f'data/multivariate/2_gridsearch_results/_results.csv'
if(exists(result_file)):
  message = f'every entry in "{result_file}" will not be retrained even if data changed, consider this'
  raise Exception(message)


# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

# for each target equation
for equation in equations:
  #store the original variable order to enable lambda call (position is relevant)
  variables = [var for var in equation['Variables']]
  lambda_variable_order = [var['name'] for var in equation['Variables']]
  eq = eval(equation['Formula_Lambda'])

  equation_name = equation['EquationName']
  equation_constraints = np.array(fc.constraints)[[ constraint['EquationName'] == equation_name for constraint in fc.constraints]][0]
  
  equation_constraints["TargetVariable"] = target_name
  equation_constraints["Degrees"] = Degrees
  equation_constraints["Lambdas"] = Lambdas
  equation_constraints["Alphas"] = Alphas
  equation_constraints["MaxInteractions"] = MaxInteractions
  equation_constraints["TrainTestSplit"] = 0.8

  with open(f'{foldername}/{equation_name}.json', 'w') as f:
    f.write(str(equation_constraints).replace('\'', '"'))

  X = np.random.uniform([var['low'] for var in equation['Variables']], [var['high'] for var in equation['Variables']], (size,len(equation['Variables'])))
  
  data = pd.DataFrame( X, columns=lambda_variable_order)
  data['target'] = [eq(row) for row in X]
  data[target_name] = ff.Noise(data['target'], noise_level=0.1) 
  data['equation_name'] = equation_name
  data['varied_variable_name'] = 'None'


  filename = f"{equation_name}"
  data.to_csv(f"{foldername}/{filename}.csv", index = False)