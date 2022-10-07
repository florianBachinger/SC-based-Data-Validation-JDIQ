from xmlrpc.client import Transport
import numpy as np 
import pandas as pd
from Feynman import Functions as ff
from scipy.stats import multivariate_normal
import Feynman.Constraints as fc

import matplotlib.pyplot as plt
import seaborn as sns
from os import walk

from SyntheticError.DataTransformation import Multivariate as mv

import random
random.seed(31415)
np.random.seed(31415)

def my_hist(x, label, color):
    ax0 = plt.gca()
    ax = ax0.twinx()
    
    sns.despine(ax=ax, left=True, top=True, right=False)
    ax.yaxis.tick_right()
    ax.set_ylabel('Counts')
    
    ax.hist(x, label=label, color=color)


######### generate dataset #########
plt.savefig('multivariate_without_error.png')
plt.clf()

Degrees = [5]
Lambdas = [10**-3]
Alphas = [0.5]
MaxInteractions = [3]

target_with_errorVariable = 'target_with_error'
target_without_noiseVariable = 'target'
target_with_error_without_noise = 'target_with_error_without_noise'
target_variable = target_without_noiseVariable
instances = [
'I.6.2',
'I.9.18', 
'I.15.3x', 
'I.30.5', 
'I.32.17',
'I.41.16', 
'I.48.20', 
'II.6.15a',
'II.11.27',
'II.11.28',
'II.35.21',
'III.10.19'
]
data_size = 1000

# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

for equation in equations[0:1]:
  number_inputs = len(equation['Variables'])
  #store the original variable order to enable lambda call (position is relevant)
  lambda_variable_order = [var['name'] for var in equation['Variables']]
  eq = eval(equation['Formula_Lambda'])
  equation_name = equation['EquationName']
  equation_constraints = np.array(fc.constraints)[[constraint['EquationName'] == equation_name for constraint in fc.constraints]][0]
  


  equation_constraints["AllowedInputs"] = "all"
  equation_constraints["TargetVariable"] = target_variable
  equation_constraints["Degrees"] = Degrees
  equation_constraints["Lambdas"] = Lambdas
  equation_constraints["Alphas"] = Alphas
  equation_constraints["MaxInteractions"] = MaxInteractions
  equation_constraints["TrainTestSplit"] = 1


  for error_width_percentage in [0.1]:
    for error_scaling_sigma in [2]:
      
      #generate uniform random input space
      X = np.random.uniform([var['low'] for var in equation['Variables']], [var['high'] for var in equation['Variables']], (data_size,number_inputs))
      data = pd.DataFrame( X, columns=lambda_variable_order)
      data["equation_name"] = [equation_name] * data_size

      #calculate equation and add to training data
      data[target_without_noiseVariable] = [eq(row) for row in X]
      sigma = np.std(data[target_without_noiseVariable])
      g = sns.PairGrid(data, diag_sharey = False)
      g.map_upper(sns.scatterplot, s=15)
      g.map_lower(sns.scatterplot, s=15)
      g.map_diag(my_hist)

      plt.savefig('multivariate_without_error.png')
      plt.clf()

      #calculated affected data in space
      dimension_data = [{
        'name': (var['name']),
        'low': (var['low']),
        'high' : var['high'],
        'width' : (var['high'] - var['low']),
        'center' : (var['low'] + (var['high'] - var['low'])/2) 
        } for var in equation['Variables']]

      affected_space_width = mv.CalculateAffectedSpace(data_size,dimension_data,error_width_percentage)
      shifted_space = mv.ShiftAffectedSpace( dimension_data, affected_space_width)
            

      for error_function in ['None', 'Square', 'Spike', 'Normal']:
        data_error = data.copy()

        data_error = mv.ApplyErrorFunction(error_function_name = error_function
                                    ,data = data_error
                                    ,input_target_name = target_variable
                                    ,output_target_name = target_with_errorVariable
                                    ,error_function_variable_name = 'error_function'
                                    ,affected_space = shifted_space
                                    ,error_value = sigma
                                    ,returnErrorFunction=True)
                              #center     #

        g = sns.PairGrid(data_error, diag_sharey = False)
        g.map_upper(sns.scatterplot, s=15)
        g.map_lower(sns.scatterplot, s=15)
        g.map_diag(my_hist)

        plt.savefig(f'multivariate_with_error_ewp{error_width_percentage}_es{error_scaling_sigma}_ef{error_function}.png')
        plt.clf()
        plt.close()


