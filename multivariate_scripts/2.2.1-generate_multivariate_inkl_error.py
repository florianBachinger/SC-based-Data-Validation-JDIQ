import numpy as np
import pandas as pd
import Feynman.Functions as ff
from SyntheticError.DataTransformation import Multivariate as mv
import Feynman.Constraints as fc
import random
random.seed(31415)
np.random.seed(31415)

size = 2000
foldername = 'data/2.2.1-multivariate_error'

Degrees = [5]
Lambdas = [10**-3]
Alphas = [0.5]
MaxInteractions = [3]

target_with_errorVariable = 'target_with_error'
target_without_noiseVariable = 'target'
target_with_error_without_noise = 'target_with_error_without_noise'
target_with_noiseVariable = 'target_with_noise'
target_variable = target_with_errorVariable
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

# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

# perpare one meta dataset with all relevant information
df = pd.DataFrame( columns = ['EquationName','FileName','ErrorFunction', 'ConstraintsViolated','DataSize','ErrorWidthPercentage','NoiseLevelPercentage','ErrorScaling'])
i = 0 

# for each target equation
for equation in equations:
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

  with open(f'{foldername}/{equation_name}.json', 'w') as f:
    f.write(str(equation_constraints).replace('\'', '"'))

  for data_size in [200, 500,]:
    for error_width_percentage in [0.05, 0.075, 0.1, 0.125, 0.15]:
      for noise_level_percentage in [0.01, 0.02, 0.03, 0.05, 0.1]:
        for error_scaling_sigma in [0.5, 1, 1.5, 2, 3]:
            
            #generate uniform random input space
            X = np.random.uniform([var['low'] for var in equation['Variables']], [var['high'] for var in equation['Variables']], (data_size,number_inputs))
            data = pd.DataFrame( X, columns=lambda_variable_order)
            data["equation_name"] = [equation_name] * data_size
            
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
            


            #calculate equation and add to training data
            data[target_without_noiseVariable] = [eq(row) for row in X]
            sigma = np.std(data[target_without_noiseVariable])
            
                       
            
            #add noise
            data[target_with_noiseVariable] = ff.Noise(data[target_without_noiseVariable]
                                                      , noise_level=noise_level_percentage
                                                      , stdDev = sigma) 

            for error_function in ['None', 'Square', 'Spike', 'Normal']:
              data_error = data.copy()

              data_error = mv.ApplyErrorFunction(error_function_name = error_function
                                          ,data = data_error
                                          ,input_target_name = target_with_noiseVariable
                                          ,output_target_name = target_with_errorVariable
                                          ,error_function_variable_name = 'error_function'
                                          ,affected_space = shifted_space
                                          ,error_value = sigma
                                          ,returnErrorFunction=True)

              print(f"{equation_name.ljust(10)} {error_function.ljust(14)} {data_size} {error_width_percentage} {noise_level_percentage} {error_scaling_sigma}" )

              filename = f"{equation_name}_s{data_size}_n{noise_level_percentage}_es{error_scaling_sigma}_ew{error_width_percentage}_{error_function}"
              data_error.to_csv(f"{foldername}/{filename}.csv", index = False)
              df.loc[i] =[equation_name,filename,error_function, (error_function != 'None'), data_size,error_width_percentage, noise_level_percentage,error_scaling_sigma]
              i = i +1


df.to_csv(f"{foldername}/info/overview.csv", index = False)