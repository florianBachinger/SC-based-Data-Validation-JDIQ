import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import random
random.seed(31415)
np.random.seed(31415)

foldername = 'data/2.1.1-univariate_error'

Degrees = [4]
Lambdas = [10**-6]
Alphas = [0]
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
# 'II.11.27',
# 'II.11.28',
'II.35.21',
'III.10.19'
]

def Monotonic(gradients):
  gradients =np.array(gradients)
  gradients = gradients[~np.isnan(gradients)]
  sign_of_first = np.sign(gradients[0])

  if sign_of_first ==  -1:
      descriptor = "decreasing"
  if sign_of_first ==  0:
      descriptor = "constant"
  if sign_of_first ==  1:
      descriptor = "increasing"
  #do all gradients have the same sign
  if(len(np.unique(np.sign(gradients)))==1):
    return descriptor
  return 'none'



# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

# perpare one meta dataset with all relevant information
df = pd.DataFrame( columns = ['EquationName','FileName','Variable', 'ErrorFunction', 'ConstraintsViolated','DataSize','ErrorWidthPercentage','NoiseLevelPercentage','ErrorScaling'])
i = 0 

# for each target equation
for equation in equations:
  #store the original variable order to enable lambda call (position is relevant)
  lambda_variable_order = [var['name'] for var in equation['Variables']]
  eq = eval(equation['Formula_Lambda'])
  equation_name = equation['EquationName']
  equation_constraints = np.array(fc.constraints)[[constraint['EquationName'] == equation_name for constraint in fc.constraints]][0]
  
  equation_constraints["TargetVariable"] = target_variable
  equation_constraints["Degrees"] = Degrees
  equation_constraints["Lambdas"] = Lambdas
  equation_constraints["Alphas"] = Alphas
  equation_constraints["MaxInteractions"] = MaxInteractions
  equation_constraints["TrainTestSplit"] = 1

  with open(f'{foldername}/{equation_name}.json', 'w') as f:
    f.write(str(equation_constraints).replace('\'', '"'))

  for current_variable in equation['Variables']:
    for data_size in [500,2000]:
      for error_width_percentage in [0.01, 0.05, 0.1, 0.15, 0.2]:
        for noise_level_percentage in [0.01, 0.05, 0.1]:
          for error_scaling_sigma in [1, 2, 3]:
            # add metadata
            data = pd.DataFrame()

            #extract data
            varied_variable_name = current_variable['name']
            variable_constraints = np.array(equation_constraints['Constraints'])[[ ((var_constraint['name'] == varied_variable_name) & (var_constraint['order_derivative'] == 1)) for var_constraint in equation_constraints['Constraints']]][0]
            data["equation_name"] = [equation_name] * data_size

            # add uniform variable
            data[varied_variable_name] = np.random.uniform(current_variable['low'], current_variable['high'],data_size )
            data["varied_variable_name"] = [varied_variable_name] * data_size

            # all variables except the current one are set to their average value
            other_variables = [ var for var in equation['Variables'] if (var['name']!=varied_variable_name) ]
            generated_variables_names = [varied_variable_name]
            for other in other_variables:
              data[other['name']] = [((other['low']))] * data_size
            
            # reorder dataframe to fit lambda of equation
            generated_variables_names.append([var['name'] for var in other_variables])
            data = data[lambda_variable_order]

            #calculate equation and add to training data
            input = data.to_numpy()
            data[target_without_noiseVariable] = [eq(row) for row in input]

            #calculate error numbers
            sigma = np.std(data[target_without_noiseVariable])
            error_value  = sigma * error_scaling_sigma
            error_length = int(data_size) * error_width_percentage
            error_start = random.randint(0, data_size - error_length - 1 )
            error_end = error_start+ error_length

            #add noise
            data[target_with_noiseVariable] = ff.Noise(data[target_without_noiseVariable]
                                                      , noise_level=noise_level_percentage
                                                      , stdDev = sigma) 

            data = data.sort_values(by=[varied_variable_name])
            data = data.reset_index(drop=True)

            for error_function in ['None', 'Square', 'Spike', 'Normal']:
              data_error = data.copy()
              data_error[target_with_errorVariable],data_error['error_function'] = ef.ApplyErrorFunction(error_function
                                                                                                        ,data[target_with_noiseVariable]
                                                                                                        ,start=error_start
                                                                                                        ,end=error_end
                                                                                                        ,error_value= error_value
                                                                                                        ,returnErrorFunction=True)
              data_error[target_with_error_without_noise] = data[target_without_noiseVariable] + (data_error['error_function'] * error_value)

              print(f"{equation_name.ljust(10)} {varied_variable_name.ljust(8)} {error_function.ljust(14)} {data_size} {error_width_percentage} {noise_level_percentage} {error_scaling_sigma}" )

              filename = f"{equation_name}_{varied_variable_name}_{error_function}_s{data_size}_n{noise_level_percentage}_es{error_scaling_sigma}_ew{error_width_percentage}"
              data_error.to_csv(f"{foldername}/{filename}.csv", index = False)
              df.loc[i] =[equation_name,filename,varied_variable_name,error_function, (error_function != 'None'), data_size,error_width_percentage, noise_level_percentage,error_scaling_sigma]
              i = i +1

df.to_csv(f"{foldername}/info/overview.csv", index = False)