import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc

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
    return descriptor.ljust(10)
  return 'none'.ljust(10)

Degrees = [3,4,5,6,7,8,9]
Lambdas = [10**-5,10**-4,10**-3,10**-2,10**-1,1]
Alphas = [0,0.5,1]
MaxInteractions = [2,3]

size = 20
foldername = 'data/univariate_error'

# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

# perpare one meta dataset with all relevant information
df = pd.DataFrame( columns = ['EquationName','FileName','Variable','OtherVariables', 'OtherVariables_Data'])
df_row = 0 

# for each target equation
for equation in equations:
  #store the original variable order to enable lambda call (position is relevant)
  lambda_variable_order = [var['name'] for var in equation['Variables']]
  eq = eval(equation['Formula_Lambda'])
  equation_name = equation['EquationName']
  equation_constraints = np.array(fc.constraints)[[ constraint['EquationName'] == equation_name for constraint in fc.constraints]][0]
  
  equation_constraints["Degrees"] = Degrees
  equation_constraints["Lambdas"] = Lambdas
  equation_constraints["Alphas"] = Alphas
  equation_constraints["MaxInteractions"] = MaxInteractions

  with open(f'{foldername}/{equation_name}.json', 'w') as f:
    f.write(str(equation_constraints).replace('\'', '"'))

  for current_variable in equation['Variables']:
    varied_variable_name = current_variable['name']
    # add metadata
    data = pd.DataFrame()


    # add uniform variable
    data[varied_variable_name] = np.random.uniform(current_variable['low'], current_variable['high'],size )

    # all variables except the current one are set to their average value
    other_variables = [ var for var in equation['Variables'] if (var['name']!=varied_variable_name) ]
    generated_variables_names = [varied_variable_name]
    for other in other_variables:
      data[other['name']] = [((other['low']))] * size
    
    # reorder dataframe to fit lambda of equation
    generated_variables_names.append([var['name'] for var in other_variables])
    data = data[lambda_variable_order]

    #calculate equation and add to training data
    input = data.to_numpy()
    data['target'] = [eq(row) for row in input]
    # 10% noise
    data['target_with_noise'] = ff.Noise([eq(row) for row in input], noise_level=0.1) 
    
    data["equation_name"] = [equation_name] * size
    data["varied_variable_name"] = [varied_variable_name] * size

    data = data.sort_values(by=[varied_variable_name])
    data = data.reset_index(drop=True)
    # print(data)

    for error_config in [
      # 'Square','Spike','RampUp','RampDown','ExponentialUp','Normal'
      'None'
    ]:
      data_error = data.copy()
      if(error_config == 'Square'):
        data_error['target_with_error'],data_error['error_function'] = ef.Square(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'Spike'):
        data_error['target_with_error'],data_error['error_function'] = ef.Spike(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'RampUp'):
        data_error['target_with_error'],data_error['error_function'] = ef.RampUp(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'RampDown'):
        data_error['target_with_error'],data_error['error_function'] = ef.RampDown(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'ExponentialUp'):
        data_error['target_with_error'],data_error['error_function'] = ef.ExponentialUp(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'Normal'):
        data_error['target_with_error'],data_error['error_function'] = ef.Normal(data['target'],start=size*0.8,end=size,returnErrorFunction=True)
      if(error_config == 'None'):
        data_error['target_with_error'] = data['target']

      print(f'{equation_name.ljust(10)} {varied_variable_name.ljust(8)} {error_config} target with error 1 { Monotonic(data_error["target_with_error"].diff()) } target with error 2 { Monotonic(data_error["target_with_error"].diff().diff()) } target 1 { Monotonic(data_error["target"].diff()) } target 2 { Monotonic(data_error["target"].diff().diff()) }')


      filename = f"{equation_name}_{varied_variable_name}_{error_config}"
      data_error.to_csv(f"{foldername}/{filename}.csv", index = False)