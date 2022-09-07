import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc

target_with_errorVariable = 'target_with_error'
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

Degrees = [3]
Lambdas = [10**-7]
Alphas = [0]
MaxInteractions = [3]

size = 2000
foldername = 'data/univariate_error'

# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]

# perpare one meta dataset with all relevant information
df = pd.DataFrame( columns = ['EquationName','FileName','Variable', 'ErrorFunction', 'ErrorDetectable'])
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

  with open(f'{foldername}/{equation_name}.json', 'w') as f:
    f.write(str(equation_constraints).replace('\'', '"'))

  for current_variable in equation['Variables']:
    varied_variable_name = current_variable['name']
    variable_constraints = np.array(equation_constraints['Constraints'])[[ ((var_constraint['name'] == varied_variable_name) & (var_constraint['order_derivative'] == 1)) for var_constraint in equation_constraints['Constraints']]][0]
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
    data[target_with_noiseVariable] = ff.Noise([eq(row) for row in input], noise_level=0.1) 
    
    data["equation_name"] = [equation_name] * size
    data["varied_variable_name"] = [varied_variable_name] * size

    data = data.sort_values(by=[varied_variable_name])
    data = data.reset_index(drop=True)

    for error_function in ['NONE', 'Square','Spike','RampUp','RampDown','ExponentialUp','Normal']:
      data_error = data.copy()
      if(error_function == 'Square'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.Square(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.Square(data['target'],start=size*0.8,end=size)

      if(error_function == 'Spike'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.Spike(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.Spike(data['target'],start=size*0.8,end=size)

      if(error_function == 'RampUp'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.RampUp(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.RampUp(data['target'],start=size*0.8,end=size)

      if(error_function == 'RampDown'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.RampDown(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.RampDown(data['target'],start=size*0.8,end=size)

      if(error_function == 'ExponentialUp'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.ExponentialUp(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.ExponentialUp(data['target'],start=size*0.8,end=size)

      if(error_function == 'Normal'):
        data_error[target_with_errorVariable],data_error['error_function'] = ef.Normal(data[target_with_noiseVariable],start=size*0.8,end=size,returnErrorFunction=True)
        data_error['target_with_error_without_noise'] = ef.Normal(data['target'],start=size*0.8,end=size)

      if(error_function == 'NONE'):
        data_error[target_with_errorVariable] = data[target_with_noiseVariable]
        data_error['target_with_error_without_noise'] = data['target']

      derivedConstraint = variable_constraints['monotonicity']
      measuredConstraint = Monotonic(data_error['target'].diff())
      errorConstraint = Monotonic(data_error['target_with_error_without_noise'].diff())
      detectable = (derivedConstraint != errorConstraint) | (derivedConstraint != measuredConstraint)
      
      print(f"{equation_name.ljust(10)} {varied_variable_name.ljust(8)} {variable_constraints['name'].ljust(8)} {error_function.ljust(14)} " 
       +f"derived: {derivedConstraint.ljust(10)} measured: {measuredConstraint.ljust(10)} withError: {errorConstraint.ljust(10)} detectable: {detectable}")

      filename = f"{equation_name}_{varied_variable_name}_{error_function}"
      data_error.to_csv(f"{foldername}/{filename}.csv", index = False)
      df.loc[i] =[equation_name,filename,varied_variable_name,error_function, detectable]
      i = i +1

df.to_csv(f"{foldername}/info/overview.csv", index = False)