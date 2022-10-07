import numpy as np
import pandas as pd
import Feynman.Functions as ff
import Feynman.Constraints as fc
import random
random.seed(31415)
np.random.seed(31415)

target_name = 'target_with_noise'
instances = [
'I.6.2',
'I.9.18', 
'I.15.3x', 
'I.30.5', 
'I.32.17',
'I.41.16', 
'I.48.20', 
'II.6.15a',
'II.35.21',
'III.10.19'
]

Degrees = [3,4,5,6,7]
Lambdas = [10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1]
Alphas = [0,0.5,1]
MaxInteractions = [2,3]

size = 2000
foldername = 'data/1.1.1-univariate_gridsearch'

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
  equation_constraints["TargetVariable"] = target_name
  equation_constraints["Degrees"] = Degrees
  equation_constraints["Lambdas"] = Lambdas
  equation_constraints["Alphas"] = Alphas
  equation_constraints["MaxInteractions"] = MaxInteractions
  equation_constraints["TrainTestSplit"] = 0.8

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
      data[other['name']] = [other['low']] * size
    
    # reorder dataframe to fit lambda of equation
    generated_variables_names.append([var['name'] for var in other_variables])
    data = data[lambda_variable_order]

    #calculate equation and add to training data
    input = data.to_numpy()
    data['target'] = [eq(row) for row in input]
    # 10% noise
    data[target_name] = ff.Noise([eq(row) for row in input], noise_level=0.1) 
    
    data["equation_name"] = [equation_name] * size
    data["varied_variable_name"] = [varied_variable_name] * size

    filename = f"{equation_name}_{varied_variable_name}"
    data.to_csv(f"{foldername}/{filename}.csv", index = False)

    df.loc[df_row] = [
      equation_name,
      filename,
      varied_variable_name,
      current_variable,
      str(other_variables)
    ]
    df_row = df_row + 1

df.to_csv(f'{foldername}/info/info.csv')