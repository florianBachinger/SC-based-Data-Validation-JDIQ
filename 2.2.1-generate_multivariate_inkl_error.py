import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc

size = 2000
foldername = 'data/2.1.1-univariate_error'

Degrees = [5]
Lambdas = [10**-3]
Alphas = [0.5]
MaxInteractions = [3]

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
df = pd.DataFrame( columns = ['EquationName','FileName','Variable', 'ErrorFunction', 'ConstraintsViolated'])
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

  X = np.random.uniform([var['low'] for var in equation['Variables']], [var['high'] for var in equation['Variables']], (size,len(equation['Variables'])))
  data = pd.DataFrame( X, columns=lambda_variable_order)

df.to_csv(f"{foldername}/info/overview.csv", index = False)