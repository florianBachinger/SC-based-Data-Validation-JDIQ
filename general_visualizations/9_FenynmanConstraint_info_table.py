import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc

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

# take only the equations specified above
filter = [ np.any( [item['DescriptiveName'].endswith(name) for name in instances] ) for item in ff.FunctionsJson]
equations = np.array(ff.FunctionsJson)[filter]



for equation in equations:
    instance = [name for name in instances if equation['DescriptiveName'].endswith(name)][0]
    first = True
    equation_name = equation['EquationName']
    lambda_variable_order = [var['name'] for var in equation['Variables']]
    equation_constraints = np.array(fc.constraints)[[ constraint['EquationName'] == equation_name for constraint in fc.constraints]][0]['Constraints']

    for current_variable in equation['Variables']:
      variable_name = current_variable['name']
      variable_constraints = np.array(np.array(equation_constraints)[[ constraint['name'] == variable_name for constraint in equation_constraints]])
      first_order = variable_constraints[[ c['order_derivative'] == 1 for c in variable_constraints]][0]
      second_order = variable_constraints[[ c['order_derivative'] == 2 for c in variable_constraints]][0]
      
      print(f"{instance if first else ''} & $\mathit{{{variable_name}}} \in [{current_variable['low']},{current_variable['high']}]$ & {first_order['descriptor']}& {second_order['descriptor']} \\\\ ")
      first = False