import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt
from ast import literal_eval

data = pd.read_csv('3.1.1_training_error.csv')
knownInvalidCount = np.sum(data['ConstraintsViolated']==True)
knownValidCount = np.sum(data['ConstraintsViolated']==False)
rows_list = []

data['RMSE'] = [literal_eval(x) for x in data['RMSE']]

exceeding_count_threshold = 1
i = 0
length = 1000
for border in np.linspace(np.min([np.min(x) for x in data['RMSE']]), np.max([np.max(x) for x in data['RMSE']]), length):
  print(f"row {i}/{length}")
  true_negatives = 0
  true_positives = 0
  false_negatives = 0
  false_positives = 0
  for (rowid, row ) in data.iterrows():
    count_exceeding = np.sum( (row['RMSE']>=border) )

    #over border, actual constraints violated
    if( (count_exceeding >= exceeding_count_threshold) & 
        (row['ConstraintsViolated']==True)):
      true_positives = true_positives + 1

    #all under border, no constraints violated
    if( (count_exceeding == 0) & 
        (row['ConstraintsViolated']==False)):
      true_negatives = true_negatives + 1 

    #all under border, actual constraints violated
    if( (count_exceeding >= exceeding_count_threshold) & 
        (row['ConstraintsViolated']==False)):
      false_positives = false_positives + 1

    #over border, no constraints violated
    if( (count_exceeding == 0) & 
        (row['ConstraintsViolated']==True)):
      false_negatives = false_negatives + 1

  tPR  = float(true_positives)/knownInvalidCount  
  fPR = 1-(float(true_negatives)/knownValidCount)
  if((tPR>1) or (tPR<0)):
    print('PROBLEM1')
  if((fPR>1) or (fPR<0)):
    print('PROBLEM2')

  rows_list.append({
    'border':border,
    'true positive rate' : tPR,
    'false positive rate' : fPR
  })
  i = i + 1

curves_results = pd.DataFrame(rows_list)               
curves_results.to_csv('3.1.3-rocCurveResults.csv', index = False)

print(f"knownInvalidCount: {knownInvalidCount} knownValidCount:{knownValidCount}")