import numpy as np
import pandas as pd
import Feynman.Functions as ff
import SyntheticError.DataTransformation as ef
import Feynman.Constraints as fc
import matplotlib.pyplot as plt


data = pd.read_csv('3.1.1_training_error.csv')
knownInvalidCount = np.sum(data['ConstraintsViolated']==True)
knownValidCount = np.sum(data['ConstraintsViolated']==False)
rows_list = []


i = 0
length = 1000
for border in np.linspace(np.min(data['RMSE']), np.max(data['RMSE']), length):
  print(f"row {i}/{length}")
  #under border, no constraints violated
  true_negatives = data[  (data['RMSE']<border) &
                                    (data['ConstraintsViolated']==False)
                                ]
  num_true_negatives = len(true_negatives)
  #over border, actual constraints violated
  true_positives = data[  (data['RMSE']>=border) &
                                  (data['ConstraintsViolated']==True)
                              ]
  num_true_positives = len(true_positives)
  #over border, no constraints violated
  false_negatives = data[  (data['RMSE']>=border) &
                                  (data['ConstraintsViolated']==False)
                              ]
  num_false_negatives = len(false_negatives)
  #under exceeded, actual constraints violated
  false_positives = data[  (data['RMSE']<border) &
                                  (data['ConstraintsViolated']==True)
                              ]

  num_false_positives = len(false_positives)

  if(len(data)!=(num_true_negatives+num_true_positives+num_false_negatives+num_false_positives)):
    print('wrong confusion matrix calculation')

  tPR  = float(num_true_positives)/knownInvalidCount  
  fPR = 1-(float(num_true_negatives)/knownValidCount)
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
curves_results.to_csv('3.1.2-rocCurveResults.csv', index = False)

print(f"knownInvalidCount: {knownInvalidCount} knownValidCount:{knownValidCount}")