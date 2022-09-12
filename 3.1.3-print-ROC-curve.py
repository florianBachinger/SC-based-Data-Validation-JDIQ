from operator import invert
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

curves_results = pd.read_csv('3.1.2-rocCurveResults.csv')
plt.figure(figsize=(10,8))

# ###################  adapt true positive rate ################### per false positive rate
# (FPR) - the x axis of the ROC curve - keep only the highest true positive rate this is caused
# by the variations in t that result in same false positives but lower true positives therefore
# we only want the one single t value that yields the highest true positive rate result for a
# given false positive rate

idx = curves_results.groupby(['false positive rate'])['true positive rate'].transform(max) == curves_results['true positive rate']
curveData = curves_results[idx]

lineData = curveData
lineData['false positive rate'] = curveData['false positive rate'].map(lambda item 
                                          : 0.0000000001 if item == 0.0 else item)
default = { 'true positive rate': 0.0,
            'false positive rate': 0.0}
lineData = lineData.append(default, ignore_index = True)
plt.step( lineData['false positive rate'],
          lineData['true positive rate'])     
  
plt.xlabel('false positive rate\npredicted invalid - actually valid')
plt.ylabel('true positive rate\npredicted invalid and actually invalid')
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100), c= 'grey', linestyle = '--')
plt.legend()

plt.savefig('fig/roc.png', transparent = True, dpi = 600)
plt.show()               