# to make plot interactive
from operator import invert
import numpy as np
from scipy.integrate import simpson
from numpy import trapz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import warnings
import matplotlib.cm as cm
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



font = {'family' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

curves_results = pd.read_csv('data/univariate/7-rocCurveResults.csv')

# perpare one meta dataset with all relevant information
df = pd.DataFrame(columns=['DataSize','ErrorWidthPercentage','NoiseLevelPercentage','ErrorScaling','AUC'])
i = 0 

for data_size in np.unique(curves_results['DataSize']):
  for error_width_percentage in np.unique(curves_results['ErrorWidthPercentage']):
    for noise_level_percentage in np.unique(curves_results['NoiseLevelPercentage']):
      for error_scaling_sigma in np.unique(curves_results['ErrorScaling']):
        filtered = curves_results
        filtered = filtered[filtered['DataSize'] == data_size]
        filtered = filtered[filtered['ErrorWidthPercentage'] == error_width_percentage]
        filtered = filtered[filtered['NoiseLevelPercentage'] == noise_level_percentage]
        filtered = filtered[filtered['ErrorScaling'] == error_scaling_sigma]


        curveData = filtered
        lineData = curveData[['false positive rate','true positive rate']]
        lineData = lineData.drop_duplicates()
        # lineData['false positive rate'] = curveData['false positive rate'].map(lambda item 
        #                                           : 0.0000000001 if item == 0.0 else item)
        idx = lineData.groupby(['false positive rate'])['true positive rate'].transform(max) == lineData['true positive rate']
        lineData = lineData[idx]
        idx = lineData.groupby(['true positive rate'])['false positive rate'].transform(min) == lineData['false positive rate']
        lineData = lineData[idx]
        defaultStart = { 'true positive rate': 0.0,
                    'false positive rate': 0.0}
        defaultEnd = { 'true positive rate': 1,
                    'false positive rate': 1}
        lineData = lineData.append(defaultStart, ignore_index = True)
        lineData = lineData.append(defaultEnd, ignore_index = True)

        lineData = lineData.sort_values(by=['false positive rate','true positive rate'], ascending=True)

        area = np.trapz(lineData['true positive rate'], x=lineData['false positive rate'])

        df.loc[i] =[data_size,error_width_percentage,noise_level_percentage,error_scaling_sigma, area]
        i = i +1

df.to_csv('data/univariate/8-AUC_comparison.csv')
