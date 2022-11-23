from operator import invert
import numpy as np
import pandas as pd
from pyparsing import line
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

curves_results = pd.read_csv('3.2.4-rocCurveResults.csv')
plt.figure(figsize=(10,8))


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
        
        lineData = lineData.sort_values(by='false positive rate', ascending=True)

        plt.close()
        plt.clf()

        plt.step( lineData['false positive rate'],
                  lineData['true positive rate'])     
          
        plt.xlabel('false positive rate\npredicted invalid - actually valid')
        plt.ylabel('true positive rate\npredicted invalid and actually invalid')
        plt.plot(np.linspace(0,1,100),np.linspace(0,1,100), c= 'grey', linestyle = '--')

        plt.savefig(f'fig/3.2.4-roc_size{data_size}_errWid{error_width_percentage}_noiseLev{noise_level_percentage}_errScal{error_scaling_sigma}.png', transparent = True, dpi = 600)
    