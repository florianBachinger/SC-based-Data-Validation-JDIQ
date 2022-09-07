import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from os import walk

def my_hist(x, label, color):
    ax0 = plt.gca()
    ax = ax0.twinx()
    
    sns.despine(ax=ax, left=True, top=True, right=False)
    ax.yaxis.tick_right()
    ax.set_ylabel('Counts')
    
    ax.hist(x, label=label, color=color)

# df = pf.Feynman2(1,3,1000)

# print('plotting')

path = 'D:\src\documents\Publications\\2021\Bachinger_DataValidation\data\miba-full\indiv'
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break


df = pd.read_excel('D:\src\documents\Publications\\2021\Bachinger_DataValidation\data\miba-full\predicted_classification.xlsx')

for index, row in df.iterrows():
  frictionTestNr = row['FrictionTestNr']
  predicted = row['PredictedClassification']
  actual = row['Classification']
  if(actual == 'error'):
    actual = 'invalid'
  matching = [s for s in f if frictionTestNr in s]
  if(len(matching) == 1):
    match = matching[0]
    df = pd.read_csv(f'D:\src\documents\Publications\\2021\Bachinger_DataValidation\data\miba-full\indiv\{match}')

    df = df[['p_spec','v_max','Tdisc_min','cf_dyn']]

    # HL Like
    # g = sns.pairplot(df)

    # HL Like (better)
    g = sns.PairGrid(df, diag_sharey = False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot, fill = True)
    g.map_diag(my_hist)


    plt.savefig(f'fig/MultiScatter/{actual}_{match}.png',dpi=600, transparent=True)
    plt.clf()
    plt.close()
    # plt.show()