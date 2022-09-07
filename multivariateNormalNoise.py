from xmlrpc.client import Transport
import numpy as np 
import pandas as pd
from scipy.stats import multivariate_normal
from Feynman import Functions as ff

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


df = np.random.multivariate_normal([0,0,0], [[1,0,0],[0,1,0],[0,0,1]], 1000)
df_multVarNor = pd.DataFrame(df,columns=['a','b','c'])
g = sns.PairGrid(df_multVarNor, diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot, fill = True)
g.map_diag(my_hist)

plt.show(block = True)

a = np.random.uniform(-2,2,1000)
b = np.random.uniform(-2,2,1000)
data = list(zip(a,b))
rv = multivariate_normal([0,0], [[0.2,0], [0,0.2]])
df = pd.DataFrame(list(zip(a,b,rv.pdf(data))),columns=['a','b','c'])

g = sns.PairGrid(df, diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot, fill = True)
g.map_diag(my_hist)

plt.show(block = True)