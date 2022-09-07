from xmlrpc.client import Transport
import numpy as np 
import pandas as pd
from Feynman import Functions as ff
from scipy.stats import multivariate_normal

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

def printAttributes(df):
    print(f"theta-max {np.max(df['theta'])} min {np.min(df['theta'])}")
    print(f"sigma-max {np.max(df['sigma'])} min {np.min(df['sigma'])}")
    print(f"f-max {np.max(df['f'])} min {np.min(df['f'])} avg {np.average(df['f'])} std {np.std(df['f'])} ")

######### generate dataset #########
df = ff.Feynman12.generate_df(size = 500)

g = sns.PairGrid(df, diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.scatterplot, s=15)
g.map_diag(my_hist)

plt.savefig('multivariate.png')
plt.clf()

# df = df.sort_values(by=['Ef', 'q2',])
filter = (((df['q2']>=4) & (df['q2']<=5)) &
        ((df['Ef']>=4) & (df['Ef']<=5)))
filter_others = [ (not val) for val in  filter] 

g = sns.PairGrid(df[filter], diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.scatterplot, s=15)
g.map_diag(my_hist)

plt.savefig('multivariate_filtered.png')
plt.clf()


rv = multivariate_normal([4.5,4.5], [[0.2,0], [0,0.2]])
error = rv.pdf(df[filter][['Ef','q2']])

df.loc[filter, 'F'] = df[filter]['F'] + (df[filter]['F'] * error)
df.loc[filter, 'error'] = error
print(df[filter][['Ef','q2','error']])

g = sns.PairGrid(df[filter], diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.scatterplot, s=15)
g.map_diag(my_hist)

plt.savefig('multivariate_filtered_error.png')
plt.clf()

g = sns.PairGrid(df, diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.scatterplot, s=15)
g.map_diag(my_hist)

plt.savefig('multivariate_error.png')
plt.clf()

