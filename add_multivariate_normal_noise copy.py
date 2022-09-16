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

plt.savefig('multivariate_without_error.png')
plt.clf()
                        #center     #
rv = multivariate_normal([4.5,3.5], [[0.07,0], [0,0.07]])
error = rv.pdf(df[['Ef','q2']])

df['error'] = error

g = sns.PairGrid(df, diag_sharey = False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.scatterplot, s=15)
g.map_diag(my_hist)

plt.savefig('multivariate_with_error.png')
plt.clf()


