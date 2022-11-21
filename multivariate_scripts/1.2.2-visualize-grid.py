import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

df = pd.read_csv('1.2.1-result_multivariate.csv')

datasets = np.unique(df['EquationName'])
lambda_values =  np.unique( df['Lambda'])
degree_values =  np.unique( df['Degree'])
alpha_values = np.unique( df['Alpha'])
max_interactions_values = np.unique( df['MaxInteractions'])

rmse_column = 'RMSE_Full'

max_training_rmse = np.max(df['RMSE_Training'])
max_test_rmse = np.max(df['RMSE_Test'])
max_rmse = np.max(df['RMSE_Full'])

print(df)
df = pd.DataFrame(df)
df.loc[df['RMSE_Training'] == -1 ,'RMSE_Training'] = max_training_rmse
df.loc[df['RMSE_Test'] == -1 ,'RMSE_Test'] = max_test_rmse

df.loc[df['RMSE_Full'] == -1 ,'RMSE_Full'] = max_rmse


i = len(df.index)+1
for dataset in datasets:
  for lambda_value in lambda_values:
    for degree in degree_values:
      for alpha_value in alpha_values:
        for interaction_value in max_interactions_values:
          if( len(df[ ((df['EquationName']==dataset) & 
                  (df['Lambda']==lambda_value) & 
                  (df['Degree']==degree) & 
                  (df['Alpha']==alpha_value) & 
                  (df['MaxInteractions']==interaction_value)) ]) == 0  ):
            df.loc[i] =[dataset, None, 'No Path', degree, lambda_value, alpha_value, interaction_value, False,0, max_training_rmse, max_test_rmse, max_rmse]
            i = i +1

f, ax = plt.subplots(len(max_interactions_values),len(alpha_values),  sharex=True, sharey=True, figsize=(7,4.5))
plt.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.924, wspace=0.05, hspace=0.1)

rmseSums = df[['Lambda', 'Degree', 'Alpha', 'MaxInteractions',rmse_column ]
              ].groupby(['Lambda', 'Degree', 'Alpha', 'MaxInteractions']).sum()

print(rmseSums.nsmallest(10, rmse_column))

norm = Normalize(vmin=np.min(rmseSums), vmax=np.max(rmseSums))
cmap = sns.color_palette("magma_r", as_cmap=True,)
cbar_ax = f.add_axes([.86, .4, .02, .40])

row = 0
for maxInteractions in np.unique( df['MaxInteractions']):
  col = 0
  for Alpha in np.unique( df['Alpha']):
    df_range = df[(df['Alpha']==Alpha) & (df['MaxInteractions']==maxInteractions)]

    
    pivot = df_range.pivot_table(index='Lambda', columns='Degree',
                           values=rmse_column, aggfunc=np.sum)
    hm = sns.heatmap(data=pivot, ax=ax[row,col], norm=norm, cmap=cmap, cbar_ax=cbar_ax)

    if(row==(len(np.unique( df['MaxInteractions']))-1)):
      ax[row,col].set_xlabel(f'{ax[row,col].get_xlabel()}\n\nAlpha: {Alpha}')
    else:
      ax[row,col].set_xlabel('')

    if(col==0):
      ax[row,col].set_ylabel(f'MaxInteractions: {maxInteractions}\n\n{ax[row,col].get_ylabel()}')
      ax[row,col].set_yticklabels(ax[row,col].get_yticks(), rotation = 0)
    else:
      ax[row,col].set_ylabel('')
    col = col+1
  row = row+1

cbar_ax.set_ylabel('RMSE')
cbar_ax.yaxis.set_label_position('right') 
plt.savefig('gridsearch.png',dpi=600, transparent=True)
plt.show()