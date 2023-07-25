import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import Normalize
import seaborn as sns

univariate_result_file = f'data/univariate/2_gridsearch_results/_results.csv'
uni = pd.read_csv(univariate_result_file)

multivariate_result_file = f'data/multivariate/2_gridsearch_results/_results.csv'
multi = pd.read_csv(multivariate_result_file)

result_figure = 'figures/experimental_setup/gridsearch_feynman.pdf'

problemTypes = [('univariate', uni),('multivariate',multi)]



datasets = np.unique(uni['EquationName'])
lambda_values =  np.unique( uni['Lambda'])
degree_values =  np.unique( uni['Degree'])
alpha_values = np.unique( uni['Alpha'])
max_interactions_values = np.unique( uni['MaxInteractions'])

rmse_column = 'RMSE_Full'


f, axis = plt.subplots(2,3, figsize=(6,3),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1,1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.16, bottom=0.2, right=0.84, top=0.99, wspace=0.1, hspace=0.1)

axisIndex = 0
for (problemtype,df) in problemTypes:
  ax = axis[axisIndex]
  

  df = df[df['Lambda'] < 1]
  df = df[df['MaxInteractions'] == 3]


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

  rmseSums = df[['Lambda', 'Degree', 'Alpha', 'MaxInteractions',rmse_column ]
                ].groupby(['Lambda', 'Degree', 'Alpha', 'MaxInteractions']).sum()

  print(rmseSums.nsmallest(10, rmse_column))

  min = np.min(rmseSums)
  max = np.max(rmseSums)
  norm = Normalize(vmin=min, vmax=max)
  cmap = sns.color_palette("mako_r", as_cmap=True,)
  cbar_ax = f.add_axes([.85, .62 - (axisIndex * 0.41), .02, .35])


  col = 0
  for Alpha in np.unique( df['Alpha']):
    df_range = df[(df['Alpha']==Alpha)]

    
    pivot = df_range.pivot_table(index='Lambda', columns='Degree',
                            values=rmse_column, aggfunc=np.sum)
    hm = sns.heatmap(data=pivot, ax=ax[col], norm=norm, cmap=cmap, cbar_ax=cbar_ax , cbar_kws=dict(ticks=[round(val,2) for val in [min, min + (max-min)/2, max]]))


    if(axisIndex == 1):
      ax[col].set_xlabel(f'd\n$\\alpha = {Alpha}$')
      if(col == 2):
        ax[col].set_xlabel(f'd - Degree\n$\\alpha = {Alpha}$')

    
    else:
      ax[col].set_xlabel(f'')

    if(col > 0):
      ax[col].set_ylabel("")
    else:
      ax[col].set_ylabel(f'{problemtype}\n$\lambda$')

    col = col+1

  axisIndex = axisIndex + 1

  cbar_ax.set_ylabel('RMSE')
  cbar_ax.yaxis.set_label_position('right') 

plt.savefig(result_figure,dpi=600, transparent=True)
plt.show()