import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

AUC_multi_file = f'data/multivariate/8-AUC_comparison.csv'
AUC_uni_file = f'data/univariate/8-AUC_comparison.csv'
result_figure1 = 'figures/experimental_setup/AUC-Heatmap.pdf'
result_figure2 = 'figures/experimental_setup/AUC-Heatmap-largerSize.pdf'

AUC_multi = pd.read_csv(AUC_multi_file)
AUC_uni = pd.read_csv(AUC_uni_file)
# df = df[df['ErrorScaling'] == 0.5]
# df = df[df['DataSize'] == 200]


combi = [(result_figure1, [(AUC_multi, 'multivariate', 200), (AUC_uni, 'univariate', 50)]), 
            (result_figure2,[(AUC_multi, 'multivariate', 500), (AUC_uni, 'univariate', 100)])]
    
for (result_figure, results) in combi:
  f, axis = plt.subplots(2,3,  sharex=True, sharey=True, figsize=(6,3.5))
  plt.subplots_adjust(left=0.138, bottom=0.26, right=0.91, top=0.99, wspace=0.05, hspace=0.1)

  norm = Normalize(vmin=0, vmax=1)
  cmap = sns.color_palette("mako", as_cmap=True)
  cbar_ax = f.add_axes([.92, .38, .02, .40])

  rowIndex = 0
  for (result,text, size) in results:
    ax = axis[rowIndex]
    data = result[result['DataSize'] == size]

    colIndex = 0
    for scaling in np.unique(data['ErrorScaling']):
      df = data[data['ErrorScaling'] == scaling]
      pivot = df.pivot_table(index='NoiseLevelPercentage', columns='ErrorWidthPercentage',
                              values='AUC', aggfunc=np.sum)
      hm = sns.heatmap(data=pivot, ax=ax[colIndex], norm=norm, cmap=cmap, cbar_ax=cbar_ax, annot=True
                        , fmt='.2f', annot_kws={"size": 8}
                        , cbar_kws=dict(ticks=[0,1]))

      ax[colIndex].set_xlabel("")
      ax[colIndex].set_ylabel("")
      if((rowIndex == 1)):
        ax[colIndex].set_xlabel(f'$\psi$\n$\phi = {scaling}$')
        if(colIndex == 2):
          ax[colIndex].set_xlabel(f'$\psi$ - error function width\n$\phi = {scaling}$ - scaling')


      if(colIndex == 0):
        ax[colIndex].set_ylabel(f'{text}\n$\zeta$')
        if((rowIndex == 0)):
          ax[colIndex].set_ylabel(f'{text}\n$\zeta$ - noise level ')



      colIndex = colIndex + 1

    rowIndex= rowIndex+1

  cbar_ax.set_xlabel(f'AUC')
  cbar_ax.set_ylabel(f'AUC')
  plt.savefig(result_figure,dpi=600, transparent=True)
  plt.show()