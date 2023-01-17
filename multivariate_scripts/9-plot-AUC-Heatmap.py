import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

result_file = f'data/multivariate/8-AUC_comparison.csv'
result_figure = 'figures/multivariate/9-AUC-Heatmap.png'

df = pd.read_csv(result_file)
df = df[df['ErrorScaling'] == 1]
df = df[df['DataSize'] == 200]
  

f, ax = plt.subplots(1,1,  sharex=True, sharey=True, figsize=(3.5,2.5))
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.79, top=0.924, wspace=0.05, hspace=0.1)

norm = Normalize(vmin=0, vmax=1)
cmap = sns.color_palette("magma", as_cmap=True)
cbar_ax = f.add_axes([.80, .4, .03, .40])


pivot = df.pivot_table(index='NoiseLevelPercentage', columns='ErrorWidthPercentage',
                        values='AUC', aggfunc=np.sum)
hm = sns.heatmap(data=pivot, ax=ax, norm=norm, cmap=cmap, cbar_ax=cbar_ax, annot=True)

ax.set_xlabel(f'error function width in %')
ax.set_ylabel(f'noise level in percentage of $\sigma$')
cbar_ax.set_xlabel(f'AUC')
cbar_ax.set_ylabel(f'AUC')

plt.savefig(result_figure,dpi=600, transparent=True)
plt.show()