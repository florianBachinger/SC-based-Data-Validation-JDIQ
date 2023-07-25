import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc

data = ff.Feynman2.generate_df()

g = sns.PairGrid(data)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.savefig('figures/experimental_setup/Feynman2_pairgrid_derviative.pdf', dpi = 600)
plt.show()
