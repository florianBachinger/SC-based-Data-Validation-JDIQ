import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shared_packages.Feynman.Functions as ff
import shared_packages.Feynman.Constraints as fc

data = ff.Feynman2.generate_df()
xlist = np.linspace(1, 3.0, 100)
ylist = np.linspace(1, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)

sigma = X
theta = Y
Z= ff.Feynman2.calculate(sigma, theta)

F2Constraints = [ x for x in fc.constraints if x['EquationName'] == 'Feynman2'][0]
F2order1sigma = [ y for y in F2Constraints['Constraints'] if ((y['name'] == 'theta') and (y['order_derivative'] == 1)) ][0]
f2derived = eval(F2order1sigma['derivative_lambda'])
Z2 = f2derived([sigma,theta])

fig,ax=plt.subplots(1,1)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.79, top=0.924, wspace=0.05, hspace=0.1)
cp = ax.contourf(X,Y,Z)
ax.set_xlabel("sigma")
ax.set_ylabel("theta")
cb = fig.colorbar(cp) # Add a colorbar to a plot
cb.set_label("f")
plt.savefig('figures/experimental_setup/Feynman2_contour.png', dpi = 600)

plt.clf()
fig,ax=plt.subplots(1,1)
plt.subplots_adjust(left=0.13, bottom=0.2, right=0.79, top=0.924, wspace=0.05, hspace=0.1)

cp = ax.contourf(X,Y,Z2)
ax.set_xlabel("sigma")
ax.set_ylabel("theta")

cb = fig.colorbar(cp) # Add a colorbar to a plot
cb.set_label( r"""$\frac{\partial f}{\partial theta}$""")
plt.savefig('figures/experimental_setup/Feynman2_contour_derviative.png', dpi = 600)