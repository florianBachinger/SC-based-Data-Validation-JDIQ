import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import shared_packages.SyntheticError.Functions as sef
font = {'size'   : 14}
plt.rc('font', **font)

num = 100
x = np.linspace(0,num,num)
y = [0]*num

pal1 = sns.color_palette("rocket_r", n_colors=10).as_hex()
pal2 = sns.color_palette("mako_r", n_colors=10).as_hex()
pal3 = sns.color_palette("YlOrBr", n_colors=10).as_hex()

spike = sef.Spike(num, start=num*0.2, end=num*0.8)
square = sef.Square(num, start=num*0.4, end=num*0.6)
normal = sef.Normal(num, start=num*0.2, end=num*0.8) * 1.5

fig, ax = plt.subplots(1,3, figsize=(8,1.8),  sharey=True, sharex=True, gridspec_kw={
    'height_ratios': [1], 'width_ratios': [1,1,1]})
plt.subplots_adjust(left=0.03, bottom=0.175, right=0.82, top=0.99, wspace=0.05, hspace=0.1)

# ----------  Spike - none ----------
ax[0].plot(x,spike, c=pal1[7])

ax[0].plot(np.linspace(num*0.2, num*0.8, num),[1.2]*num,ls='--', lw=1.5, c=pal1[7])
ax[0].text(50,1.3, "$\psi=0.6$", ha='center')

# ----------  Square - width ----------
ax[1].plot(x,square, c=pal2[7])

ax[1].plot(np.linspace(num*0.4, num*0.6, num),[1.2]*num,ls='--', lw=1.5, c=pal2[7], label='width $\psi$')
ax[1].text(50,1.3, "$\psi=0.2$", ha='center')

ax[1].vlines(x=75, ymin=0, ymax=1,  color=pal2[7], ls=':', lw=1.5, label='scaling $\phi$')
ax[1].text(76,0.5, "$\phi=1$")

ax[1].legend(bbox_to_anchor=(2.02, 1), loc='upper left', framealpha=0, handlelength=1.5)

# ----------  Normal - height ----------
ax[2].plot(x,normal, c=pal3[7])
ax[2].vlines(x=63, ymin=0, ymax=1.5,  color=pal3[7], ls=':', lw=1.5)
ax[2].text(65,0.5, "$\phi=1.5$")

ax[0].set_yticks([0,1])
ax[0].set_yticklabels(['0','1'])

ax[0].set_xticks([0,1])
ax[0].set_xticklabels(['0','1'])
ax[1].set_xticks([0,100])
ax[1].set_xticklabels(['0','1'])
ax[2].set_xticks([0,100])
ax[2].set_xticklabels(['0','1'])

ax[0].tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
ax[1].tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=False, labeltop=False, labelright=False, labelbottom=True)
ax[2].tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=False, labeltop=False, labelright=False, labelbottom=True)

plt.savefig('figures/experimental_setup/univariate_error_functions.png', dpi = 600)
plt.show()
