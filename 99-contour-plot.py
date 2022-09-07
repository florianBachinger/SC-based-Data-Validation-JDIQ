import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(1, 3.0, 1000)
ylist = np.linspace(1, 3.0, 1000)
theta, sigma = np.meshgrid(xlist, ylist)
Z = np.exp(-(theta/sigma)**2/2)/(np.sqrt(2*np.pi)*sigma)

clev = np.arange(0,Z.max(),.001) #Adjust the .001 to get finer gradient
CS = plt.contourf(theta, sigma, Z, clev, cmap=plt.cm.coolwarm,extend='both')
plt.show()
plt.clf()

fig,ax=plt.subplots(1,1)
cp = ax.contourf(theta, sigma, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_xlabel('theta')
ax.set_ylabel('sigma')
plt.show()