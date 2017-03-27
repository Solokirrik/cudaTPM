from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Z = np.loadtxt('ZarrInp')
Z = np.loadtxt('ZarrOut')

X = np.arange(1, Z.shape[0] + 1, 1)
Y = np.arange(1, Z.shape[1] + 1, 1)
X, Y = np.meshgrid(X, Y)

surf = ax.plot_surface(X, Y, Z, cmap=cm.Blues, rstride=1, cstride=1, linewidth=1, edgecolors='black')

ax.set_zlim(Z.min(), Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.title("Входное воздействие")
plt.title("Результат")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()
