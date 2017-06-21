import numpy as np
import matplotlib.pyplot as plt

temp = np.loadtxt('temp.txt')
X = np.arange(1, temp.shape[1] + 1, 1)
# Y = np.arange(1, 21, 1)

fig, ax = plt.subplots()
plt.plot(X, temp)
ax.grid(True)

plt.ylabel("temp, °C")
plt.xlabel("Lenght, mm")
plt.show()
