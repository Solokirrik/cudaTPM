import sys
# import seaborn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

temp = np.loadtxt('temp.txt')
X = np.arange(1, temp.shape[1] + 1, 1)

fig, ax = plt.subplots()
fig.set_tight_layout(True)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

line, = ax.plot(X, temp[0], 'b-', linewidth=2)

plt.ylim(-1, temp.max())
plt.grid()
def update(i):
    label = 'sample {0}'.format(i)
    print(label)
    line.set_ydata(temp[i])
    ax.set_xlabel(label)
    return line, ax

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=np.arange(0, temp.shape[0], 5), interval=10)
    anim.save('line4.gif', dpi=80, writer='imagemagick')
    plt.show()
