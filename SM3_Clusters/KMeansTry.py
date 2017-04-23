from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

params = {'figure.subplot.left': 0.0,
          'figure.subplot.right': 1.0,
          'figure.subplot.bottom': 0.0,
          'figure.subplot.top': 1.0}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.gca(projection='3d')

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0] * image.shape[1], 3))

clusters = 64
d_min = np.min(data)
d_max = np.max(data)
t0 = time.time()


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects

d_depth = data.shape[1]
cntrs = np.random.randint(data.min(), data.max(), [clusters, d_depth])

new_cnt = 1
while new_cnt > 0:
    t1 = time.time()
    new_cnt = 0
    cent_data = init_list_of_objects(clusters)
    for dot in range(len(data)):
        clust_dist = np.sum(np.abs(cntrs - np.array([data[dot]] * 64)) ** 2, axis=-1) ** (1. / 2)
        min_ind = clust_dist.argmin()
        cent_data[min_ind].append(dot)
    # ax.scatter(cntrs.T[0], cntrs.T[1], cntrs.T[2])
    print(time.time() - t1)

    for string_n in range(len(cent_data)):
        if len(cent_data[string_n]) > 0:
            new_cen = np.zeros([1, d_depth])
            for elem_s in range(len(cent_data[string_n])):
                new_cen = new_cen + data[cent_data[string_n][elem_s]]
            dif_cen = (new_cen / (len(cent_data[string_n]))).astype(int)
            if (cntrs[string_n] - dif_cen != np.zeros([1, d_depth])).all():
                new_cnt += 1
                cntrs[string_n] = dif_cen

new_data = np.zeros(data.shape)
for row in range(len(cent_data)):
    print(row, cntrs[row], len(cent_data[row]))
    for item in range(len(cent_data[row])):
        new_data[item] = cntrs[row]

print(time.time() - t0)
print(cntrs)

# ax.scatter(centers.T[0], centers.T[1], centers.T[2])
# ax.set_zlim(d_min, d_max)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.legend()
# plt.show()
