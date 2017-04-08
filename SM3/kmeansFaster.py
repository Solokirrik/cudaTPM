from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time

params = {'figure.subplot.left': 0.0,
          'figure.subplot.right': 1.0,
          'figure.subplot.bottom': 0.0,
          'figure.subplot.top': 1.0}
plt.rcParams.update(params)

fig = plt.figure()
ax = fig.gca(projection='3d')

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0]*image.shape[1],3))

clusters = 64
d_min = np.min(data)
d_max = np.max(data)

centers = np.random.randint(d_min, d_max, [clusters, data.shape[1]])
sort_cen = centers[np.argsort(centers.sum(axis=1))]
cen_mean = sort_cen.mean().astype(int)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append( list() )
    return list_of_objects

# dots_according_clusters
cent_data = init_list_of_objects(clusters)
cent_arr = np.ndarray([clusters, 1])
t0 = time.time()
for eachD in range(len(data)):
    min_ind = np.sum(np.abs(centers - np.array([data[eachD]]*64))**2, axis=-1)**(1./2).argmin()
    cent_data[min_ind].append(eachD)
print(time.time() - t0)

