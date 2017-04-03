from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sympy.external.tests.test_scipy import scipy

params = {'figure.subplot.left': 0.0,
          'figure.subplot.right': 1.0,
          'figure.subplot.bottom': 0.0,
          'figure.subplot.top': 1.0}

plt.rcParams.update(params)
fig = plt.figure()
ax = fig.gca(projection='3d')

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0]*image.shape[1],3))

class KMean:  # или MiniBatchKMeans
    def __init__(self, clusters, init=0):
        self.clusters = clusters
        pass

    def fit(self, data):
        d_min = np.min(data)
        d_max = np.max(data)
        centers = np.ndarray([self.clusters, data.shape[1]])
        for ce_ind in range(self.clusters):
            for c_ind in range(data.shape[1]):
                centers[ce_ind][c_ind] = np.random.randint(d_min, d_max)

        # cT[3, 64]
        cT = centers.T
        # dataT[196215, 3]
        dataT = data.T

        # surf = ax.scatter(dataT[0], dataT[1], dataT[2])
        surf = ax.scatter(cT[0], cT[1], cT[2])
        for indo in range(0, len(data)):
            pass
        ax.set_zlim(d_min, d_max)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        pass

    def predict(self, data):
        pass

# KMean(64).fit(data)

clusters = 64
d_min = np.min(data)
d_max = np.max(data)
centers = np.ndarray([clusters, data.shape[1]])
for ce_ind in range(clusters):
    for c_ind in range(data.shape[1]):
        centers[ce_ind][c_ind] = np.random.randint(d_min, d_max)

# cT[3, 64]
# dataT[196215, 3]
cT = centers.T
dataT = data.T

# D = np.sqrt((centers[0][0] - data[0][0])**2
#             + (centers[0][1] - data[0][1])**2
#             + (centers[0][2] - data[0][2])**2)
# D2 = np.linalg.norm(centers[0] - data[0])
# print(centers[0], data[0], D, D2)

roArr = np.ndarray([data.shape[0], 1, 3])
# for eachD in range(0, len(data) - 1):
#     minRo = float("inf")
#     list
#     for eachC in range(0, len(centers) - 1):
#         ro = np.linalg.norm(centers[eachC] - data[eachD])
#         if minRo > ro:
#             minRo, ro = ro, minRo
#             roArr[eachD] = eachC
    # print(roArr[eachD])
# print(roArr)


# for inp in arto:
#     if len(inp) > 0:
#         for inh in inp:
#             cgr = np.sum(inh, axis=0)
#             print(cgr, inh)

print(range(0, len(data) - 1))

for strings in arto:
    for elems in range(0, len(strings) - 1):

        new_cen = np.ndarray([1, 3])
        for elems in range(0, len(arto[47]) - 1):
            np.sum(new_cen, data[arto[47][elems]])