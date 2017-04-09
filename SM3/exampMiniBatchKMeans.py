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


class MiniBatchKMeans:
    def __init__(self, clusters):
        self.clusters = clusters
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0] * image.shape[1], 3))
clusters_num = 64

data_size = data.shape[0]
features = data.shape[1]
means_config = clusters_num
iterations = 10
batch_size = 10000

data = np.ndarray([data_size, features])
clusters = np.ndarray([batch_size, ])
batch = np.ndarray([batch_size, features])
means = np.ndarray([means_config, features])
centroids = np.ndarray([means_config, features])
cluster_counts = np.ndarray([means_config, ])


def initialize_means():
    for i in range(means_config):
        index = np.random.randint() % data_size
        for j in range(features):
            means[i][j] = data[index][j]


def select_batch():
    for i in range(batch_size):
        index = np.random.randint() % data_size
        for j in range(features):
            batch[i][j] = data[index][j]


def cache_clusters():
    for i in range(batch_size):
        dist_min = float("inf")
        for j in range(means_config):
            dist = 0
            for k in range(features):
                diff = batch[i][k] - means[j][k]
                dist += diff * diff
            if dist < dist_min:
                dist_min = dist
                clusters[i] = j


def update_means():
    for i in range(means_config):
        for j in range(features):
            centroids[i][j] = means[i][j]
    for i in range(batch_size):
        index = clusters[i]
        cluster_counts[index] += 1
        eta = 1 / cluster_counts[index]
        for j in range(features):
            means[index][j] = (1.0 - eta) * means[index][j] + eta * batch[i][j]
    pass


def calculate_error():
    for i in range(means_config):
        dist = 0
        for j in range(features):
            diff = centroids[i][j] - means[i][j]
            dist = diff * diff
    pass

for i in range(iterations):
    select_batch()
    cache_clusters()
    update_means()
    calculate_error()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

data_size = data.shape[0]
features = data.shape[1]
means_config = clusters_num
iterations = 10
batch_size = data_size // 10

# data = np.zeros([data_size, features])
clusters = np.zeros([batch_size, ])
batch = np.zeros([batch_size, features])
means = np.zeros([means_config, features])
centroids = np.zeros([means_config, features])
cluster_counts = np.zeros([means_config, ])
dist = np.zeros([means_config, ])

d_min = data.min()
d_max = data.max()

# initialize_means
for i in range(means_config):
    index = np.random.randint(0, data_size)
    means[i] = data[index]

for i in range(iterations):

    # select_batch() - batch reset
    for i in range(batch_size):
        index = np.random.randint(0, data_size)
        batch[i] = data[index]

    # cache_clusters()
    for i in range(batch_size):
        diff = np.abs(means - np.array([batch[i]] * means_config)) ** 2
        dist = np.sum(diff, axis=-1) ** (1. / 2)
        min_ind = dist.argmin()
        clusters[i] = min_ind.astype(np.uint)

    # update_means()
    centroids = means
    for i in range(batch_size):
        index = int(clusters[i])
        # print(index)
        cluster_counts[index] += 1
        eta = 1 / cluster_counts[index]
        means[index] = (1.0 - eta) * means[index] + eta * batch[i]

    # calculate_error()
    for i in range(means_config):
        diff = np.abs(means - np.array([centroids[i]] * means_config)) ** 2
        dist = np.sum(diff, axis=-1) ** (1. / 2)