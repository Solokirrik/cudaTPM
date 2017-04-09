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
clusters_num = 64

DATA_SIZE = data.shape[0]
FEATURES = data.shape[1]
MEANS = clusters_num
ITERATIONS = 10
BATCH_SIZE = 10000

data = np.ndarray([DATA_SIZE, FEATURES])
clusters = np.ndarray([BATCH_SIZE,])
batch = np.ndarray([BATCH_SIZE, FEATURES])
means = np.ndarray([MEANS, FEATURES])
centroids = np.ndarray([MEANS, FEATURES])
cluster_counts = np.ndarray([MEANS, ])


def initialize_means():
    for i in range(MEANS):
        index = np.random.randint() % DATA_SIZE
        for j in range(FEATURES):
            means[i][j] = data[index][j]


def select_batch():
    for i in range(BATCH_SIZE):
        index = np.random.randint() % DATA_SIZE
        for j in range(FEATURES):
            batch[i][j] = data[index][j]


def cache_clusters():
    for i in range(BATCH_SIZE):
        dist_min = float("inf")
        for j in range(MEANS):
            dist = 0
            for k in range(FEATURES):
                diff = batch[i][k] - means[j][k]
                dist += diff * diff
            if dist < dist_min:
                dist_min = dist
                clusters[i] = j


def update_means():
    for i in range(MEANS):
        for j in range(FEATURES):
            centroids[i][j] = means[i][j]
    for i in range(BATCH_SIZE):
        index = clusters[i]
        cluster_counts[index] += 1
        eta = 1 / cluster_counts[index]
        for j in range(FEATURES):
            means[index][j] = (1.0 - eta) * means[index][j] + eta * batch[i][j]
    pass


def calculate_error():
    for i in range(MEANS):
        dist = 0
        for j in range(FEATURES):
            diff = centroids[i][j] - means[i][j]
            dist = diff * diff
    pass

for i in range(ITERATIONS):
    select_batch()
    cache_clusters()
    update_means()
    calculate_error()
    pass

