from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.cluster import KMeans

params = {'figure.subplot.left': 0.0,
          'figure.subplot.right': 1.0,
          'figure.subplot.bottom': 0.0,
          'figure.subplot.top': 1.0}

plt.rcParams.update(params)
fig = plt.figure()
# ax = fig.gca(projection='3d')


class KMean:  # или MiniBatchKMeans
    def __init__(self, clusters):
        self.clusters = clusters

    def __init_mass_centers(self):
        self.d_depth = self.data.shape[1]
        return np.random.randint(self.data.min(), self.data.max(), [self.clusters, self.d_depth])

    def __init_list_of_objects(self):
        list_of_objects = list()
        for i in range(0, self.clusters):
            list_of_objects.append(list())
        return list_of_objects

    def __rclc_mass_centers(self, new_cnt):
        for string_n in range(len(self.cent_data)):
            if len(self.cent_data[string_n]) > 0:
                new_cen = np.zeros([1, self.data.shape[1]])
                for elem_s in range(len(self.cent_data[string_n])):
                    new_cen = new_cen + self.data[self.cent_data[string_n][elem_s]]
                dif_cen = (new_cen / (len(self.cent_data[string_n]))).astype(int)
                if (self.cntrs[string_n] - dif_cen != np.zeros([1, self.data.shape[1]])).all():
                    new_cnt += 1
                    self.cntrs[string_n] = dif_cen
        return new_cnt

    def fit(self, data_g):
        t1 = time.time()
        self.data = data_g
        self.cntrs = self.__init_mass_centers()
        print("Dots moved:", "\t", "Step time sec:")
        new_cnt = 1
        while new_cnt > 0:
            t0 = time.time()
            new_cnt = 0
            self.cent_data = self.__init_list_of_objects()
            for dot in range(len(self.data)):
                clust_dist = np.sum(np.abs(self.cntrs - np.array([self.data[dot]] * 64)) ** 2, axis=-1) ** (1. / 2)
                min_ind = clust_dist.argmin()
                self.cent_data[min_ind].append(dot)
            new_cnt = self.__rclc_mass_centers(new_cnt)
            print("%11d \t %.3f" % (new_cnt, time.time() - t0))

        print("Done in: ")
        print("%.3fsec" % (time.time() - t1))
        return self

    def predict(self, data_g):
        t1 = time.time()
        predict_cent_data = self.__init_list_of_objects()
        for dot in range(len(data_g)):
            clust_dist = np.sum(np.abs(self.cntrs - np.array([data_g[dot]] * 64)) ** 2, axis=-1) ** (1. / 2)
            min_ind = clust_dist.argmin()
            predict_cent_data[min_ind].append(dot)

        new_data = np.zeros(data_g.shape)
        for row in range(len(predict_cent_data)):
            for item in range(len(predict_cent_data[row])):
                new_data[predict_cent_data[row][item]] = self.cntrs[row]
        print("Done in: ")
        print(time.time() - t1)
        return new_data


image = mpimg.imread('./mailru.jpg')
plt.axis("off")
plt.imshow(image)
plt.show()

clusters = 64
data = image.reshape((image.shape[0] * image.shape[1], 3))
print(image.shape)
new_cntrs = KMean(clusters).fit(data)

image_g = mpimg.imread('./mailru.jpg')
data_g = image_g.reshape((image_g.shape[0]*image_g.shape[1], 3))
new_image = new_cntrs.predict(data_g)
print(image_g.shape)
new_image = new_image.reshape((image_g.shape[0], image_g.shape[1], 3)).astype(np.uint8)

plt.axis("off")
plt.imshow(image_g)
plt.imshow(new_image)
plt.show()
