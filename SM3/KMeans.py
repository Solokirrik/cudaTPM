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
# ax = fig.gca(projection='3d')

image = mpimg.imread('./mailru.jpg')
data = image.reshape((image.shape[0]*image.shape[1], 3))


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
                new_cen = np.zeros([1, self.d_depth])
                for elem_s in range(len(self.cent_data[string_n])):
                    new_cen = new_cen + self.data[self.cent_data[string_n][elem_s]]
                dif_cen = (new_cen / (len(self.cent_data[string_n]))).astype(int)
                if (self.cntrs[string_n] - dif_cen != np.zeros([1, self.d_depth])).all():
                    new_cnt += 1
                    self.cntrs[string_n] = dif_cen
        return new_cnt

    def fit(self, data_g):
        self.data = data_g
        self.cntrs = self.__init_mass_centers()

        new_cnt = 1
        while new_cnt > 0:
            t0 = time.time()
            new_cnt = 0
            self.cent_data = self.__init_list_of_objects()
            for dot in range(len(self.data)):
                clust_dist = np.sum(np.abs(self.cntrs - np.array([self.data[dot]] * 64)) ** 2, axis=-1) ** (1. / 2)
                min_ind = clust_dist.argmin()
                self.cent_data[min_ind].append(dot)
            # ax.scatter(self.cntrs.T[0], self.cntrs.T[1], self.cntrs.T[2])
            print(time.time() - t0)

            new_cnt = self.__rclc_mass_centers(new_cnt)

        new_data = np.zeros(self.data.shape)
        for row in range(len(self.cent_data)):
            # print(row, self.cntrs[row], len(self.cent_data[row]))
            for item in range(len(self.cent_data[row])):
                new_data[self.cent_data[row][item]] = self.cntrs[row]

        # ax.scatter(self.cntrs.T[0], self.cntrs.T[1], self.cntrs.T[2], label="stop")
        # ax.set_zlim(self.data.min(), self.data.max())
        # plt.xlabel("X")
        # plt.ylabel("Y")
        # plt.legend()
        # plt.show()
        return new_data

    def predict(self, data_g):
        t1 = time.time()
        predict_cent_data = self.__init_list_of_objects()
        for dot in range(len(data_g)):
            clust_dist = np.sum(np.abs(self.cntrs - np.array([data_g[dot]] * 64)) ** 2, axis=-1) ** (1. / 2)
            min_ind = clust_dist.argmin()
            predict_cent_data[min_ind].append(dot)

        new_data = np.zeros(self.data.shape)
        for row in range(len(self.cent_data)):
            print(row, self.cntrs[row], len(self.cent_data[row]))
            for item in range(len(self.cent_data[row])):
                new_data[self.cent_data[row][item]] = self.cntrs[row]
        print(time.time() - t1)
        return new_data


var_s = 64
new_image = KMean(var_s).fit(data)
new_image = new_image.reshape((image.shape[0], image.shape[1], 3)).astype(np.uint8)
# new_image.reshape((image.shape[0], image.shape[1], 3))
plt.axis("off")
plt.imshow(new_image)
plt.show()
print("\n")

image_g = mpimg.imread('./tp6jDLpgnuc.jpg')
data_g = image_g.reshape((image_g.shape[0] * image_g.shape[1], 3))
new_image_g = KMean(var_s).predict(data_g)
new_image_g = new_image_g.reshape((image_g.shape[0], image_g.shape[1], 3)).astype(np.uint8)
