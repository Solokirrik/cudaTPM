from mpl_toolkits.mplot3d import Axes3D
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
roArr = np.ndarray([data.shape[0], 1])

# centers = np.zeros([clusters, data.shape[1]])
# for ce_ind in range(clusters):
#     for c_ind in range(data.shape[1]):
#         centers[ce_ind][c_ind] = np.random.randint(d_min, d_max)

centers = np.array([[32., 107., 200.], [222., 162., 230.], [157., 126., 21.], [190., 215., 150.],
                    [158., 222., 229.], [216., 183., 57.], [160., 62., 194.], [201., 78., 92.],
                    [184., 84., 102.], [220., 135., 164.], [187., 219., 245.], [208., 227., 121.],
                    [118., 88., 166.], [25., 72., 228.], [127., 199., 96.], [193., 83., 195.],
                    [116., 141., 136.], [226., 200., 230.], [88., 52., 95.], [206., 182., 69.],
                    [96., 38., 94.], [223., 127., 212.], [178., 238., 93.], [202., 211., 161.],
                    [180., 66., 53.], [176., 77., 241.], [217., 47., 42.], [125., 137., 226.],
                    [60., 81., 144.], [185., 95., 194.], [13., 241., 35.], [29., 21., 200.],
                    [140., 71., 144.], [60., 61., 207.], [22., 73., 58.], [44., 229., 42.],
                    [77., 119., 56.], [163., 240., 65.], [16., 160., 44.], [124., 18., 237.],
                    [83., 243., 65.], [178., 241., 117.], [69., 232., 61.], [169., 119., 154.],
                    [199., 124., 177.], [70., 36., 86.], [157., 83., 116.], [124., 200., 57.],
                    [188., 26., 253.], [145., 217., 203.], [160., 97., 113.], [235., 199., 173.],
                    [199., 141., 230.], [232., 66., 217.], [166., 143., 139.], [221., 17., 91.],
                    [216., 23., 165.], [18., 158., 32.], [89., 155., 191.], [67., 16., 124.],
                    [58., 19., 13.], [30., 242., 161.], [160., 9., 127.], [138., 230., 130.]])


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects

# dots_according_clusters
arto = init_list_of_objects(clusters)
roArr = np.ndarray([data.shape[0], 1])
cent_arr = np.ndarray([clusters, 1])
for eachD in range(0, len(data) - 1):
    minRo = float("inf")
    min_ind = int(0)
    for eachC in range(0, len(centers) - 1):
        ro = np.linalg.norm(centers[eachC] - data[eachD])
        if minRo > ro:
            minRo, ro = ro, minRo
            min_ind = eachC
            # roArr[eachD] = eachC
    arto[min_ind].append(eachD)
# for elem in range(0, len(roArr) - 1):
#     arto[int(roArr[elem])].append(elem)

# new_center_of_mass
new_cen1 = np.zeros([64, 3])
for string_n in range(len(arto)):
    if len(arto[string_n]) > 0:
        new_cen = np.zeros([1, 3])
        for elem_s in range(len(arto[string_n])):
            new_cen = new_cen + data[arto[string_n][elem_s]]
        new_cen1[string_n] = (new_cen / (len(arto[string_n]))).astype(int)
        print(string_n, new_cen1[string_n], len(arto[string_n]))

print(centers[47])

# # cT                - просто для отображения random CoM в
# # dataT             - просто для отображения data
# # roArr[196215, 1]  - соответствие индекса точки кластеру
# # arto[64,[]]       - лист на 64 листа точек соответствующего кластера
# # new_cen[1, 3]     - временное хранилище 1 элемента списка из arto[i]
# # new_cen1[64, 3]   - проверочный массив новых центров масс

# # plot
# cT = centers.T
# dataT = data.T
# surf1 = ax.scatter(cT[0], cT[1], cT[2])
# ax.set_zlim(d_min, d_max)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# # arto № + new_cen1 + count
# 2 [ 123.  131.   46.] 26
# 3 [ 180.  179.  154.] 26
# 4 [ 165.  192.  219.] 582
# 7 [ 193.   82.   73.] 34
# 8 [ 170.   94.   81.] 52
# 9 [ 206.  166.  142.] 9
# 10 [ 188.  198.  210.] 192
# 12 [ 105.   95.  123.] 2
# 14 [ 147.  155.   80.] 1
# 16 [ 119.  125.  124.] 32063
# 17 [ 228.  226.  225.] 11963
# 18 [ 83.  83.  85.] 7861
# 20 [ 111.   54.   51.] 118
# 23 [ 186.  187.  185.] 7063
# 24 [ 151.   78.   59.] 2068
# 27 [ 132.  166.  199.] 6781
# 28 [  63.   90.  115.] 8446
# 34 [ 48.  61.  47.] 11099
# 36 [ 87.  92.  56.] 10603
# 43 [ 141.  122.  144.] 1
# 44 [ 189.  169.  181.] 1
# 45 [ 61.  64.  66.] 3061
# 46 [ 127.   87.   84.] 119
# 47 [ 135.  150.   64.] 6
# 49 [ 156.  180.  204.] 7457
# 50 [ 134.  115.   94.] 3965
# 51 [ 212.  206.  194.] 326
# 54 [ 154.  154.  147.] 13285
# 58 [ 118.  153.  188.] 29222
# 60 [ 29.  35.  14.] 39782

