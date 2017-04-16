import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./data/cs-training.csv', sep=',')
df.head()

df = df.dropna()
X = df.as_matrix(columns=df.columns[1:])
y = df.as_matrix(columns=df.columns[:1])
y = y.reshape(y.shape[0])
# gkf = KFold(n_splits=5, shuffle=True)


class DecisionTreeClassifier:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def fit_predict(self):
        pass


class RandomForestClassifier:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def fit_predict(self):
        pass


# clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
# clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

# clf = DecisionTreeClassifier()
# clf = RandomForestClassifier()
# for train, test in gkf.split(X, y):
#     X_train, y_train = X[train], y[train]
#     X_test, y_test = X[test], y[test]
#     clf.fit(X_train, y_train)
#     print(accuracy_score(y_pred=clf.predict(X_test), y_true=y_test))

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects

# list_size = pow(2, X.shape[1] + 1) - 1
# treeList = init_list_of_objects(pow(2, X.shape[1] + 1) - 1)
# sizeArr = np.zeros((X.shape[1] + 1, 2))
# print(len(treeList))
#
# treeList[0] = np.where(y > y.min() - 1)
# sizeArr[0] = treeList[0][0].shape[0]/treeList[0][0].shape[0]
# print(treeList[0][0].shape[0], sizeArr[0], treeList[0])
#
# mask = y[treeList[0][0]] > 0
# treeList[1] = np.where(mask)
# sizeArr[1][0] = treeList[1][0].shape[0]/treeList[0][0].shape[0]
# sizeArr[1][1] = treeList[1][0].shape[0]/treeList[0][0].shape[0]
# print(treeList[1][0].shape[0], sizeArr[1], treeList[1])
#
# treeList[2] = np.where(~mask)
# sizeArr[2][0] = treeList[2][0].shape[0]/treeList[0][0].shape[0]
# sizeArr[2][1] = treeList[2][0].shape[0]/treeList[0][0].shape[0]
# print(treeList[2][0].shape[0], sizeArr[2], treeList[2])


def normalize_feature(vector):
    return vector / vector.sum()


def get_divn():
    beta = 1
    return beta


def choose_best_feature():
    feature = 1
    return feature

# for i in range(3, len(treeList)):
#     if i % 2:
#         mask = treeList[i//2] > get_divn()
#         treeList[i] = np.where(mask)
#     else:
#         mask = treeList[i//2 - 1] > get_divn()
#         treeList[i] = np.where(~mask)


def get_h(X):
    # t1 = time.time()
    new_r = X.T[0]
    ar_h1_h2_g = np.zeros((X.T[0].shape[0], 3))
    for el in range(len(new_r)):
        beta = new_r[el]
        if beta != 0:
            p1 = (beta > new_r).sum() / new_r.shape[0]
            p2 = (beta <= new_r).sum() / new_r.shape[0]
            ar_h1_h2_g[el][0] = -p1*np.log(p1)
            ar_h1_h2_g[el][1] = -p2*np.log(p2)
            ar_h1_h2_g[el][2] = p1*ar_h1_h2_g[el][0] + p2*ar_h1_h2_g[el][1]
    # print("%.3fsec" % (time.time() - t1))
    print(ar_h1_h2_g)
    print(ar_h1_h2_g.T[2].min(), ar_h1_h2_g.T[2].argmin())
# get_h(X)

newR = X.T[0]
ar_H1H2G = np.zeros((newR.shape[0], ))
t1 = time.time()
for el in range(len(newR)):
    beta1 = ((np.array([newR.shape[0], ])*newR[el] - newR) > 0).sum()
    if (beta1 != 0) and (beta1 != newR.shape[0]):
        p1 = beta1 / newR.shape[0]
        # p2 = 1 - p1
        ar_H1H2G[el] = -pow(p1, 2)*np.log(p1) - pow(1 - p1, 2)*np.log(1 - p1)
    else:
        ar_H1H2G[el] = 5
print("%.3fsec" % (time.time() - t1))
print(ar_H1H2G)
print(ar_H1H2G.min(), ar_H1H2G.argmin())


l = 0
r = newR.shape[0] - 1
ar_len = newR.shape[0]
while r - l > 1:
    m = (l + r) // 2
    bm = newR[m]
    pm = ((y > 0)*(X.T[0] > bm)).sum() / ar_len
    if pm != 0:
        gm = -pow(pm, 2)*np.log(pm) - pow((1 - pm), 2)*np.log(1 - pm)
        br = newR[r]
        pr = ((y > 0)*(X.T[0] > br)).sum() / ar_len
        if pr != 0:
            gr = -pow(pr, 2)*np.log(pr) - pow((1 - pr), 2)*np.log(1 - pr)
            if gm < gr:
                r = m
            else:
                l = m
            print(gm + gr)
#             print("---------------------------")
#             print("m=", m, "newR[m]=", newR[m])
#             print("r=", r, "newR[r]=", newR[r])
#             print("pm=", pm, "gm=", gm)
#             print("pr=", pr, "gr=", gr)
        else:
            r = r - 1
#             print("r0!", pr, r)
    else:
        m = m - 1
#         print("m0!", pm, m)
