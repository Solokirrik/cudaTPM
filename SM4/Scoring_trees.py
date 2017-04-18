import time
import numpy as np
import pandas as pd
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


def get_split_p(y_vec, inp_vec, beta, print_q=False, eps=pow(10, -10)):
    new_r = np.sort(inp_vec)
    ll = new_r.argmin()
    rr = new_r.shape[0] - 1
    l = ll
    r = rr
    x_len = new_r.shape[0]
    iter_io = 0
    min_g = 2
    g = np.zeros((2,))
    b = np.zeros((2,))
    out = np.zeros((2,))
    while 1:
        iter_io += 1
        m0 = (l + r) // 2
        m1 = m0 + (r - m0) // 2
        b[0] = new_r[m0]
        b[1] = new_r[m1]
        x0l_len = m0 - ll
        x0r_len = rr - m0
        x1l_len = m1 - ll
        x1r_len = rr - m1

        p00l = ((y_vec > beta) * (new_r <= b[0])).sum() / x0l_len
        p00r = ((y_vec > beta) * (new_r > b[0])).sum() / x0r_len
        p10l = ((y_vec <= beta) * (new_r <= b[0])).sum() / x0l_len
        p10r = ((y_vec <= beta) * (new_r > b[0])).sum() / x0r_len

        p01l = ((y_vec > beta) * (new_r <= b[1])).sum() / x1l_len
        p01r = ((y_vec > beta) * (new_r > b[1])).sum() / x1r_len
        p11l = ((y_vec <= beta) * (new_r <= b[1])).sum() / x1l_len
        p11r = ((y_vec <= beta) * (new_r > b[1])).sum() / x1r_len

        hl = p00l * p10l
        hr = p00r * p10r
        h1l = p01l * p11l
        h1r = p01r * p11r

        g[0] = (x0l_len / x_len) * hl + (x0r_len / x_len) * hr
        g[1] = (x1l_len / x_len) * h1l + (x1r_len / x_len) * h1r

        if g[0] < g[1]:
            r = m0
        else:
            l = m0
        delta = min_g - g.min()
        if eps < delta:
            min_g = g.min()
        else:
            if print_q:
                print(iter_io, ")", l, m0, m1, r)
                print("delta", "\t", delta)
                print("argG", "\t", g.argmin())
                print("g0", "\t\t", '%.8f' % g[0])
                print("g1", "\t\t", '%.8f' % g[1])
            # out[0] = b[g.argmin()]
            # out[1] = g[g.argmin()]
            out = np.array([b[g.argmin()], g[g.argmin()], g[g.argmax()]])
            return out
        if print_q:
            print(iter_io, ")", l, m0, m1, r)
            print("delta", "\t", delta)
            print("g0", "\t\t", '%.8f' % g[0])
            print("g1", "\t\t", '%.8f' % g[1])
            print("-------------------------")

bet = 0
for i in range(X.shape[1]):
    gb = get_split_p(y, X.T[i], bet, print_q=True)
    bet = gb[0]
    print('%i)' % (i+1), "beta =", bet)
    print("G1 =", '%.5f' % gb[1])
    print("G2 =", '%.5f' % gb[2])
    print(np.min(X.T[i]), np.mean(X.T[i]), np.max(X.T[i]))
    print("-------------------------")

