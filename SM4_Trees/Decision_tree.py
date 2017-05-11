# coding=utf-8
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

class DecisionTreeClassifier:
    def __init__(self, max_depth = 5, max_leafsize = 2):
        self._depth_ = max_depth
        self._leafsize_ = max_leafsize
        self._list_size_ = int(pow(2, self._depth_ + 1) - 1) * 2

    def _init_list_of_arrs_(self, size):
        list_of_arrs = list()
        for i in range(0, size):
            list_of_arrs.append(np.zeros((1, 1)))
        return list_of_arrs

    def _entropy_criteria_(self, p0l, p0r, p1l, p1r):
        if p0l == 0:
            hl = -p1l * np.log2(p1l)
        elif p1l == 0:
            hl = -p0l * np.log2(p0l)
        else:
            hl = -p0l * np.log2(p0l) - p1l * np.log2(p1l)
        if p0r == 0:
            hr = -p1r * np.log2(p1r)
        elif p1r == 0:
            hr = -p0r * np.log2(p0r)
        else:
            hr = -p0r * np.log2(p0r) - p1r * np.log2(p1r)
        return hl, hr

    def _gini_criteria_(self, p0l, p0r, p1l, p1r):
        hl = p0l * p1l
        hr = p0r * p1r
        return hl, hr

    def _get_split_(self, y_vec, y_beta, inp_vec):
        w_vec = inp_vec
        if w_vec.shape[0] > 10000:
            perc = 0.02
        elif w_vec.shape[0] > 5000:
            perc = 0.06
        elif w_vec.shape[0] > 1000:
            perc = 0.08
        elif w_vec.shape[0] > 500:
            perc = 0.5
        else:
            perc = 1
        btch_sz = int(w_vec.shape[0] * perc)
        arr_of_numbs = np.linspace(0, w_vec.shape[0] - 1, btch_sz, dtype=int)
        item_beta_g_arr = np.zeros((btch_sz, 3))
        iter_c = 0
        for item in arr_of_numbs:
            beta = w_vec[item]
            xl_len = (w_vec[arr_of_numbs] <= beta).sum()
            xr_len = btch_sz - xl_len
            if xl_len != 0 and xr_len != 0:

                # y_0 = (y_vec[arr_of_numbs] <= y_beta).sum()
                # y_1 = (y_vec[arr_of_numbs] > y_beta).sum()

                p0l = ((y_vec[arr_of_numbs] <= y_beta) * (w_vec[arr_of_numbs] <= beta)).sum() / xl_len
                p0r = ((y_vec[arr_of_numbs] <= y_beta) * (w_vec[arr_of_numbs] > beta)).sum() / xr_len
                p1l = ((y_vec[arr_of_numbs] > y_beta) * (w_vec[arr_of_numbs] <= beta)).sum() / xl_len
                p1r = ((y_vec[arr_of_numbs] > y_beta) * (w_vec[arr_of_numbs] > beta)).sum() / xr_len

                item_beta_g_arr[iter_c][0] = item
                item_beta_g_arr[iter_c][1] = beta

                # hl, hr = self._gini_criteria_(p0l, p0r, p1l, p1r)
                hl, hr = self._entropy_criteria_(p0l, p0r, p1l, p1r)
                G_l = (xl_len / btch_sz) * hl
                G_r = (xr_len / btch_sz) * hr
                G = G_l + G_r
                item_beta_g_arr[iter_c][2] = G
            else:
                item_beta_g_arr[iter_c][2] = 10
            iter_c += 1
        bestG = item_beta_g_arr.T[2].argmin()
        out = np.array([item_beta_g_arr[bestG][1], item_beta_g_arr[bestG][2]])
        return out

    def fit(self, X_train, y_train):
        arrList = self._init_list_of_arrs_(self._list_size_)
        best_features_beta = np.zeros((self._list_size_, 2))

        features = X_train.shape[1]
        features_list = [i for i in range(features)]
        b_g_arr = np.zeros((features, 2))

        y_median = np.median(y_train)
        arrList[0] = X_train
        print("max_depth = ", self._depth_, "\t", "list_size = ", self._list_size_)
        print(0, arrList[0].shape)
        print("-----------")
        iter = 0
        max_dict = []
        for i in range(int(pow(2, self._depth_ + 1) - 2)):
            if arrList[i].shape[0] > 0:
                if arrList[i].shape[0] == 1:
                    pass
                    # print(i * 2 + 1, "\t", "--", "\t", 0, "\t", 0)
                    # print(i * 2 + 2, "\t", "--", "\t", 0, "\t", 0)
                else:
                    # choose best feature
                    for feature in features_list:
                        b_g_arr[feature] = self._get_split_(y_train, y_median, arrList[i].T[feature])
                    bst_feat = b_g_arr.T[1].argmin()

                    beta = b_g_arr[bst_feat][0]
                    best_features_beta[i][0] = bst_feat
                    best_features_beta[i][1] = beta
                    mask = arrList[i].T[bst_feat] > beta

                    arrList[i * 2 + 1] = np.copy(arrList[i][mask])
                    arrList[i * 2 + 2] = np.copy(arrList[i][~mask])

                    print(i * 2 + 1, "\t", bst_feat, "\t", arrList[i * 2 + 1].shape[0])
                    print(i * 2 + 2, "\t", bst_feat, "\t", arrList[i * 2 + 2].shape[0])
            else:
                print(i * 2 + 1, "\t", "\t", arrList[i * 2 + 1].shape[0])
                print(i * 2 + 2, "\t", "\t", arrList[i * 2 + 2].shape[0])

            max_dict.append(np.maximum(arrList[i * 2 + 1].shape[0], arrList[i * 2 + 2].shape[0]))
            # lvl separators
            if (i // (pow(2, iter + 2) - 2)) or i == 0:
                iter += 1
                print("-----------")
                if max(max_dict) == self._leafsize_:
                    break
                else:
                    max_dict.clear()

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

df = pd.read_csv('./data/cs-training.csv', sep=',')
df = df.dropna()
X = df.as_matrix(columns=df.columns[1:])
y = df.as_matrix(columns=df.columns[:1])
y = y.reshape(y.shape[0])

# iris = load_iris()
# X = iris.data
# y = iris.target

tree = DecisionTreeClassifier(max_depth=20, max_leafsize = 1)
tree.fit(X, y)