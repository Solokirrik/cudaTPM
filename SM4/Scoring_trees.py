import time
import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./data/cs-training.csv', sep=',')
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
            out = np.array([b[g.argmin()], g[g.argmin()]])
            return out
        if print_q:
            print(iter_io, ")", l, m0, m1, r)
            print("delta", "\t", delta)
            print("g0", "\t\t", '%.8f' % g[0])
            print("g1", "\t\t", '%.8f' % g[1])
            print("-------------------------")


def get_beta(y_vec, y_beta, inp_vec, print_q=False, eps=pow(10, -10)):
    new_r = np.sort(inp_vec)
    ll = 0
    rr = new_r.shape[0] - 1
    l = ll
    r = rr
    x_len = new_r.shape[0]
    min_g = 2
    g = np.zeros((2, ))
    b = np.zeros((2, ))
    m = np.zeros((2, ))
    out = np.zeros((2,))
    while r - l > 1:
        m[0] = (l + r) // 2
        m[1] = m[0] + (r - m[0]) // 2

        # print(l, r, m)
        # print(new_r[int(m[0])])

        b[0] = new_r[int(m[0])]
        b[1] = new_r[int(m[1])]

        x0l_len = m[0] - ll
        x0r_len = rr - m[0]
        x1l_len = m[1] - ll
        x1r_len = rr - m[1]

        p00l = ((y_vec > y_beta) * (new_r <= b[0])).sum() / x0l_len
        p00r = ((y_vec > y_beta) * (new_r > b[0])).sum() / x0r_len
        p10l = ((y_vec <= y_beta) * (new_r <= b[0])).sum() / x0l_len
        p10r = ((y_vec <= y_beta) * (new_r > b[0])).sum() / x0r_len

        p01l = ((y_vec > y_beta) * (new_r <= b[1])).sum() / x1l_len
        p01r = ((y_vec > y_beta) * (new_r > b[1])).sum() / x1r_len
        p11l = ((y_vec <= y_beta) * (new_r <= b[1])).sum() / x1l_len
        p11r = ((y_vec <= y_beta) * (new_r > b[1])).sum() / x1r_len

        hl = p00l * p10l
        hr = p00r * p10r
        h1l = p01l * p11l
        h1r = p01r * p11r

        g[0] = (x0l_len / x_len) * hl + (x0r_len / x_len) * hr
        g[1] = (x1l_len / x_len) * h1l + (x1r_len / x_len) * h1r

        if g[0] < g[1]:
            r = m[0]
        else:
            l = m[0]
        # print(l, r, m)
        delta = min_g - g.min()
        if eps < delta:
            min_g = g.min()
        else:
            out = np.array([b[g.argmin()], g[g.argmin()]])
            return out


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def index_to_vec(inp_list, x, col):
    out_vec = np.zeros((1, len(inp_list)))
    for index in range(len(inp_list)):
        out_vec[0][index] = x.T[col][inp_list[index]]
    return out_vec


def init_list_of_arrs(size):
    list_of_arrs = list()
    for i in range(0, size):
        list_of_arrs.append(np.zeros((1, 1)))
    return list_of_arrs


# for i in range(3, len(treeList)):
#     if i % 2:
#         mask = treeList[i//2] > get_divn()
#         treeList[i] = treeList[np.where(mask)]
#     else:
#         mask = treeList[i//2 - 1] > get_divn()
#         treeList[i] = np.where(~mask)

list_size = pow(2, X.shape[1] + 1) - 1
arrList = init_list_of_arrs(list_size)
best_features_beta = np.zeros((list_size, 2))

t1 = time.time()

arrList[0] = X
beta = 0
features = X.shape[1]
mask = y > beta
arrList[1] = X[mask]
arrList[2] = X[~mask]

print(0, arrList[0].shape)
print("-----------")
print(1, arrList[1].shape)
print(2, arrList[2].shape)
print("-----------")

features_list = []
for i in range(features):
    features_list.append(i)
iter = 0
b_g_arr = np.zeros((features, 2))
for item in range(3, int(list_size)):
    for feature in features_list:
        if item % 2:
            b_g_arr[feature] = get_beta(y, 0, arrList[item // 2].T[feature])
        else:
            b_g_arr[feature] = get_beta(y, 0, arrList[item // 2 - 1].T[feature])
    bst_feat = b_g_arr.T[1:2].argmin()
    # print(item, iter, bst_feat)
    beta = b_g_arr[bst_feat][0]
    if item % 2:
        mask = arrList[item // 2].T[bst_feat] > beta
        arrList[item] = arrList[item // 2][mask]
    else:
        mask = arrList[item // 2 - 1].T[bst_feat] > beta
        arrList[item] = arrList[item // 2 - 1][~mask]
    best_features_beta[0] = bst_feat
    best_features_beta[1] = beta
    if (item // (pow(2, iter + 3) - 2)):
        iter += 1
        # b_g_arr[bst_feat] = float("nan")
        b_g_arr[bst_feat] = float("inf")
        features_list.remove(bst_feat)
        print(b_g_arr)
        print("-----------")
    print(item, iter, bst_feat, arrList[item].shape)
print("%.3fsec" % (time.time() - t1))
