import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.datasets.samples_generator import make_blobs

params = {'figure.subplot.left': 0.07,
          'figure.figsize': (10, 10),
          'figure.subplot.right': 0.95,
          'figure.subplot.bottom': 0.07,
          'figure.subplot.top': 0.95}
plt.rcParams.update(params)


class LinearRegression():
    def __init__(self):
        self.epoch = 0
        self._fitted = False
        self.loss_history = []
        self._delta_W = float("inf")

    def _norm_and_add(self, X):
        Xn = (X - X.min()) / X.ptp(0)
        X1 = np.c_[np.ones((Xn.shape[0])), Xn]
        return X1

    def fit(self, X, y, alpha=0.01, d_alp=0.97, eps=1e-8):
        X1 = self._norm_and_add(X)
        if not self._fitted:
            self.W = np.random.uniform(size=(X1.shape[1],))
            self._fitted = True
        while 1:
            self._delta_W = np.linalg.norm(self.W)
            preds = X1.dot(self.W)
            error = preds - y
            gradient = X1.T.dot(error) / X1.shape[0]
            self.W += - alpha * gradient
            alpha *= d_alp
            self.epoch += 1
            self.loss_history.append(np.sum(error) / X1.shape[0])
            if abs(self._delta_W - np.linalg.norm(self.W)) < eps:
                break

    def predict(self, X):
        X1 = self._norm_and_add(X)
        return X1.dot(self.W)

    def fit_predict(self, X, y, X_pred):
        self.fit(X, y)
        self.predict(X_pred)


class LogisticRegression():
    def __init__(self):
        self.loss_history = []
        self._fitted = False
        self._delta_W = float("inf")
        self.epoch = 0

    def _norm_and_add(self, X):
        Xn = (X - X.min()) / X.ptp(0)
        X1 = np.c_[np.ones((Xn.shape[0])), Xn]
        return X1

    def _sigmoid_activation(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, y, alpha=0.1, d_alp=0.97, eps=1e-8):
        X1 = self._norm_and_add(X)
        if not self._fitted:
            self.W = np.random.uniform(size=(X1.shape[1],))
            self._fitted = True
        while 1:
            self._delta_W = np.linalg.norm(self.W)
            preds = self._sigmoid_activation(X1.dot(self.W))
            error = preds - y
            gradient = X1.T.dot(error) / X1.shape[0]
            self.W += - alpha * gradient
            alpha *= d_alp
            self.epoch += 1
            self.loss_history.append(np.sum(error) / X1.shape[0])
            if abs(self._delta_W - np.linalg.norm(self.W)) < eps:
                break

    def predict(self, X):
        X1 = self._norm_and_add(X)
        return self._sigmoid_activation(X1.dot(self.W))

    def fit_predict(self, X, y, X_pred):
        self.fit(X, y)
        self.predict(X_pred)


df = pd.read_csv('./data/cs-training.csv', sep=',')
df = df.dropna()
X = df.as_matrix(columns=df.columns[1:])
y = df.as_matrix(columns=df.columns[:1])
y = y.reshape(y.shape[0])

# (X, y) = make_blobs(n_samples=25000, n_features=10, centers=2, cluster_std=1.5, random_state=20)

gkf = KFold(n_splits=5, shuffle=False)


# Logistic usage
log = LogisticRegression()
for train, test in gkf.split(X, y):
    t1 = time.time()
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    log.fit(X_train, y_train, alpha=0.02, d_alp=0.99, eps=1e-5)
    print(roc_auc_score(y_score=log.predict(X_test), y_true=y_test))
    print("%.3fsec" % (time.time() - t1))
    print("----------------------------------")

fpr, tpr, _ = metrics.roc_curve(y_score=log.predict(X_test), y_true=y_test)
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.grid(True)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve')
plt.plot(fpr, tpr)
plt.show()

# Linear usage
# lin = LinearRegression()
# for train, test in gkf.split(X, y):
#     t1 = time.time()
#     X_train, y_train = X[train], y[train]
#     X_test, y_test = X[test], y[test]
#
#     lin.fit(X_train, y_train, d_alp=0.99, eps=1e-12)
#     print(roc_auc_score(y_score=lin.predict(X_test), y_true=y_test))
#     print("%.3fsec" % (time.time() - t1))
#     print("----------------------------------")
#
# fpr, tpr, _ = metrics.roc_curve(y_score=lin.predict(X_test), y_true=y_test)
# plt.xlim(0, 1.1)
# plt.ylim(0, 1.1)
# plt.grid(True)
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.title('ROC-curve')
# plt.plot(fpr, tpr)
# plt.show()


# Plot loss
# fig = plt.figure()
# plt.grid(True)
# fig.suptitle("Training Loss")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss")
# plt.xlim(0, lin.epoch)
# plt.ylim(min(lin.loss_history), max(lin.loss_history))
# plt.plot(np.arange(0, lin.epoch), lin.loss_history)
# plt.show()

# # add 1st column
# x = np.array([[ 0,  1,  2],
#             [ 3,  4,  5],
#             [ 6,  7,  8],
#             [ 9, 10, 11]])
# x = np.array([[ 0,  1,  2]])
# b = np.ones((x.shape[0],x.shape[1] + 1))
# b.T[1:] = x.T
# print(x)
# print(b)
