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
        self.W = np.random.uniform(size=(1, ))
        self.fitted = False

    def linear_activation(self, x):
        return (x > 0) * 1

    def fit(self, X, y, epochs=10000):
        if not self.fitted:
            # self.W = np.random.uniform(size=(X.shape[1],))
            self.W = np.ones((X.shape[1], ))
            self.fitted = True
        epoch = 0
        alpha = 0.1
        while epoch < epochs:
            epoch += 1
            # alpha /= epoch
            # preds = self.linear_activation(np.dot(X, self.W))
            preds = np.dot(X, self.W)
            error = preds - y
            gradient = X.T.dot(error) / X.shape[0]
            # oldW = np.copy(self.W)
            self.W += - alpha * gradient
            # eps = np.linalg.norm(self.W - oldW)
            # if eps < 0.001:
            #     break
            # print(eps)
        # print(self.W)

    def predict(self, X):
        return (self.linear_activation(np.dot(X, self.W)) > 0) * 1

    def fit_predict(self):
        pass


class LogisticRegression():
    def __init__(self):
        self.W = np.random.uniform(size=(1, ))
        self.fitted = False

    def sigmoid_activation(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self, X, y, epochs=100):
        Xn = (X - X.min()) / X.ptp(0)
        X1 = np.c_[np.ones((Xn.shape[0])), Xn]
        if not self.fitted:
            self.W = np.random.uniform(size=(X1.shape[1], ))
            self.fitted = True
        epoch = 0
        alpha = 0.1
        while epoch < epochs:
            epoch += 1
            preds = self.sigmoid_activation(X1.dot(self.W))
            error = preds - y
            loss = np.sum(error) / X1.shape[0]
            gradient = X1.T.dot(error) / X1.shape[0]
            self.W += - alpha * gradient
            # alpha = np.linalg.norm(gradient) / np.linalg.norm(preds)
            # print(loss)
        # print(self.W)

    def predict(self, X):
        Xn = (X - X.min()) / X.ptp(0)
        X1 = np.c_[np.ones((Xn.shape[0])), Xn]
        return (self.sigmoid_activation(X1.dot(self.W)) >= 0.5) * 1

    def fit_predict(self):
        pass


df = pd.read_csv('./data/cs-training.csv', sep=',')
df = df.dropna()
X = df.as_matrix(columns=df.columns[1:])
y = df.as_matrix(columns=df.columns[:1])
y = y.reshape(y.shape[0])

# (X, y) = make_blobs(n_samples=250, n_features=10, centers=2, cluster_std=1.05, random_state=20)

gkf = KFold(n_splits=5, shuffle=True)

print(X.shape)
print(y.shape)


lin = LinearRegression()
log = LogisticRegression()

# X_train, y_train = X, y
# log.fit(X_train, y_train)

for train, test in gkf.split(X, y):
    t1 = time.time()
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]

    log.fit(X_train, y_train)
    print(roc_auc_score(y_score=log.predict(X_test), y_true=y_test))
    print("%.3fsec" % (time.time() - t1))

y_score=log.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_score=y_score, y_true=y_test)

plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.grid(True)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC-curve')
plt.show()
