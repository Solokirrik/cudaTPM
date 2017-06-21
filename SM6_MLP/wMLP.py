# from mnist import MNIST
# from sklearn import datasets
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class MLP():
    def __init__(self):
        self.fitted = False
        self.out_size = 10
        self.hidLaySize = 64

    def _save_w(self):
        np.save('./W1.txt', self.W1)
        np.save('./W2.txt', self.W2)

    def _sigm_activation(self, x):
        return 1.0 / (1 + np.exp(-x))

    def _sigm_der(self, x):
        return (1 - self._sigm_activation(x)) * self._sigm_activation(x)

    def _tanh_activation(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def _tanh_der(self, x):
        return 1 - self._tanh_activation(x)**2

    def _softmax(self, x):
        e = np.exp(float(x))
        return e / np.sum(e)

    def _norm_X(self, x):
        # x_max = x.max()
        # return x / x_max
        return np.c_[np.ones((x.shape[0])), x]

    def _to_target(self, y):
        y_ret = np.zeros((1, self.out_size))
        y_ret[0][y] = 1
        return y_ret

    def fit(self, X_in, y_in, E=0.07, alpha=0.3, eps=1e-7):
        features = X_in.shape[1]
        samples = X_in.shape[0]
        epoch = 0
        dW10 = 0
        dW20 = 0

        for i in range(samples):
            while True:
                if not self.fitted:
                    np.random.seed(56)
                    self.W1 = np.random.random((features, self.hidLaySize))
                    self.W2 = np.random.random((self.hidLaySize, self.out_size))
                    self.fitted = True
                else:
                    # X = self._norm_X(X_in[i].reshape((1, features)))
                    X = X_in[i].reshape((1, features))
                    L1 = self._sigm_activation(X.dot(self.W1))
                        # .reshape((self.hidLaySize, 1))
                    L2 = self._sigm_activation(L1.dot(self.W2))
                        # .reshape((self.out_size, 1))
                    target = self._to_target(y_in[i])
                    L2_err = target - L2
                    loss = (L2_err**2).sum()
                    L2_delta = L2_err*self._sigm_der(L2)
                    L1_err = L2_delta.dot(self.W2.T)
                    L1_delta = L1_err*self._sigm_der(L1)
                    dW2 = L1.T.dot(L2_delta) + alpha * dW20
                    dW1 = X.T.dot(L1_delta) + alpha * dW10
                    self.W2 += 1 * dW2
                    self.W1 += 1 * dW1
                    epoch += 1
                    if loss < eps:
                        epoch = 0
                        break
                    print(i, '\t', 'loss', '\t', epoch, '\t', loss)
                    dW10 = dW1
                    dW20 = dW2
        self._save_w()

    def predict(self, X_in):
        L1 = self._sigm_activation(X_in.dot(self.W1))
        L2 = self._sigm_activation(L1.dot(self.W2))
        pred_result = L2.argmax(axis=1)
        return pred_result

    def fit_predict(self, X_in, y_in, X_pred):
        self.fit(X_in, y_in)
        self.predict(X_pred)

# mndata = MNIST('./data')
# training = mndata.load_training()
# testing = mndata.load_testing()

# dataset = datasets.fetch_mldata("MNIST Original")
# (trainX, testX, trainY, testY) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"),
# test_size = 0.33)

digits = load_digits()

zeros   = digits.data[np.where(digits.target==0)[0]]
ones    = digits.data[np.where(digits.target==1)[0]]
two     = digits.data[np.where(digits.target==2)[0]]
three   = digits.data[np.where(digits.target==3)[0]]
four    = digits.data[np.where(digits.target==4)[0]]
five    = digits.data[np.where(digits.target==5)[0]]
six     = digits.data[np.where(digits.target==6)[0]]
seven   = digits.data[np.where(digits.target==7)[0]]
eight   = digits.data[np.where(digits.target==8)[0]]
nine    = digits.data[np.where(digits.target==9)[0]]

data_set = np.concatenate([zeros, ones, two, three, four, five, six, seven, eight, nine])

a2 = np.zeros((two.shape[0], 1))
a3 = np.zeros((three.shape[0], 1))
a4 = np.zeros((four.shape[0], 1))
a5 = np.zeros((five.shape[0], 1))
a6 = np.zeros((six.shape[0],1))
a7 = np.zeros((seven.shape[0],1))
a8 = np.zeros((eight.shape[0],1))
a9 = np.zeros((nine.shape[0],1))

a2.fill(2)
a3.fill(3)
a4.fill(4)
a5.fill(5)
a6.fill(6)
a7.fill(7)
a8.fill(8)
a9.fill(9)

target = np.concatenate([np.zeros((zeros.shape[0],1)),
                         np.ones((ones.shape[0],1)),
                         a2, a3, a4, a5, a6, a7, a8, a9])[:,0].T

# target = digits.target
# data_set = digits.data

ids = np.arange(target.shape[0])
np.random.shuffle(ids)
data_set = data_set[ids]
target = target[ids]

train_data_set, test_data_set, train_target, test_target = train_test_split(data_set,
                                                                            target,
                                                                            test_size=0.33,
                                                                            random_state=42)
mlp = MLP()

mlp.W1 = np.load('./W1.npy')
mlp.W2 = np.load('./W2.npy')

# mlp.fit(train_data_set, train_target)
pred = mlp.predict(test_data_set)
print("Target:\t\t", test_target[:37].astype(int))
print("Predicted:\t", pred[:37])

# print(roc_auc_score(y_score=pred[:,1], y_true=test_target))

# plt.matshow(digits.images[0])
# plt.gray()
# plt.show()