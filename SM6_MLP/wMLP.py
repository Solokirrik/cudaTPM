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
        self.hidLaySize = 10

    def _sigmoid_activation(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigm_out_2_der(self, output):
        return (1 - output)*output

    def softmax(self, x):
        e = np.exp(float(x))
        return e / np.sum(e)

    def fit(self, X_in, y_in, alpha=0.005, eps=1e-8):
        features = X_in.shape[1]
        epoch = 0
        while True:
            if not self.fitted:
                np.random.seed(5)
                self.W2 = np.random.random((features, self.hidLaySize))
                self.W3 = np.random.random((self.hidLaySize, 1))
                # self.d2 = np.zeros(self.W2.shape)
                # self.d3 = np.zeros((X_in.shape[0], self.hidLaySize))
                # self.H3 = np.zeros((X_in.shape[0], self.hidLaySize))
                # self.G2 = np.zeros((self.hidLaySize, features))
                self.delta0_old = np.zeros((X_in.shape[0], self.hidLaySize))
                self.fitted = True
            else:
                H2 = self._sigmoid_activation(X_in.dot(self.W2))
                out = self._sigmoid_activation(H2.dot(self.W3))
                delta0 = y_in.reshape(1, y_in.shape[0]).T - out
                d3 = self.sigm_out_2_der(H2)*(delta0.dot(self.W3.T))
                G3 = H2.T.dot(delta0)
                dW3 = alpha*G3
                G2 = X_in.T.dot(d3)
                dW2 = alpha*G2
                self.W3 += dW3
                self.W2 += dW2
                epoch += 1
                loss = ((self.delta0_old - delta0)**2).sum()**0.5
                print('loss', '\t', epoch, '\t', loss)
                if loss < eps:
                    break
                self.delta0_old = delta0

    def predict(self, X_in):
        H2 = self._sigmoid_activation(X_in.dot(self.W2))
        return self._sigmoid_activation(H2.dot(self.W3))

    def fit_predict(self):
        pass

# mndata = MNIST('./data')
# training = mndata.load_training()
# testing = mndata.load_testing()

# dataset = datasets.fetch_mldata("MNIST Original")
# (trainX, testX, trainY, testY) = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"),
# test_size = 0.33)

digits = load_digits()
# ~178 "0" 8x8 -> 1x64
zeros = digits.data[np.where(digits.target==0)[0]]
# ~182 "1" 8x8 -> 1x64
ones = digits.data[np.where(digits.target==1)[0]]
# ~177 "2" 8x8 -> 1x64
twos = digits.data[np.where(digits.target==2)[0]]

data_set = np.concatenate([ones, zeros, twos])
target = np.concatenate([np.ones((zeros.shape[0],1)),
                         np.zeros((ones.shape[0],1)),
                         np.zeros((twos.shape[0],1))])[:,0].T

ids = np.arange(target.shape[0])
np.random.shuffle(ids)
data_set = data_set[ids]
target = target[ids]

train_data_set, test_data_set, train_target, test_target = train_test_split(data_set, target, test_size=0.33,
                                                                            random_state=42)

mlp = MLP()
mlp.fit(train_data_set, train_target)
pred = mlp.predict(test_data_set)
print(pred)

print(roc_auc_score(y_score=pred[:,0], y_true=test_target))

# plt.matshow(digits.images[0])
# plt.gray()
# plt.show()