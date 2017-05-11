# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

digits = load_digits()
plt.gray()
plt.matshow(digits.images[0])
plt.show()

zeros = digits.data[np.where(digits.target==0)[0]]
ones = digits.data[np.where(digits.target==1)[0]]
two = digits.data[np.where(digits.target==2)[0]]
three = digits.data[np.where(digits.target==3)[0]]
four = digits.data[np.where(digits.target==4)[0]]
five = digits.data[np.where(digits.target==5)[0]]
data_set = np.concatenate([ones,zeros, two, three, four, five])
target = np.concatenate([np.ones((ones.shape[0],1)),np.zeros((zeros.shape[0],1))
                            # ,np.zeros((two.shape[0],1)), np.zeros((three.shape[0], 1)),
                         # np.zeros((four.shape[0],1)), np.zeros((five.shape[0],1))
                        ])[:,0].T
ids = np.arange(target.shape[0])
np.random.shuffle(ids)
data_set = data_set[ids]
target = target[ids]

train_data_set, test_data_set, train_target, test_target = train_test_split(data_set, target, test_size=0.33,
                                                                            random_state=42)

clf = MLPClassifier(solver='sgd', activation='logistic', alpha=0.001, hidden_layer_sizes=(32, 10))

clf.fit(train_data_set, train_target)
pred = clf.predict_proba(test_data_set)
print(pred[:10])

print(roc_auc_score(y_score=pred[:,1], y_true=test_target))
