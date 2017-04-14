import pandas as pd
import numpy as np
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

treeList = init_list_of_objects(pow(2, X.shape[1] + 1) - 1)
sizeArr = np.zeros((X.shape[1] + 1, 2))
print(len(treeList))

treeList[0] = np.where(y > y.min() - 1)
sizeArr[0] = treeList[0][0].shape[0]/treeList[0][0].shape[0]
print(treeList[0][0].shape[0], sizeArr[0], treeList[0])

mask = y[treeList[0][0]] > 0
treeList[1] = np.where(mask)
sizeArr[1][0] = treeList[1][0].shape[0]/treeList[0][0].shape[0]
sizeArr[1][1] = treeList[1][0].shape[0]/treeList[0][0].shape[0]
print(treeList[1][0].shape[0], sizeArr[1], treeList[1])

treeList[2] = np.where(~mask)
sizeArr[2][0] = treeList[2][0].shape[0]/treeList[0][0].shape[0]
sizeArr[2][1] = treeList[2][0].shape[0]/treeList[0][0].shape[0]
print(treeList[2][0].shape[0], sizeArr[2], treeList[2])

