from Classifier import Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def q8():
    df = pd.read_csv('dataset/D2z.txt', sep=' ', names=['x1', 'x2', 'y'])
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    num_features = X.shape[1]
    classifier = Classifier(np.eye(num_features))
    classifier.fit(X, y)
    y_pred = classifier.classify(X)

    X_train_0 = X[y_pred == 0]
    X_train_1 = X[y_pred == 1]

    X1_test, X2_test = np.meshgrid(np.arange(-2, 2.01, 0.1), np.arange(-2, 2.01, 0.1))
    X_test = np.array([X1_test.flatten(), X2_test.flatten()]).T
    y_test_pred = classifier.classify(X_test)
    X_test_0 = X_test[y_test_pred == 0]
    X_test_1 = X_test[y_test_pred == 1]

    fig, ax = plt.subplots()
    ax.scatter(X_test_0[:, 0], X_test_0[:, 1], c='b', marker='.', label='test y=0')
    ax.scatter(X_test_1[:, 0], X_test_1[:, 1], c='b', marker='P', label='test y=1')
    ax.scatter(X_train_0[:, 0], X_train_0[:, 1], c='r', marker='.', label='train y=0')
    ax.scatter(X_train_1[:, 0], X_train_1[:, 1], c='r', marker='P', label='train y=1')
    ax.legend()
    plt.show()
    print('FINISH')

def q9(fname = 'dataset/D2a.txt'):
    df = pd.read_csv(fname, sep=' ')
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    num_features = X.shape[1]

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)

    # no normalization
    sum = 0
    for train_index, test_index in kf.split(X):
        classifier = Classifier(np.eye(num_features))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.classify(X_test)
        error = 1.0 - accuracy_score(y_test, y_pred)
        sum += error
        print("unnormalized error rate: %f"%error)
    avg = sum/5
    print("average rate: %f"%avg)

    # normalization
    sum = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        var = np.var(X_train, axis=0)
        A = (np.diag(1/var))
        classifier = Classifier(A)
        classifier.fit(X_train, y_train)
        y_pred = classifier.classify(X_test)
        error = 1.0 - accuracy_score(y_test, y_pred)
        sum += error
        print("normalized error rate: %f"%error)
    avg = sum/5
    print("average error rate: %f"%avg)

def q10():
    q9('dataset/D2b.txt')


def q12():
    b = pd.read_csv('dataset/D2b.txt', sep=' ', names=['x1', 'x2', 'y'])
    b.to_csv('weka/D2b.csv', sep=',', index=False)
    a = pd.read_csv('dataset/D2a.txt', sep=' ', names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y'])
    a.to_csv('weka/D2a.csv', sep=',', index=False)

def main():
    q9()
    q10()


if  __name__ == '__main__':
    main()