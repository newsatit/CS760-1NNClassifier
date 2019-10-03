from Classifier import Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def q8():
    df = pd.read_csv('dataset/D2z.txt', sep=' ', names=['x1', 'x2', 'y'])
    X = df.iloc[:, :2].to_numpy()
    y = df.iloc[:, 2].to_numpy()
    classifier = Classifier(np.eye(2))
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

def main():
    q8()
if  __name__ == '__main__':
    main()