import numpy as np


class Classifier:

    def __init__(self, A):
        self.X = None
        self.y = None
        self.A = A

    def fit(self, X, y):
        self.X = X
        self.y = y

    def classify(self, X):
        return np.apply_along_axis(self.classify_row, 1, X)

    def classify_row(self, x):
        def dist(dx):
            return np.sqrt(np.matmul(np.matmul(dx.T, self.A), dx))
        num_features = self.X.shape[1]
        num_samples = self.X.shape[0]
        x = np.reshape(x, (-1, num_features))
        x = np.broadcast_to(x, (num_samples, num_features))
        # i = np.argmin(dist(x, self.X)) # i - index of neighbor of nearest dist
        i = np.argmin(np.apply_along_axis(dist, 1, self.X - x))
        return self.y[i]





