import numpy as np
from numpy.linalg import norm

from tqdm import tqdm

class KNN():
    def __init__(self, k, dist=2):
        self.k = k
        self.dist = dist
    
    def train(self, X, y):
        """
        Training the classifier
        No actual training process, just storing the dataset
        """
        self.X = X
        self.y = y
    
    def compute_distances(self, X):
        """
        Computing all distances between data pairs
        (self.dist indicates the order of the norm)
        """
        num_test, num_train = X.shape[0], self.X.shape[0]
        dists = np.zeros((num_test, num_train))

        if self.dist == 2:
            test_sqr = np.sum(X**2, axis=1).reshape((num_test, 1))
            train_sqr = np.sum(self.X**2, axis=1).reshape((1, num_train))
            test_train = np.matmul(X, self.X.T)

            # This can cause runtime error since the argument can be a negative number
            dists = np.sqrt(test_sqr+train_sqr-2*test_train)
            # Replace nan values with 0
            np.nan_to_num(dists, copy=False)
        else:
            for i in range(num_test):
                dists[i, :] = norm(self.X - X[i], self.dist, axis=1)
        
        return dists

    def predict(self, X):
        dists = self.compute_distances(X)
        dists_sort = np.argsort(dists, axis=1)

        X_sz = X.shape[0]
        preds = np.zeros(X_sz)

        for i in range(X_sz):
            nearest_neighbors = self.y[dists_sort[i][:self.k]]
            uniq, cnt = np.unique(nearest_neighbors, return_counts=True)
            max_cnt = np.sort(cnt)[-1]
            preds[i] = uniq[np.where(cnt==max_cnt)[0][0]]
        
        return preds
    
    def get_nearest_neighbors(self, x):
        dists = self.compute_distances(x.reshape(1, x.shape[0]))
        dists_sort = np.argsort(dists, axis=1)
        return dists_sort