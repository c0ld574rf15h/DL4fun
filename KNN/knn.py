import numpy as np
from numpy.linalg import norm

from tqdm import tqdm

class KNN():
    def __init__(self, k, dist):
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
        dists = np.zeros((X_sz, self.X.shape[0]))

        if self.dist == 2:
            test_sqr = np.reshape(np.sum(X**2, axis=1), [num_test, 1])
            train_sqr = np.reshape(np.sum(self.X**2, axis=1), [1, num_train)
            test_train = np.matmul(X, self.X.T)
            
            dists = np.sqrt(test_sqr + train_sqr - 2 * test_train)
        else:
            with tqdm(total = X_sz) as pbar:
                for i in range(X_sz):
                    dists[i, :] = norm(self.X - X[i], self.dist, axis = 1)
                    pbar.update(1)
        
        return dists

    def predict(self, X):
        dists = self.compute_distances(X)
        dists_sort = np.argsort(dists, axis = 1)

        X_sz = X.shape[0]
        preds = np.zeros(X_sz)

        for i in range(X_sz):
            nearest_neighbors = self.y[dists_sort[i][:self.k]]
            uniq, cnt = np.unique(nearest_neighbors, return_counts = True)
            max_cnt = np.sort(cnt)[-1]
            preds[i] = uniq[np.where(cnt == max_cnt)[0][0]]
        
        return preds