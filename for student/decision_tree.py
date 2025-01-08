import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.data_size = X.shape[0]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


def split_dataset(X, y, feature_index, threshold):
    raise NotImplementedError

# Find the best split for the dataset
def find_best_split(X, y):
    raise NotImplementedError

def entropy(y):
    raise NotImplementedError
