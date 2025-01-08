import numpy as np
import pandas as pd

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, grow_tree, predict
    3) You can ignore the suggested data type if you want
"""
class DecisionTree:
    def __init__(self, max_depth: int):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.tree = self.grow_tree(X, y, 0)

    def grow_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        raise NotImplementedError

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Traverse the decision tree to return the classes of the testing dataset
        raise NotImplementedError

def find_best_split(X: pd.DataFrame, y: np.ndarray):
    # (TODO) Find the best split for a dataset
    raise NotImplementedError
    return best_feature_index, best_threshold

def split_dataset(X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
    # (TODO) split one node into left and right node 
    raise NotImplementedError
    return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

def entropy(y: np.ndarray)->float:
    # (TODO) Return the entropy
    raise NotImplementedError
