import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from decision_tree import DecisionTree

"""
Notice:
    1) You can't add any additional package
    2) You don't have to change anything in main()
    3) You can ignore the suggested data type if you want
"""

def preprocess(df: pd.DataFrame)->pd.DataFrame:
    # (TODO): You need to change the string inside df into number,
    #         for example: change satisfied in df['satisfaction'] into 1, dissatisfied into 0
    raise NotImplementedError

def accuracy_score(y_trues: np.ndarray, y_preds: np.ndarray) -> float:
    # (TODO): Return the calculated accuracy
    raise NotImplementedError

def plot_accuracy(max_depths: list, accuracy: list):
    # (TODO) Draw the plot of different max_depth(y axis) and accuracy(x axis)
    raise NotImplementedError

def main():
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    X_train = train_df.drop(['satisfaction'], axis=1)
    y_train = train_df['satisfaction'].to_numpy()

    X_test = test_df.drop(['satisfaction'], axis=1)
    y_test = test_df['satisfaction'].to_numpy()

    save_accuracy = []
    max_depths = [5,10,15,20,25]
    for depth in max_depths:
        tree = DecisionTree(
            max_depth=depth
        )
        tree.fit(X_train, y_train)
        y_pred_classes = tree.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_classes)
        save_accuracy.append(accuracy)
        logger.info(f'DecisionTree | max_depth={depth:2d} | Accuracy: {accuracy:.4f}')
    plot_accuracy(max_depths, save_accuracy)

if __name__ == '__main__':
    main()
