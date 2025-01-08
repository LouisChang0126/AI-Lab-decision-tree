import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from decision_tree import DecisionTree


def preprocess(df: pd.DataFrame)->pd.DataFrame:
    # (TODO): You need to change the string to number
    raise NotImplementedError

def accuracy_score(y_trues, y_preds) -> float:
    # (TODO): Return the accuracy
    raise NotImplementedError

def plot_accuracy(max_depths, accuracy):
    # (TODO) Draw the plot of different max_depth(y axis) and accuracy(x axis)
    raise NotImplementedError

def main():
    """
    Notice:
        1)
        2)
        3)
    """
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    X_train = train_df.drop(['satisfaction'], axis=1)
    y_train = train_df['satisfaction'].to_numpy()

    X_test = test_df.drop(['satisfaction'], axis=1)
    y_test = test_df['satisfaction'].to_numpy()

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """

    Accuracy = []
    max_depths = [5,10,15,20,25]
    for depth in max_depths:
        clf_tree = DecisionTree(
            max_depth=depth,
        )
        clf_tree.fit(X_train, y_train)
        y_pred_classes = clf_tree.predict(X_test)
        accuracy_ = accuracy_score(y_test, y_pred_classes)
        Accuracy.append(accuracy_)
        logger.info(f'DecisionTree | max_depth={depth} | Accuracy: {accuracy_:.4f}')
    plot_accuracy(max_depths, Accuracy)

if __name__ == '__main__':
    main()
