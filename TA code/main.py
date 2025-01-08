import pandas as pd
from loguru import logger
import random
import matplotlib.pyplot as plt
from decision_tree import DecisionTree


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    df['satisfaction'] = df['satisfaction'].apply(lambda x: 1 if x == 'satisfied' else 0)
    df['Customer Type'] = df['Customer Type'].apply(lambda x: 1 if x == 'Loyal Customer' else 0)
    df['Type of Travel'] = df['Type of Travel'].apply(lambda x: 1 if x == 'Personal Travel' else 0)
    df['Class'] = df['Class'].map({
        'Eco': 1,
        'Eco Plus': 2,
        'Business': 3
    })
    # df.drop(columns=['Inflight entertainment','Seat comfort'], inplace=True)
    return df

def accuracy_score(y_trues, y_preds) -> float:
    return (y_trues == y_preds).sum() / y_trues.shape[0]

def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # (TODO): Implement you preprocessing function.
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
        logger.info(f'DecisionTree | max_depth={depth:2d} | Accuracy: {accuracy_:.4f}')
    # (TODO) Draw the plot of different max_depth
    plt.figure(figsize=(8, 8))
    plt.plot(max_depths, Accuracy)
    plt.xlabel('depth')
    plt.ylabel('accuracy')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
