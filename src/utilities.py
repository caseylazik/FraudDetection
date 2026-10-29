import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    seed = 23
    test_size = 0.15
    train_size = 0.7
    val_size = 0.15

    df = pd.read_csv("../data/data.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Since we have a super imbalanced class where 0.172749% of cases are
    # fraudulent, and 99.82% are not, we should stratify with stratify = y
    # such that the model preserves roughly the same distribution between
    # these two classes. Stratify ensures that the training and test splits
    # have about the same proportions of classes (doesn't change test to
    # train size ratio, just means that one of the splits won't have much
    # more fraud cases since so few to begin with).
    X_temp, X_test, y_temp, y_test = train_test_split(X, y,
                                                      test_size=test_size,
                                                      random_state=seed,
                                                      stratify=y)

    # Since X_temp and y_temp are currently 0.85 of the total data, to get
    # validation to be also 0.15 like test, we need the relative size,
    # Proportion relative to original
    val_prop = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=val_prop,
                                                      random_state=seed,
                                                      stratify=y_temp)

    return X_temp, X_test, y_temp, y_test, X_train, X_val, y_train, y_val
