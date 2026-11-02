import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

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


def seed():
    return 23





def find_optimal_classification_threshold(pipeline, X_val, y_val, X_train, y_train):


    pipeline.fit(X_train, y_train)

    # 2. Get fraud probabilities from validation data
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]

    # 3. Define thresholds to test
    thresholds = np.arange(0.01, 1.00, 0.01)

    # 4. Evaluate each threshold
    results = []

    for threshold in thresholds:

        y_val_pred = (y_val_proba >= threshold).astype(int)

        precision = precision_score(
            y_val, y_val_pred, zero_division=False   # Since we are using such low classification thresholdd, we can get no intances, so add zero_div = F
        )

        recall = recall_score(
            y_val, y_val_pred
        )

        f1 = f1_score(
            y_val, y_val_pred
        )

        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # 5. Convert results into DataFrame
    results = pd.DataFrame(results)

    # 6. Find threshold with highest F1
    best_idx = results['f1'].idxmax()
    best_result = results.loc[best_idx]

    best_threshold = best_result['threshold']

    print("Optimal Threshold:", best_threshold)
    print("Precision:", best_result['precision'])
    print("Recall:", best_result['recall'])
    print("F1:", best_result['f1'])

    # 7. Visualize results
    plt.figure(figsize=(10, 6))

    plt.plot(
        results['threshold'],
        results['precision'],
        label='Precision'
    )

    plt.plot(
        results['threshold'],
        results['recall'],
        label='Recall'
    )

    plt.plot(
        results['threshold'],
        results['f1'],
        label='F1'
    )

    # Mark optimal threshold
    plt.axvline(
        best_threshold,
        color='black',
        linestyle='--',
        label=f'Optimal Threshold: {best_threshold:.2f}'
    )

    plt.scatter(
        best_threshold,
        best_result['f1'],
        zorder=5
    )

    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 vs. Classification Threshold')

    plt.legend()
    plt.grid()
    plt.show()

    return best_threshold
