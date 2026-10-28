from utilities import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def main():

    seed = 23

    def eda():
        df = pd.read_csv("../data/data.csv")

        print("\nData information:")
        df.info()  # Time is how long since first transaction

        print("\nMissing values per column:")
        print(df.isnull().sum())

        # No missing data

        # Number of fraud cases to non-fraud cases (492 fraud, 284315 not)
        print("\nClass counts:")
        print(df["Class"].value_counts())

        # only 0.172749% of cases are fraudelent, 99.82%  are not
        print(df["Class"].value_counts(normalize=True) * 100)

        # Average transaction amount of dataset is $88.35
        print(df["Amount"].mean())
        # print(np.std(df["Amount"].to_numpy()))

        # Standard deviation 250.12
        print(df["Amount"].std(), "\n")

        # 1825 instances of 0 dollar transaction
        # Happens when verifying identity, or fraudsters checking etc.
        print((df["Amount"] == 0).sum())

        # Only Instances of fraud
        fraud_only_df = df[df["Class"] == 1]

        print(fraud_only_df["Amount"].mean())

        print(fraud_only_df["Amount"].std())

        # Must use log1p because instances of 0 Amount
        df["LogAmount"] = np.log1p(df["Amount"])
        # Distribution plot of fraud vs no fraud transaction amounts
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x="Class", y="LogAmount")
        plt.title("Log Transaction Amount by Class")
        plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
        plt.ylabel("Log (1 + Amount)")
        plt.show()

        # Distribution of transaction time
        plt.figure(figsize=(10, 5))

        sns.histplot(
            data=df,
            x="Time",
            hue="Class",
            bins=50,
            stat="percent",
            common_norm=False,
            element="step",
            fill=False,
            alpha=0.9,
            palette={0: "blue", 1: "red"}
        )

        plt.title("Transaction Time Distribution by Class")
        plt.xlabel("Time (since first transaction)")
        plt.ylabel("Percentage within Class")
        plt.legend(title="Class", labels=["Fraud", "Non-Fraud"])
        plt.tight_layout()
        plt.show()

        # Individual V1 - V28 correlation with Class. Non have a strong value
        # So weaker linear individual relationships with Class.
        class_corr = (df.drop(columns=["LogAmount",
                                       "Time", "Amount"])
                      .corr()["Class"]
                      .drop("Class")
                      )

        plt.figure(figsize=(9, 8))
        class_corr.plot(kind="barh")
        plt.title("Correlation of Features with Fraud Class")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.show()

        pca_features = [f"V{i}" for i in range(1, 29)]

        # Same with Amount and time
        class_corr = (df.drop(columns=pca_features + ["LogAmount"])
                      .corr()["Class"]
                      .drop("Class"))

        plt.figure(figsize=(9, 8))
        class_corr.plot(kind="barh")
        plt.title("Correlation of Features with Fraud Class")
        plt.xlabel("Correlation")
        plt.tight_layout()
        plt.show()

    '''
    Since our target variable is categorical (binary in this case whether
    fraud or not), let's first try a logistic regression model
    '''
    def logistic_regression_model():

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

        # Let's do a 70% training, 15% validation, 15% test split.
        test_size = 0.15
        train_size = 0.7
        val_size = 0.15

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

        print("Training samples:", len(X_train))
        print("Validation samples:", len(X_val))
        print("Test samples:", len(X_test))

        # Even though we have mostly PCA variables, I will be
        # using amount and time, which can be large.
        # Models that rely on distance, similarity, or gradient
        # magnitude—such as k‑NN, SVMs, logistic regression,
        # neural networks, and k-means/clustering algorithms—need feature
        # scaling to perform correctly. Also when doing PCA (non-model)
        # you do it.
        scaler = StandardScaler()  # Manual way to scale, can also use pipeline
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)

        # Hyperparameters of Logistic Regression model:
        # Maximum Iterations: How long solver to run
        # Class Weight: Adjust the penalty for misclassifying each class,
        # useful for imbalanced datasets.
        # Regularization Strength (C): How strongly coefficents are penalized
        # Penalty type l1 vs l2 (lasso vs ridge - default).
        # Hyperparameter testing
        # for i in [1, 0]:
        #     for C in [0.01, 0.05, 0.1, 0.5, 1, 5]:

        #         model = LogisticRegression(
        #             C=C,
        #             # whether l1 or l2, (values between 0,1 are elastic)
        #             l1_ratio=i,
        #             max_iter=2000,
        #             # Need solver that supports ridge and lasso (l1 and l2)
        #             solver="saga"
        #         )

        #         model.fit(X_train_scaled, y_train)
        #         y_val_pred = model.predict(X_val_scaled)

        #         print(
        #             f"C={C}, l1_ratio={i}, "
        #             f"Precision={precision_score(y_val, y_val_pred,
        #                                          zero_division=0):.3f}, "
        #             f"Recall={recall_score(y_val, y_val_pred,
        #                                    zero_division=0):.3f}, "
        #             f"F1={f1_score(y_val, y_val_pred, zero_division=0):.3f}"
        #         )

        # Can now use combined training/validation data
        X_train = X_temp
        y_train = y_temp

        # NOTE: Regularization makes coefficients of unimportant features
        # small (L1 goes to 0, L2 goes really small). So don't need to
        # remove features

        # Pipeline is easier
        final_model = make_pipeline(
                # After testing hyperparams, base offered best recall
                # Scaling over the PCA hurt performance
                LogisticRegression(C=0.5)
            )
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        # Since we are doing classification, we should test Accuracy,
        # Precision, Recall, and F1-Score

        # First get our confusion matrix, since binary classification it's 2x2
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Fraud (0)", "Fraud (1)"]
        )

        disp.plot()
        plt.show()

        # TN = cm[0, 0]
        # TP = cm[1, 1]
        # FN = cm[1, 0]
        # FP = cm[0, 1]

        # print("TN:", TN)
        # print("FP:", TP)
        # print("FN:", FN)
        # print("TP:", FP)

        # Math behind it
        # accuracy = (TP+TN)/(TP+TN+FP+FN)
        # precision = TP/(TP+FP)
        # recall = TP/(TP+FN)
        # f1 = 2*(precision*recall)/(precision+recall)) 

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))  # 0.791
        print("Recall:", recall_score(y_test, y_pred))  # 0.7162
        print("F1:", f1_score(y_test, y_pred))  # .7517

        print()

        # Since credit card fraud is only about 0.173% of transactions,
        # accuracy alone isn't enough, since most transactions are legit.
        # Precision and recall would be more helpful here along with F1.
        # Recall especially.

    # Run the code

    # eda()

    logistic_regression_model()


if __name__ == '__main__':
    main()
