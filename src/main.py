import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from utilities import load_data, seed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from utilities import find_optimal_classification_threshold

def main():

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

        (X_temp, X_test, y_temp, y_test,
         X_train, X_val, y_train, y_val) = load_data()

        print("Training samples:", len(X_train))
        print("Validation samples:", len(X_val))
        print("Test samples:", len(X_test))

        # Even though we have mostly PCA variables, I will be
        # using amount and time, which can be large.
        # Models that rely on distance, similarity, or gradient
        # magnitude—such as k‑NN, SVMs, logistic regression,
        # neural networks, and k-means/clustering algorithms—need feature
        # scaling to perform correctly. Also when doing PCA (non-model)
        # scaler = StandardScaler()  # Manual scale, can also use pipeline
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_val_scaled = scaler.transform(X_val)

        # df = pd.read_csv("../data/data.csv")
        # X = df.drop(columns=["Class"])

        # X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        # X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)

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
        #             f"Precision={precision_score(y_val, y_val_pred):.3f}, "
        #             f"Recall={recall_score(y_val, y_val_pred):.3f}, "
        #             f"F1={f1_score(y_val, y_val_pred):.3f}"
        #         )

        # NOTE: Regularization makes coefficients of unimportant features
        # small (L1 goes to 0, L2 goes really small). So don't need to
        # remove features

        # Pipeline is easier for preprocessing, like adding standard scalar, 
        # or PCA. Can still use if not but uneccessary.
        # After testing hyperparams, base offered best recall
        # Scaling over the PCA hurt performance # L2 - Ridge
        final_model = LogisticRegression(C=0.5, l1_ratio=0, max_iter=10000)

        threshold = find_optimal_classification_threshold(final_model, X_val,
                                                          y_val, X_train, 
                                                          y_train)

        # Can now use combined training/validation data
        X_train = X_temp
        y_train = y_temp

        final_model.fit(X_train, y_train)

        y_prob = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)


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
        # base  # 0.791    # 0.7162   # .7517
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))  
        print("Recall:", recall_score(y_test, y_pred))   
        print("F1:", f1_score(y_test, y_pred))   

        # With threshold: Precision: 0.7402597402597403
        # Recall: 0.7702702702702703
        # F1: 0.7549668874172185

        print()

        # Since credit card fraud is only about 0.173% of transactions,
        # accuracy alone isn't enough, since most transactions are legit.
        # Precision and recall would be more helpful here along with F1.
        # Recall especially.

    def random_forest():
        (X_temp, X_test, y_temp, y_test,
         X_train, X_val, y_train, y_val) = load_data()

        # best_f1 = 0
        # best_params = []
        # cur_forest = 1

        # for n_est in [100, 250, 400]:
        #     for max_depth in [None, 10, 20, 30]:
        #         for min_samples_leaf in [1, 2, 4]:
        #             for min_samples_split in [2, 5, 10]:
        #                 for max_features in ["sqrt", 0.5]:
        #                     print("Forest number", cur_forest)
        #                     cur_forest += 1
        #                     # Need seed because generating forest is random.
        #                     # Also, n_jobs=-1 speeds up process by using all
        #                     # cpu cores. Pipeline still works but not
        #                     # necessary since no preprocessing
        #                     cur_model = RandomForestClassifier(n_estimators=n_est,
        #                                                        max_depth=max_depth,
        #                                                        min_samples_leaf=min_samples_leaf,
        #                                                        min_samples_split=min_samples_split,
        #                                                        max_features=max_features,n_jobs=-1,
        #                                                        random_state=seed())

        #                     cur_model.fit(X_train, y_train)
        #                     y_val_pred = cur_model.predict(X_val)

        #                     cur_f1 = f1_score(y_val, y_val_pred)

        #                     if cur_f1 > best_f1: # Let's try to prioritize f1
        #                         best_f1 = cur_f1
        #                         best_params = [n_est, max_depth,
        #                                        min_samples_leaf,
        #                                        min_samples_split,
        #                                        max_features]
        #                         print(best_f1)

        # print(best_params, best_f1)  # [100, None, 1, 2, 0.5] 0.89208, on validation data

        # Again baseline model is the best
        final_model = RandomForestClassifier(random_state=seed())

        threshold = find_optimal_classification_threshold(final_model, X_val,
                                                          y_val, X_train,
                                                          y_train)

        # Combine together
        X_train = X_temp
        y_train = y_temp

        # final_model.fit(X_train, y_train)

        y_proba = final_model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        # Since we are doing classification, test Accuracy/Precision/Recall/F1

        # First get our confusion matrix, since binary classification it's 2x2
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Fraud (0)", "Fraud (1)"]
        )

        disp.plot()
        plt.show()

        # Way better than logistic regression
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))  # 0.9508
        print("Recall:", recall_score(y_test, y_pred))  # 0.78378
        print("F1:", f1_score(y_test, y_pred))  # 0.8592

    def neural_network():
        (X_temp, X_test, y_temp, y_test,
         X_train, X_val, y_train, y_val) = load_data()

        # NEED standard scalar otherwise awful.

        # Uses L2 by default
        # mlp = make_pipeline(StandardScaler(), MLPClassifier(
        #     random_state=seed(),
        #     early_stopping=True,
        #     activation='logistic', # Sigmoid since binary classification.
        #     solver='sgd'
        # ))

        # since using pipeline, need to prefix with mlpclassifier
        # param_grid = {
        #     'mlpclassifier__hidden_layer_sizes': [
        #         (64, 64),
        #         (64, 128),
        #         (256, 128),
        #         (256, 128, 64)
        #     ],
        #     'mlpclassifier__batch_size': [64, 128, 256],
        #     'mlpclassifier__learning_rate_init': [0.0001, 0.001, 0.005, 0.01],
        #     'mlpclassifier__alpha': [0.0001, 0.001, 0.01, 0.1], # Since using L2, this controls that
        # }

        # grid = GridSearchCV(
        #     estimator=mlp,
        #     param_grid=param_grid,
        #     cv=3,
        #     scoring='f1',
        #     n_jobs=-1,
        #     verbose=1  # print progress
        # )

        # grid.fit(X_train, y_train)

        # print("Best parameters:", grid.best_params_)
        # print("Best score:", grid.best_score_)

        # Best parameters: {'mlpclassifier__alpha': 0.0001,
        #                   'mlpclassifier__batch_size': 64,
        #                   'mlpclassifier__hidden_layer_sizes': (64, 64),
        #                   'mlpclassifier__learning_rate_init': 0.01}
        # Best score: 0.8223367329542938

        # Precision: 0.90476   Recall: 0.7702702 F1: 0.83211 - base
        final_model = make_pipeline(StandardScaler(),
                                    MLPClassifier(random_state=seed(),
                                                  activation="logistic",
                                                  early_stopping=True))

        threshold = find_optimal_classification_threshold(final_model, X_val,
                                                          y_val, X_train,
                                                          y_train)

        X_train = X_temp
        y_train = y_temp

        final_model.fit(X_train, y_train)

        y_prob = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(cm)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Fraud (0)", "Fraud (1)"]
        )

        disp.plot()
        plt.show()

        # NN classifier better than logistic, worse than random forest
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))   
        print("Recall:", recall_score(y_test, y_pred))   
        print("F1:", f1_score(y_test, y_pred))   

    # Run the code

    # eda()

    logistic_regression_model()

    # random_forest()

    # neural_network()


if __name__ == '__main__':
    main()
