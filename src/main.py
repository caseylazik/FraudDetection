from utilities import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

 

def main():

    seed = 23

    def eda():
        df = pd.read_csv("../data/data.csv")

        print("\nData information:")
        df.info() # Time is how long since first transaction


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

        # Distribution plot of fraud vs no fraud transaction amounts
        df["LogAmount"] = np.log1p(df["Amount"]) # Must use log1p because instances of 0 Amount

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
        class_corr = (
        df.drop(columns=["LogAmount", "Time", "Amount"])
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
    Since our target variable is categorical (binary in this case whether fraud or not),
    let's first try a logistic regression model
    '''
    def logistic_regression_model():

        df = pd.read_csv("../data/data.csv")

        X = df.drop(columns=["Class"])
        y = df["Class"]

        # Since we have a super imbalanced class where 0.172749% of cases are fraudelent, and 99.82% are not,
        # we should stratify with stratify=y such that the model preserves roughly the same distribution between
        # these two classes. Stratify ensures that the training and test splits
        # have about the same proportions of classes (doesn't change test to train size ratio, just means that one
        # of the splits won't have much more fraud cases since so few to begin wwith) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y) # 20% data used for testing

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Since we are doing classification, we should test Accuracy, Precision, Recall, and F1-Score


        # First let's get our confusion matrix, since binary classification it's 2x2

        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

        print(cm)

         
 

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Fraud (0)", "Fraud (1)"]
        )

        disp.plot()
        plt.show()

                
        TN = cm[0, 0]
        TP = cm[1, 1]
        FN = cm[1, 0]
        FP = cm[0, 1]

        print("TN:", TN)
        print("FP:", TP)
        print("FN:", FN)
        print("TP:", FP)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, zero_division=0))
        print("Recall:", recall_score(y_test, y_pred, zero_division=0))
        print("F1:", f1_score(y_test, y_pred, zero_division=0))

        print()

        # Math behind it
        # accuracy = (TP+TN)/(TP+TN+FP+FN)
        # precision = TP/(TP+FP)
        # recall = TP/(TP+FN)

        # print("Accuracy:", accuracy) # How many predictions correct
        # print("Precision:",precision) # How many cases were actually fraud out of the predicted fraud cases
        # print("Recall:",recall) # How many fraud cases caught
        # print("F1:",2*(precision*recall)/(precision+recall)) # Balance between precision and recall


        # Since credit card fraud is only about 0.173% of transactions, accuracy alone isn't 
        # enough, since most transactions are legit. Precision and recall would be more helpful here
        # along with F1. Recall especially.




    # Run the code
    
    # eda()

    logistic_regression_model()


if __name__ == '__main__':
    main()
