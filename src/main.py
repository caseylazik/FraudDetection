from utilities import load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():

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


    # Run the code

    # eda()

    logistic_regression_model







if __name__ == '__main__':
    main()
