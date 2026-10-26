import pandas as pd
import numpy as np


def load_data():
    data = pd.read_csv("./data/data.csv")

    # Need to convert the panda column objects (panda series) to numpy array via .values
 
    # Since we are classifying whether fraudulent or not, we do .astype(np.int64) for y
    y = data["Class"].values.astype(np.int64)

    # Features will be floats this time and we need to
    time_amount = data[["Time", "Amount"]].values.astype(np.float32)

    return x_train, y_train, x_val, y_val