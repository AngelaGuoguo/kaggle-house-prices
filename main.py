import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def oh_encode(data_frame):
    """Encodes categorical columns into their One-Hot representations."""
    data_encoded = {}
    for feature in data_frame:
        data_i = df_train[feature]
        encoder = None
        if df_train[feature].dtype == 'O' or feature == "MSSubClass":  # is data categorical?
            encoder = LabelBinarizer()
            encoder.fit(list(set(df_train[feature])))
            data_i = encoder.transform(data_i)
        data_encoded[feature] = [data_i, encoder]
    return data_encoded


df_train = pd.read_csv('./data/train.csv', keep_default_na=False)
# drop useless data
df_train = df_train.drop(['Street', 'LotFrontage', 'LandSlope', 'YearBuilt', 'YearRemodAdd',
                          'MasVnrArea', 'Foundation', 'GarageYrBlt', 'MoSold', 'YrSold'], 1)
df_train_encoded = oh_encode(df_train)
# print(df_train_encoded)
