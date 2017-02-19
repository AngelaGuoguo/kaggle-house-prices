#import tensorflow as tf
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


def batch_generator(data_frame_encoded):
    """Generates data to be fed to the neural network."""
    labels = data_frame_encoded['SalePrice'][0]
    del data_frame_encoded['SalePrice']
    data = np.array(list(data_frame_encoded.values()))[:, 0]
    data = [np.array(d) for d in data]

    NUM_FEATURES = len(data)
    NUM_BATCHES = len(data[0])
    for i in range(NUM_BATCHES):
        batch_compiled = []
        for j in range(NUM_FEATURES):
            if type(data[j][i]) is np.ndarray:
                batch_compiled.extend(data[j][i])
            else:
                batch_compiled.extend([data[j][i]])
        yield batch_compiled, labels[i]


df_train = pd.read_csv('./data/train.csv', keep_default_na=False)
# drop useless data
df_train = df_train.drop(['Street', 'LotFrontage', 'LandSlope', 'YearBuilt', 'YearRemodAdd',
                          'MasVnrArea', 'Foundation', 'GarageYrBlt', 'MoSold', 'YrSold'], 1)
df_train_encoded = oh_encode(df_train)
batch_gen = batch_generator(df_train_encoded)
# print(next(batch_gen))
# print('*'*8)
# print(next(batch_gen))
