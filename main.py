import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def oh_encode(data_frame):
    """Encodes categorical columns into their One-Hot representations."""
    data_encoded = []
    for feature in data_frame:
        data_i = df_train[feature]
        encoder = None
        if df_train[feature].dtype == 'O' or feature == "MSSubClass":  # is data categorical?
            encoder = LabelBinarizer()
            encoder.fit(list(set(df_train[feature])))
            data_i = encoder.transform(data_i)
        data_i = np.array(data_i, dtype=np.float32)
        data_encoded.append([data_i, encoder])
    return np.array(data_encoded)


def normalize(data_frame_encoded):
    """Normalize the data using log function."""
    data = data_frame_encoded[:, 0]
    encoders = data_frame_encoded[:, 1]
    data = [np.log(tt + 1) for tt in data]
    return np.array([[d, e] for d, e in zip(data, encoders)])


def batch_generator(data_frame_encoded):
    """Generates data to be fed to the neural network."""
    labels = data_frame_encoded[-1][0]
    data = np.delete(data_frame_encoded, -1, axis=0)
    data = data_frame_encoded[:, 0]
    # data = [np.array(d) for d in data]

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
df_train = df_train.drop(['Id', 'Street', 'LotFrontage', 'LandSlope', 'YearBuilt', 'YearRemodAdd',
                          'MasVnrArea', 'Foundation', 'GarageYrBlt', 'MoSold', 'YrSold'], 1)
column_names = df_train.columns.values
df_train_encoded = oh_encode(df_train)
df_train_encoded_normalized = normalize(df_train_encoded)

batch_gen = batch_generator(df_train_encoded_normalized)
NUM_FEATURES = 299

all_examples = np.array([[np.array(b), l] for b, l in batch_gen])
X_train, X_test, y_train, y_test = train_test_split(all_examples[:, 0], all_examples[:, 1], test_size=0.25)
X_train = np.concatenate(X_train)
X_train = np.reshape(X_train, [-1, 299])

X_test = np.concatenate(X_test)
X_test = np.reshape(X_test, [-1, 299])
clf = SGDRegressor()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print(score)
