import tensorflow as tf
import numpy as np
import pandas as pd
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


def normalize(data_frame_encoded):
    """Normalize the data using log function."""
    data = data_frame_encoded[:, 0]
    encoders = data_frame_encoded[:, 1]
    data = [np.piecewise(tt, [tt > 1., tt <= 1.], [lambda tt: np.log(tt), lambda tt: tt]) for tt in data]
    return np.array([[d, e] for d, e in zip(data, encoders)])


df_train = pd.read_csv('./data/train.csv', keep_default_na=False)
# drop useless data
df_train = df_train.drop(['Id', 'Street', 'LotFrontage', 'LandSlope', 'YearBuilt', 'YearRemodAdd',
                          'MasVnrArea', 'Foundation', 'GarageYrBlt', 'MoSold', 'YrSold'], 1)
column_names = df_train.columns.values
df_train_encoded = oh_encode(df_train)
df_train_encoded_normalized = normalize(df_train_encoded)

batch_gen = batch_generator(df_train_encoded_normalized)
NUM_FEATURES = 299

# create the neural network model
input_layer = tf.placeholder(tf.float32, [None, NUM_FEATURES])
W1 = tf.Variable(tf.random_normal([NUM_FEATURES, 100], stddev=.1))
b1 = tf.Variable(tf.random_normal([100], stddev=.1))
h1_layer = tf.matmul(input_layer, W1) + b1
h1_layer = tf.nn.relu(h1_layer)

W2 = tf.Variable(tf.random_normal([100, 100], stddev=.1))
b2 = tf.Variable(tf.random_normal([100], stddev=.1))
h2_layer = tf.matmul(h1_layer, W2) + b2
h2_layer = tf.nn.relu(h2_layer)

W3 = tf.Variable(tf.random_normal([100, 1], stddev=.1))
b3 = tf.Variable(tf.random_normal([1], stddev=.1))
output_layer = tf.reduce_sum(tf.matmul(h2_layer, W3) + b3)
y = tf.placeholder(tf.float32, shape=[None, 1])

loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(y, output_layer))))
# cross_entropy = -tf.reduce_sum(y * tf.log(output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

all_examples = np.array([[np.array(b), l] for b, l in batch_gen])

# get the next batch
for i in range(50000):
    idx = np.random.randint(0, len(all_examples), 50)
    batches = all_examples[idx]
    labels = batches[:, 1]
    batches = np.array(batches[:, 0])
    batches = np.concatenate(batches)
    batches = np.reshape(batches, [50, 299])
    labels = np.reshape(labels, [50, 1])
    # feed the batch
    o, l = sess.run([optimizer, loss], feed_dict={input_layer: batches, y: labels})
    # log results
    if i % 500 == 0:
        print('epoch: {}, loss: {}'.format(i, l))

sess.close()
