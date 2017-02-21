import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

TRAINING_EXAMPLES = 128
NUM_EPOCHS = 16000
HIDDEN_SIZE = 500
num_features = 835


def oh_encode(data_frame):
    """Encodes categorical columns into their One-Hot representations."""
    data_encoded = []
    for feature in data_frame:
        data_i = df_train[feature]
        encoder = None
        if df_train[feature].dtype == 'O':  # is data categorical?
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
    data = data[:, 0]

    num_features = len(data)
    num_batches = len(data[0])
    for i in range(num_batches):
        batch_compiled = []
        for j in range(num_features):
            if type(data[j][i]) is np.ndarray:
                batch_compiled.extend(data[j][i])
            else:
                batch_compiled.extend([data[j][i]])
        yield batch_compiled, labels[i]


df_train = pd.read_csv('./data/train.csv', keep_default_na=False)
# drop useless data
# df_train = df_train[['LotArea', 'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', '1stFlrSF', '2ndFlrSF',
#                      'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#                      'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'Functional', 'GarageType', 'WoodDeckSF',
#                      'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'Fence',
#                      'GarageArea', 'SalePrice']]
df_train = df_train.drop(['Id'], 1)
column_names = df_train.columns.values
df_train_encoded = oh_encode(df_train)
df_train_encoded_normalized = normalize(df_train_encoded)

batch_gen = batch_generator(df_train_encoded_normalized)

# create the neural network model
keep_prob = tf.placeholder(tf.float32)
# learning_rate = tf.Variable(1e-1, trainable=False)
prev_loss = tf.Variable(0., trainable=False)

input_layer = tf.placeholder(tf.float32, [None, num_features])
W1 = tf.Variable(tf.random_normal([num_features, HIDDEN_SIZE], stddev=.01))
b1 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01))
h1_layer = tf.matmul(input_layer, W1) + b1
h1_layer = tf.nn.relu(h1_layer)
h1_layer = tf.nn.dropout(h1_layer, keep_prob)

W2 = tf.Variable(tf.random_normal([HIDDEN_SIZE, HIDDEN_SIZE], stddev=.01))
b2 = tf.Variable(tf.random_normal([HIDDEN_SIZE], stddev=.01))
h2_layer = tf.matmul(h1_layer, W2) + b2
h2_layer = tf.nn.relu(h2_layer)
h2_layer = tf.nn.dropout(h2_layer, keep_prob)

W3 = tf.Variable(tf.random_normal([HIDDEN_SIZE, 1], stddev=.01))
b3 = tf.Variable(tf.random_normal([1], stddev=.01))
output_layer = tf.reduce_sum(tf.matmul(h2_layer, W3) + b3)
y = tf.placeholder(tf.float32, shape=[None, 1])

# loss = tf.divide(tf.reduce_sum(tf.square(tf.subtract(y, output_layer))), 2.)
loss = tf.subtract(y, output_layer)
loss = tf.square(loss)
loss = tf.reduce_sum(loss)
loss = tf.divide(loss, 2.)
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_constant = 0.01  # Choose an appropriate one.
loss = loss + reg_constant * sum(reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-7).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

all_examples = np.array([[np.array(b), l] for b, l in batch_gen])
# split data into train and validation
train_examples = all_examples[:1150]
valid_examples = all_examples[1150:]
valid_labels = valid_examples[:, 1]
valid_labels = np.reshape(valid_labels, [-1, 1])

valid_batches = np.array(valid_examples[:, 0])
valid_len = len(valid_batches)
valid_batches = np.concatenate(valid_batches)
valid_batches = np.reshape(valid_batches, [valid_len, -1])
saver = tf.train.Saver(max_to_keep=15)
min_loss = 50000000

# get the next batch
for i in range(NUM_EPOCHS):
    idx = np.random.randint(0, len(train_examples), TRAINING_EXAMPLES)
    train_batches = train_examples[idx]
    train_labels = np.array([train_batches[:, 1]])
    train_labels = np.reshape(train_labels, [TRAINING_EXAMPLES, 1])

    train_batches = np.array(train_batches[:, 0])
    train_batches = np.concatenate(train_batches)
    train_batches = np.reshape(train_batches, [TRAINING_EXAMPLES, -1])
    # feed the batch
    _, train_loss = sess.run([optimizer, loss], feed_dict={input_layer: train_batches, y: train_labels, keep_prob: .5})
    valid_loss = sess.run(loss, feed_dict={input_layer: valid_batches, y: valid_labels, keep_prob: 1.})
    # log results
    if i % 1000 == 0:
        print('epoch: {}, train loss: {}, valid loss: {}'.format(i, train_loss, valid_loss))
    if valid_loss < min_loss:
        min_loss = valid_loss
        save_path = saver.save(sess, "./saves/model{}.ckpt".format(valid_loss))
        print('Best model so far: loss: {}, epoch: {}'.format(valid_loss, i))

sess.close()
