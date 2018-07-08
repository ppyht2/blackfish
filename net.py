from constants import BOARD_SIZE, N_ACTIONS, N_FEATURE
import tensorflow as tf
import numpy as np
import joblib


N_PLANES = 13
DEFAULT_N_FILTER = 256

N_POLICY_FILTER = 2
N_VALUE_FILTER = 1

MODEL_NAME = 'model.params'


class Model():

    def __init__(self,  n_residual):
        self.n_residual = n_residual

        self.build_graph()

    def build_graph(self, reset=False):
        if reset:
            tf.reset_default_graph()

        self.state = tf.placeholder(tf.float32,
                                    [None, BOARD_SIZE, BOARD_SIZE, N_FEATURE])
        self.train_ind = tf.placeholder(tf.bool, [])

        h = convolution_block(self.state, self.train_ind)

        for i in range(self.n_residual):
            h = residual_block(h, self.train_ind)

            logits = policy_head(h, self.train_ind)
            self.policy = tf.nn.softmax(logits)
            self.value = value_head(h, self.train_ind)

        # Training

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.params = tf.trainable_variables()

    def evaluate(self, inputs):
        return self.sess.run([self.value, self.policy],
                             feed_dict={self.state: inputs, self.train_ind: False})

    def save(self):
        ps = self.sess.run(self.params)
        joblib.dump(ps, MODEL_NAME)

    def load(self):
        ps = joblib.load(MODEL_NAME)
        restore_ops = []
        for t, v in zip(self.params, ps):
            restore_ops.append(t.assign(v))
        self.sess.run(restore_ops)


def convolution_block(inputs, train_ind):
    conv = tf.layers.conv2d(inputs, filters=DEFAULT_N_FILTER,
                            kernel_size=3, strides=(1, 1), padding='SAME')
    batch = tf.layers.batch_normalization(conv, training=train_ind)
    act = tf.nn.relu(batch)
    return act


def residual_block(inputs, train_ind):
    conv1 = tf.layers.conv2d(inputs, filters=DEFAULT_N_FILTER,
                             kernel_size=3, strides=(1, 1), padding='SAME')
    batch1 = tf.layers.batch_normalization(conv1, training=train_ind)
    act1 = tf.nn.relu(batch1)
    conv2 = tf.layers.conv2d(act1, filters=DEFAULT_N_FILTER,
                             kernel_size=3, strides=(1, 1), padding='SAME')
    batch2 = tf.layers.batch_normalization(conv2, training=train_ind)
    connect = batch2 + inputs
    act2 = tf.nn.relu(connect)
    return act2


def policy_head(inputs, train_ind):
    conv = tf.layers.conv2d(inputs, filters=N_POLICY_FILTER, kernel_size=1, padding='SAME')
    batch = tf.layers.batch_normalization(conv, training=train_ind)
    act = tf.nn.relu(batch)
    flat = tf.layers.flatten(act)
    logits = tf.layers.dense(flat, N_ACTIONS)
    return logits


def value_head(inputs, train_ind):
    conv = tf.layers.conv2d(inputs, filters=N_VALUE_FILTER, kernel_size=1, padding='SAME')
    batch = tf.layers.batch_normalization(conv, training=train_ind)
    act1 = tf.nn.relu(batch)
    dense = tf.layers.dense(act1, 256)
    act2 = tf.nn.relu(dense)
    flat = tf.layers.flatten(act2)
    dense2 = tf.layers.dense(flat, 1)
    value = tf.nn.tanh(dense2)
    return value


if __name__ == "__main__":
    n_plane = 13
    model = Model(n_plane, 3)
    model.build_graph()
    state = np.random.rand(1, BOARD_SIZE, BOARD_SIZE, n_plane)
    value, policy = model.evaluate(state)
    # For now, just test it gives output
    print(value)
    print(policy)
