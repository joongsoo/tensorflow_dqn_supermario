# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name="main", is_training=True):
        self.session = session
        self.input_size = input_size

        self.output_size = output_size

        self.net_name = name

        if is_training:
            self.keep_prob = 0.7
        else:
            self.keep_prob = 1.0

        self._build_network()
        self.saver = tf.train.Saver()
        self.save_path = "./save/save_model_" + self.net_name + ".ckpt"
        tf.logging.info(name + " - initialized")

    def _build_network(self, l_rate=0.001):
        with tf.variable_scope(self.net_name):
            keep_prob = self.keep_prob

            # input place holders
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self.X_img = tf.reshape(self._X, [-1, 100, 100, 3])

            # Conv
            W1 = tf.Variable(tf.random_normal([5, 5, 3, 16], stddev=0.01))
            net = tf.nn.conv2d(self.X_img, W1, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                                 strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.dropout(net, keep_prob=keep_prob)

            # Conv
            W2 = tf.Variable(tf.random_normal([3, 3, 16, 32], stddev=0.01))
            net = tf.nn.conv2d(net, W2, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.dropout(net, keep_prob=keep_prob)

            # Conv
            W3 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))
            net = tf.nn.conv2d(net, W3, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.relu(net)
            print net
            net = tf.reshape(net, [-1, 13 * 13 * 64])

            net = tf.layers.dense(net, 2000, activation=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            net = tf.layers.dense(net, 4000, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.output_size)
            self._Qpred = net

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        #self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        self._train = tf.train.RMSPropOptimizer(
            l_rate, momentum=0.95, epsilon=0.01).minimize(self._loss)

    def save(self, episode=0):
        self.saver.save(self.session, self.save_path, global_step=episode)

    def restore(self, episode=0):
        load_path = self.save_path + "-" + str(episode)
        self.saver.restore(self.session, load_path)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        predict = self.session.run(self._Qpred, feed_dict={self._X: x})
        return predict

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack,
            self._Y: y_stack
        })
