# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.keep_prob = 0.7

        self._build_network()
        self.saver = tf.train.Saver()
        self.save_path = "./save/save_model_" + name

    def _build_network(self, h_size=16, l_rate=0.001):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            net = self._X
            keep_prob = self.keep_prob

            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.relu)
            self._Qpred = net
            '''
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")

            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            layer1 = tf.nn.relu(tf.matmul(self._X, W1))

            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())

            self._Qpred = tf.matmul(layer1, W2)
            '''

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def save(self):
        self.saver.save(self.session, self.save_path)

    def restore(self):
        self.saver.restore(self.session, self.save_path)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        predict = self.session.run(self._Qpred, feed_dict={self._X: x})
        predict = np.reshape(predict, [self.output_size])
        res = []
        for idx in range(len(predict)):
            if predict[idx] > 0:
                res.append(int(1))
            else:
                res.append(int(0))
        return [[res]]

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack,
            self._Y: y_stack
        })
