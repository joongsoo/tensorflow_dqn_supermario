# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class DQN:
    def __init__(self, session, input_size, output_size, name="main", keep_prob=0.7):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        self.keep_prob = keep_prob

        self._build_network()
        self.saver = tf.train.Saver()
        self.save_path = "./save/save_model_" + name + "ckpt"
        tf.logging.info(name + " - initialized")

    def _build_network(self, h_size=150, l_rate=0.001):
        with tf.variable_scope(self.net_name):
            #self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            #net = self._X
            keep_prob = self.keep_prob


            # input place holders
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self.X_img = tf.reshape(self._X, [-1, 100, 100, 1])

            # Conv
            W1 = tf.Variable(tf.random_normal([2, 2, 1, 20], stddev=0.01))
            net = tf.nn.conv2d(self.X_img, W1, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.dropout(net, keep_prob=keep_prob)

            # Conv
            W2 = tf.Variable(tf.random_normal([2, 2, 20, 40], stddev=0.01))
            net = tf.nn.conv2d(net, W2, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.dropout(net, keep_prob=keep_prob)

            net = tf.reshape(net, [-1, 7*7*40])

            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=keep_prob)
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=keep_prob)

            net = tf.layers.dense(net, self.output_size*2)
            #net = tf.reshape(net, [self.output_size, 2])
            self._Qpred = net

        self._Y = tf.placeholder(shape=[None, self.output_size*2], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        #self._loss = -tf.reduce_mean(self._Y * tf.log(self._Qpred) + (1 - self._Y) * tf.log(1 - self._Qpred))
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

    def save(self, global_step=0):
        self.saver.save(self.session, self.save_path)

    def restore(self, global_step=0):
        self.saver.restore(self.session, self.save_path)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        predict = self.session.run(self._Qpred, feed_dict={self._X: x})
        predict = np.reshape(predict, [self.output_size, 2])
        return predict

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={
            self._X: x_stack,
            self._Y: y_stack
        })
