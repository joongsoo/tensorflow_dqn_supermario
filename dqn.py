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

    def _build_network(self, l_rate=0.03):
        with tf.variable_scope(self.net_name):
            keep_prob = self.keep_prob

            # input place holders
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name="input_x")
            self._keep_prob = tf.placeholder(tf.float32, name="kp")
            self.X_img = tf.reshape(self._X, [-1, 120, 120, 1])

            # Conv
            W1 = tf.Variable(tf.random_normal([4, 4, 1, 16], stddev=0.01))
            net = tf.nn.conv2d(self.X_img, W1, strides=[1, 3, 3, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(net, ksize=[1, 4, 4, 1],
                                 strides=[1, 2, 2, 1], padding='SAME')
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)

            # Conv
            W2 = tf.Variable(tf.random_normal([2, 2, 16, 32], stddev=0.01))
            net = tf.nn.conv2d(net, W2, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)

            # Conv
            W3 = tf.Variable(tf.random_normal([2, 2, 32, 64], stddev=0.01))
            net = tf.nn.conv2d(net, W3, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)

            # Conv
            W3 = tf.Variable(tf.random_normal([2, 2, 64, 128], stddev=0.01))
            net = tf.nn.conv2d(net, W3, strides=[1, 2, 2, 1], padding='SAME')
            net = tf.nn.relu(net)
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)

            print net
            net = tf.reshape(net, [-1, 3 * 3 * 128])

            net = tf.layers.dense(net, 2048, activation=tf.nn.relu)
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu)
            #net = tf.nn.dropout(net, keep_prob=self._keep_prob)
            net = tf.layers.dense(net, self.output_size)
            self._Qpred = net

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        #self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

        self._train = tf.train.RMSPropOptimizer(
            l_rate, momentum=0.95, epsilon=0.01).minimize(self._loss)


        correct_prediction = tf.equal(tf.argmax(self._Qpred, 1), tf.argmax(self._Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def save(self, episode=0):
        self.saver.save(self.session, self.save_path+ "-" + str(episode))

    def restore(self, episode=0):
        load_path = self.save_path + "-" + str(episode)
        self.saver.restore(self.session, load_path)

    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        predict = self.session.run(self._Qpred, feed_dict={self._X: x, self._keep_prob: 1.0})
        return predict

    def val_test(self, x_stack, y_stack):
        return self.session.run([self._Qpred, self._Y], feed_dict={
            self._X: x_stack,
            self._Y: y_stack,
            self._keep_prob: 0.7
        })

    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train, self.accuracy], feed_dict={
            self._X: x_stack,
            self._Y: y_stack,
            self._keep_prob: 0.7
        })
