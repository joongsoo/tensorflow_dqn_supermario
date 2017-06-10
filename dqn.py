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
            self._keep_prob = tf.placeholder(tf.float32, name="kp")
            self.X_img = tf.reshape(self._X, [-1, 120, 120, 1])

            # Conv
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            L1 = tf.nn.conv2d(self.X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

            # L2 ImgIn shape=(?, 14, 14, 32)
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
            L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

            # L3 ImgIn shape=(?, 7, 7, 64)
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                1, 2, 2, 1], padding='SAME')
            L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

            L3 = tf.reshape(L3, [-1, 128 * 15 * 15])

            # L4 FC 4x4x128 inputs -> 625 outputs
            W4 = tf.get_variable("W4", shape=[128 * 15 * 15, 625],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

            # L5 Final FC 625 inputs -> 10 outputs
            W5 = tf.get_variable("W5", shape=[625, self.output_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([self.output_size]))

            hypothesis = tf.matmul(L4, W5) + b5

            # cost/loss function
            # cost = tf.reduce_mean(tf.square(Y - hypothesis))
            #net = tf.layers.dense(net, self.output_size)
            self._Qpred = hypothesis

        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)

        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))

        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)

        #self._train = tf.train.RMSPropOptimizer(
        #    l_rate, momentum=0.95, epsilon=0.01).minimize(self._loss)


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
