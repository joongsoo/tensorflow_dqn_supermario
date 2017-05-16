# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import dqn
from collections import deque
from data.env import Env
import multiprocessing
from multiprocessing import Manager
import time


class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = self.env.action_n

        self.dis = 0.9
        self.REPLAY_MEMORY = 9000
        self.max_episodes = 1500
        self.replay_buffer = deque()
        self.val = 0
        self.save_path = "./save/save_model"

    def control_start(self):
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            tf.global_variables_initializer().run()

            mainDQN.restore()

            for episode in range(40, self.max_episodes):
                done = False
                step_count = 0
                state = self.env.reset()
                max_x = 0

                while not done:
                    action = np.argmax(mainDQN.predict(state))
                    print action
                    max_x = self.env.step_play(action)
                    step_count += 1

                print("Episode: {}  steps: {}  max_x: {}".format(episode, step_count, max_x))


def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()
