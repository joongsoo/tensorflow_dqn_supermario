# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from data.env import Env
from tensorflow.python.framework.errors_impl import NotFoundError

class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = self.env.action_n

        self.max_episodes = 1500
        self.val = 0
        self.save_path = "./save/save_model"

    def generate_action(self, predict):
        key_up_down = np.argmax(predict[0][0:3])
        key_left_right = np.argmax(predict[0][3:6])
        key_a = np.argmax(predict[0][6:8])
        key_b = np.argmax(predict[0][8:10])

        action = [0, 0, 0, 0, 0, 0]
        if key_up_down == 0:
            action[0] = 1
        elif key_up_down == 1:
            action[1] = 1
        if key_left_right == 0:
            action[2] = 1
        elif key_left_right == 1:
            action[3] = 1
        if key_a == 0:
            action[4] = 1
        if key_b == 0:
            action[5] = 1

        return action

    def control_start(self):
        import dqn
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size,
                              name="main", is_training=False)
            tf.global_variables_initializer().run()

            mainDQN.restore(100)

            for episode in range(self.max_episodes):
                done = False
                clear = False
                state = self.env.reset()

                while not done and not clear:
                    action = self.generate_action(mainDQN.predict(state))
                    next_state, reward, done, clear, max_x = self.env.step(action)
                    state = next_state

def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()

if __name__ == "__main__":
    main()
