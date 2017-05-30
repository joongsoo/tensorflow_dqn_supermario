# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from data.env import Env
from tensorflow.python.framework.errors_impl import NotFoundError

class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = 12

        self.max_episodes = 1500
        self.val = 0
        self.save_path = "./save/save_model"

    def control_start(self):
        import dqn
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size,
                              name="main", is_training=False)
            tf.global_variables_initializer().run()

            mainDQN.restore(750)

            for episode in range(self.max_episodes):
                done = False
                clear = False
                state = self.env.reset()

                while not done and not clear:
                    action = np.argmax(mainDQN.predict(state))
                    next_state, reward, done, clear, max_x, _, _ = self.env.step(action)
                    state = next_state

def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()

if __name__ == "__main__":
    main()
