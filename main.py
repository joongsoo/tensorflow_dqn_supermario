# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from collections import deque
from data.env import Env
from tensorflow.python.framework.errors_impl import NotFoundError


class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = self.env.action_n

        #self.dis = 0.9
        self.dis = 1.3
        self.REPLAY_MEMORY = 10000
        self.max_episodes = 15000
        self.replay_buffer = deque()
        self.val = 0
        self.save_path = "./save/save_model"

    def replay_train(self, mainDQN, targetDQN, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size*2)


        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)

            if not done:
                reward = reward + self.dis * np.max(targetDQN.predict(next_state))

            for idx in range(len(action)):
                Q[idx, action[idx]] = reward

            state = np.reshape(state, [self.input_size])
            Q = np.reshape(Q, [self.output_size*2])

            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state])
        return mainDQN.update(x_stack, y_stack)

    def get_copy_var_ops(self, dest_scope_name="target", src_scope_name="main"):
        op_holder = []

        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

    def control_start(self):
        import dqn
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")

            tf.global_variables_initializer().run()

            try:
                mainDQN.restore()
                targetDQN.restore()
            except NotFoundError:
                pass

            copy_ops = self.get_copy_var_ops()
            sess.run(copy_ops)

            for episode in range(self.max_episodes):
                e = 1. / ((episode / 10) + 1)
                done = False
                clear = False
                step_count = 0
                state = self.env.reset()
                max_x = 0

                while not done and not clear:
                    if np.random.rand(1) < e:
                        action = self.env.get_random_actions()
                    else:
                        predict = mainDQN.predict(state)
                        action = []
                        for p in predict:
                            action.append(np.argmax(p))

                    next_state, reward, done, clear, max_x = self.env.step(action)


                    if clear:
                        reward += 10000

                    if done:
                        reward = -10000

                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > self.REPLAY_MEMORY:
                        self.replay_buffer.popleft()

                    state = next_state
                    step_count += 1


                print("Episode: {}  steps: {}  max_x: {}".format(episode, step_count, max_x))

                if episode % 50 == 0:
                    mainDQN.save()
                    targetDQN.save()

                #minibatch = random.sample(self.replay_buffer, int(len(self.replay_buffer) / 100))
                loss = self.replay_train(mainDQN, targetDQN, self.replay_buffer)

                self.replay_buffer = deque()

                print("Loss: ", loss)
                sess.run(copy_ops)

def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()
