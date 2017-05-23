# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from collections import deque
from data.env import Env
from tensorflow.python.framework.errors_impl import NotFoundError
import pygame



class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = 12

        #self.dis = 0.9
        self.dis = 0.9
        self.REPLAY_MEMORY = 15000
        self.max_episodes = 3000
        self.replay_buffer = deque()
        self.val = 0
        self.save_path = "./save/save_model"

    def replay_train(self, mainDQN, targetDQN, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)
            if done:
                Q[0, np.argmax(action)] = reward
            else:
                Q[0, np.argmax(action)] = reward + self.dis * np.max(targetDQN.predict(next_state))

            state = np.reshape(state, [self.input_size])
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
            mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")

            tf.global_variables_initializer().run()

            try:
                pass
                #mainDQN.restore(1400)
                #targetDQN.restore(1400)
            except NotFoundError:
                print "??"
                pass

            copy_ops = self.get_copy_var_ops()
            sess.run(copy_ops)

            episode = 0
            train = True

            while episode < self.max_episodes:
                e = 1. / ((episode / 10) + 1)
                done = False
                clear = False
                step_count = 0
                state = self.env.reset()
                max_x = 0
                reward_sum = 0

                while not done and not clear:
                    if np.random.rand(1) < e:
                        action = self.env.get_random_actions()
                    else:
                        action = np.argmax(mainDQN.predict(state))
                    next_state, reward, done, clear, max_x = self.env.step(action)

                    if done:
                        reward = -1500
                    if clear:
                        reward += 10000
                        done = True

                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > self.REPLAY_MEMORY:
                        self.replay_buffer.popleft()

                    state = next_state
                    step_count += 1

                    reward_sum += reward

                    if step_count > 100 and max_x < 202:
                        episode -= 1
                        train = False
                        print "pass"
                        break

                if len(self.replay_buffer) > 50 and train:
                    print("Episode: {}  steps: {}  max_x: {}  reward: {}".format(episode, step_count, max_x, reward_sum))

                    for idx in range(50):
                        #minibatch = random.sample(self.replay_buffer, int(len(self.replay_buffer) / 30))
                        minibatch = random.sample(self.replay_buffer, 30)
                        loss = self.replay_train(mainDQN, targetDQN, minibatch)
                    print("Loss: ", loss)
                    sess.run(copy_ops)

                #self.replay_buffer = deque()

                if episode % 100 == 0:
                    mainDQN.save(episode=episode)
                    targetDQN.save(episode=episode)
                    pass
                episode += 1



def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()

#lightdm