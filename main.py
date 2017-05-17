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
        self.output_size = 2

        #self.dis = 0.9
        self.dis = 1.5
        self.REPLAY_MEMORY = 9000
        self.max_episodes = 1500
        self.replay_buffer = deque()
        self.val = 0
        self.save_path = "./save/save_model"

    def replay_train(self, mainDQN, targetDQN, train_batch):
        results = []
        for idx in range(len(zip(mainDQN, targetDQN))):
            x_stack = np.empty(0).reshape(0, self.input_size)
            y_stack = np.empty(0).reshape(0, self.output_size)
            for state, action, reward, next_state, done in train_batch:
                Q = mainDQN[idx].predict(state)
                # Q = [[]] 2차원
                # Q의 값은 0과 1 사이여야한다
                # linear regration
                if done:
                    Q[0, action] = reward
                else:
                    # dis를 늘리면 지금 한 행동으로 인해 미래에 얻는 보상이 현재 얻을 보상보다 값지다
                    Q[0, action] = reward + self.dis * np.max(targetDQN[idx].predict(next_state))

                state = np.reshape(state, [self.input_size])
                y_stack = np.vstack([y_stack, Q])
                x_stack = np.vstack([x_stack, state])
            results.append(mainDQN[idx].update(x_stack, y_stack))

        return results


    def get_copy_var_ops(self, dest_dqn_arr, src_dqn_arr):
        op_holders = []
        for dest, src in zip(dest_dqn_arr, src_dqn_arr):
            op_holder = []

            src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src.net_name)
            dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest.net_name)

            for src_var, dest_var in zip(src_vars, dest_vars):
                op_holder.append(dest_var.assign(src_var.value()))
            op_holders.append(op_holder)

        return op_holders

    def control_start(self):
        import dqn
        with tf.Session() as sess:
            mainDQN = [
                dqn.DQN(sess, self.input_size, self.output_size, name="main0"),
                dqn.DQN(sess, self.input_size, self.output_size, name="main1"),
                dqn.DQN(sess, self.input_size, self.output_size, name="main2"),
                dqn.DQN(sess, self.input_size, self.output_size, name="main3"),
                dqn.DQN(sess, self.input_size, self.output_size, name="main4"),
                dqn.DQN(sess, self.input_size, self.output_size, name="main5")
            ]

            targetDQN = [
                dqn.DQN(sess, self.input_size, self.output_size, name="target1"),
                dqn.DQN(sess, self.input_size, self.output_size, name="target2"),
                dqn.DQN(sess, self.input_size, self.output_size, name="target3"),
                dqn.DQN(sess, self.input_size, self.output_size, name="target4"),
                dqn.DQN(sess, self.input_size, self.output_size, name="target5"),
                dqn.DQN(sess, self.input_size, self.output_size, name="target6")
            ]

            tf.global_variables_initializer().run()

            try:
                for main, target in zip(mainDQN, targetDQN):
                    main.restore()
                    target.restore()
            except NotFoundError:
                pass

            copy_ops = self.get_copy_var_ops(targetDQN, mainDQN)
            sess.run(copy_ops)

            for episode in range(1000, self.max_episodes):
                e = 1. / ((episode / 10) + 1)
                done = False
                step_count = 0
                state = self.env.reset()
                max_x = 0

                while not done:
                    if np.random.rand(1) < e:
                        action = self.env.get_random_actions()
                    else:
                        action = []
                        for dqn in mainDQN:
                            pre = np.argmax(dqn.predict(state))

                            action.append(pre)

                        print("action", action)

                    next_state, reward, done, max_x = self.env.step(action)

                    if done:
                        reward = -10000

                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > self.REPLAY_MEMORY:
                        self.replay_buffer.popleft()

                    state = next_state
                    step_count += 1


                print("Episode: {}  steps: {}  max_x: {}".format(episode, step_count, max_x))

                if step_count % 10 == 0:
                    for main, target in zip(mainDQN, targetDQN):
                        main.save()
                        target.save()

                print step_count
                print("len buffer ", len(self.replay_buffer))
                #minibatch = random.sample(self.replay_buffer, int(len(self.replay_buffer) / 100))
                loss = self.replay_train(mainDQN, targetDQN, self.replay_buffer)

                print("Loss: ", loss)
                sess.run(copy_ops)

def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()
