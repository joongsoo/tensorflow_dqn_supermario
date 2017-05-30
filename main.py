# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random

from collections import deque
from data.env import Env
from tensorflow.python.framework.errors_impl import NotFoundError
import time
import threading
import png



class AIControl:
    def __init__(self, env):
        self.env = env

        self.input_size = self.env.state_n
        self.output_size = 12

        #self.dis = 0.9
        self.dis = 0.9
        self.val = 0
        self.save_path = "./save/save_model"

        self.max_episodes = 2000

        self.replay_buffer = deque()
        self.episode_buffer = deque()

        self.START_BUFFER_SIZE = 400
        self.MAX_BUFFER_SIZE = 20000
        self.BUFFER_RATE = 2000
        self.W = (self.MAX_BUFFER_SIZE - self.START_BUFFER_SIZE) / float(self.max_episodes)

    def get_memory_size(self, episode):
        return 50000
        if episode > self.BUFFER_RATE:
            episode = self.BUFFER_RATE
        return self.W * episode + (self.START_BUFFER_SIZE - (self.START_BUFFER_SIZE - self.BUFFER_RATE))


    def async_training(self, sess, ops):
        while True:
            if len(self.episode_buffer) > 0:
                replay_buffer, episode, step_count, max_x, reward_sum, input_list = self.episode_buffer.popleft()
                print ''
                print("Episode: {}  steps: {}  max_x: {}  reward: {}".format(episode, step_count, max_x, reward_sum))
                for idx in range(10):
                    minibatch = random.sample(replay_buffer, int(len(replay_buffer) * 0.1))
                    loss = self.replay_train(self.mainDQN, self.targetDQN, minibatch)
                    print '.',
                print ''
                print("Loss: ", loss)
                sess.run(ops)

                with open('input_log/input_' + str(episode), 'w') as fp:
                    fp.write(str(input_list))

                # 50 에피소드마다 저장한다
                if episode % 50 == 0:
                    self.mainDQN.save(episode=episode)
                    self.targetDQN.save(episode=episode)
            else:
                time.sleep(1)
            pass



    def replay_train(self, mainDQN, targetDQN, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)
            predict = targetDQN.predict(next_state)

            if done:
                Q[0, action] = reward
            else:
                # 보상 + 미래에 받을 수 있는 보상의 최대값
                Q[0, action] = reward + self.dis * np.max(predict)

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
            self.mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            self.targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")
            tf.global_variables_initializer().run()

            episode = 0
            try:
                self.mainDQN.restore(episode)
                self.targetDQN.restore(episode)
            except NotFoundError:
                print "save file not found"

            copy_ops = self.get_copy_var_ops()
            sess.run(copy_ops)
            training_thread = threading.Thread(target=self.async_training, args=(sess, copy_ops))
            training_thread.start()

            start_position = 0

            #REPLAY_MEMORY = self.get_memory_size(episode)
            while episode < self.max_episodes:
                e = 1. / ((episode / 50) + 1)#min(0.5, 1. / ((episode / 50) + 1))
                done = False
                clear = False
                step_count = 0
                state = self.env.reset(start_position=start_position)
                max_x = 0
                now_x = 0
                reward_sum = 0
                REPLAY_MEMORY = self.get_memory_size(episode)
                before_action = [0, 0, 0, 0, 0, 0]

                input_list = []

                hold_frame = 0
                before_max_x = 200

                while not done and not clear:
                    if step_count % 2 == 0:
                        if np.random.rand(1) < e:
                            action = self.env.get_random_actions()
                        else:
                            action = np.argmax(self.mainDQN.predict(state))
                            input_list.append(action)
                    else:
                        action = before_action
                    next_state, reward, done, clear, max_x, timeout, now_x = self.env.step(action)

                    if done and not timeout:
                        reward = -500
                    if clear:
                        reward += 10000
                        done = True


                    self.replay_buffer.append((state, action, reward, next_state, done))
                    if len(self.replay_buffer) > REPLAY_MEMORY:
                        self.replay_buffer.popleft()

                    state = next_state
                    step_count += 1

                    reward_sum += reward
                    before_action = action

                    # 앞으로 나아가지 못하는 상황이 1000프레임 이상이면 종료하고 학습한다.
                    if now_x <= before_max_x:
                        hold_frame += 1
                        if hold_frame > 1000:
                            timeout = True
                            break
                    else:
                        hold_frame = 0
                        before_max_x = max_x

                    #png.from_array(next_state, 'RGB').save('capture/'+str(step_count) + '.png')


                # 샘플링 하기에 작은 사이즈는 트레이닝 시키지 않는다
                if step_count > 40:
                    self.episode_buffer.append((self.replay_buffer, episode, step_count, max_x, reward_sum, input_list))
                    '''
                    print ''
                    print("Episode: {}  steps: {}  max_x: {}  reward: {}".format(episode, step_count, max_x, reward_sum))
                    for idx in range(20):
                        minibatch = random.sample(self.replay_buffer, int(len(self.replay_buffer) * 0.05))
                        #minibatch = random.sample(self.replay_buffer, 30)
                        loss = self.replay_train(mainDQN, targetDQN, minibatch)
                        print '.',
                    print ''
                    print("Loss: ", loss)
                    sess.run(copy_ops)

                    with open('input_log/input_' + str(episode), 'w') as fp:
                        fp.write(str(input_list))
                    '''
                else:
                    episode -= 1

                self.replay_buffer = deque()

                episode += 1

                # 죽은 경우 죽은 지점의 200픽셀 이전에서 살아나서 다시 시도한다
                if done and not timeout:
                    start_position = now_x - 600
                else:
                    start_position = 0



def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()

#lightdm