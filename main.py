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

        self.max_episodes = 2000001

        self.replay_buffer = deque()
        self.episode_buffer = deque()

        self.MAX_BUFFER_SIZE = 20000

        self.frame_action = 3
        self.training = True


    def async_training(self, sess, ops, ops_temp):
        while True:
            if len(self.episode_buffer) > 0:
                replay_buffer, episode, step_count, max_x, reward_sum = self.episode_buffer.popleft()
                for idx in range(5):
                    #minibatch = random.sample(replay_buffer, int(len(replay_buffer) * 0.8))
                    minibatch = replay_buffer
                    loss = self.replay_train(self.tempDQN, self.targetDQN, minibatch)
                print("Episode: {}  Loss: {}".format(episode, loss))

                sess.run(ops)
                sess.run(ops_temp)

                # 100 에피소드마다 저장한다
                if episode % 100 == 0:
                    self.mainDQN.save(episode=episode)
                    self.targetDQN.save(episode=episode)
                    self.tempDQN.save(episode=episode)

            elif not self.training:
                break

            else:
                time.sleep(1)



    def replay_train(self, mainDQN, targetDQN, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)
        step = 0
        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)
            #png.from_array(next_state, 'RGB').save('capture/' + str(step) + '.png')

            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + self.dis * targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]


            state = np.reshape(state, [self.input_size])
            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state])
            step += 1

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
            self.mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            self.targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")
            self.tempDQN = dqn.DQN(sess, self.input_size, self.output_size, name="temp")
            tf.global_variables_initializer().run()

            episode = 0
            try:
                self.mainDQN.restore(episode)
                self.targetDQN.restore(episode)
                self.tempDQN.restore(episode)
            except NotFoundError:
                print "save file not found"

            copy_ops = self.get_copy_var_ops()
            copy_ops_temp = self.get_copy_var_ops(dest_scope_name="main", src_scope_name="temp")
            copy_ops_temp2 = self.get_copy_var_ops(dest_scope_name="temp", src_scope_name="main")
            sess.run(copy_ops)
            sess.run(copy_ops_temp2)

            training_thread = threading.Thread(target=self.async_training, args=(sess, copy_ops, copy_ops_temp))
            training_thread.start()

            start_position = 0

            #REPLAY_MEMORY = self.get_memory_size(episode)
            while episode < self.max_episodes:
                e = max(0.05, min(0.7, 1. / ((episode / 50) + 1)))
                done = False
                clear = False
                step_count = 0
                state = self.env.reset(start_position=start_position)
                max_x = 0
                now_x = 0
                reward_sum = 0
                before_action = [0, 0, 0, 0, 0, 0]

                input_list = []

                hold_frame = 0
                before_max_x = 200

                step_reward = 0

                while not done and not clear:

                    if step_count % self.frame_action == 0:
                        if np.random.rand(1) < e:
                            action = self.env.get_random_actions()
                        else:
                            action = np.argmax(self.mainDQN.predict(state))
                            input_list.append(action)
                    else:
                        action = before_action

                    next_state, reward, done, clear, max_x, timeout, now_x = self.env.step(action)
                    #print state
                    step_reward += reward


                    if step_count % self.frame_action == self.frame_action-1 \
                            or done or timeout or clear:
                        if done and not timeout:
                            reward = -10000
                        if clear:
                            reward += 10000
                            done = True

                        self.replay_buffer.append((state, action, step_reward, next_state, done))
                        if len(self.replay_buffer) > self.MAX_BUFFER_SIZE:
                            self.replay_buffer.popleft()
                        step_reward = 0




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
                    #print next_state
                    #png.from_array(next_state, 'RGB').save('capture/' + str(step_count) + '.png')

                print("Buffer: {}  Episode: {}  steps: {}  max_x: {}  reward: {}".format(len(self.episode_buffer),
                                                                                         episode, step_count, max_x,
                                                                                         reward_sum))

                with open('input_log/input_' + str(episode), 'w') as fp:
                    fp.write(str(input_list))

                # 샘플링 하기에 작은 사이즈는 트레이닝 시키지 않는다
                if episode % 2 == 0 and len(self.replay_buffer) > 50:
                    self.episode_buffer.append((self.replay_buffer, episode, step_count, max_x, reward_sum))
                    if len(self.episode_buffer) > 0:
                        print 'buffer flush... plz wait...'
                        while len(self.episode_buffer) != 0:
                            time.sleep(1)
                    self.replay_buffer = deque()

                episode += 1

                # 죽은 경우 죽은 지점의 600픽셀 이전에서 살아나서 다시 시도한다
                '''
                if done and not timeout:
                    start_position = now_x - 800
                else:
                    start_position = 0
                '''

            # 에피소드가 끝나면 종료하지말고 버퍼에있는 트레이닝을 마친다
            self.training = False
            training_thread.join()



def main():
    env = Env()
    controller = AIControl(env)
    controller.control_start()


if __name__ == "__main__":
    main()

#lightdm