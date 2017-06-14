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
        self.output_size = 14

        #self.dis = 0.9
        self.dis = 0.9
        self.val = 0
        self.save_path = "./save/save_model"

        self.max_episodes = 20000001

        self.replay_buffer = deque()
        self.episode_buffer = deque()

        self.MAX_BUFFER_SIZE = 20000

        self.frame_action = 3
        self.training = True


    def async_training(self, sess, ops, ops_temp):
        step = 51
        epoch = 30
        batch_size = 2000
        while self.training:
            if len(self.episode_buffer) > 0:
                replay_buffer, episode, step_count, max_x, reward_sum = self.episode_buffer.popleft()
                replay_buffer = list(replay_buffer)
                for idx in range(epoch):
                    start_idx = 0
                    #batch = random.sample(replay_buffer, int(len(replay_buffer) * 0.2))
                    #batch = random.sample(replay_buffer, batch_size)
                    #_, _, _, _, batch_done = batch[-1]
                    #_, _, _, _, replay_done = replay_buffer[-1]
                    #if not batch_done and replay_done:
                    #    batch.append(replay_buffer[-1])
                    batch = replay_buffer
                    #loss = self.replay_train(self.tempDQN, self.targetDQN, batch)

                    loss = 0
                    learn_cnt = 0
                    while start_idx-batch_size < len(batch):
                        #minibatch = replay_buffer
                        minibatch = batch[start_idx:start_idx+batch_size]
                        if len(minibatch) == 0:
                            break
                        loss += self.replay_train(self.tempDQN, self.targetDQN, minibatch)[0]
                        start_idx += batch_size
                        learn_cnt += 1
                    #print("Step: {}  Loss: {}".format(idx, loss))

                    print("Step: {}-{}  Avg-Loss: {}".format(step, idx, loss/learn_cnt))
                '''
                for idx in range(100):
                    minibatch = random.sample(self.replay_buffer, int(len(self.replay_buffer) * 0.03))
                    #minibatch = replay_buffer
                    loss = self.replay_train(self.tempDQN, self.targetDQN, minibatch)
                print("Episode: {}  Loss: {}".format(step, loss))
                '''

                sess.run(ops)
                sess.run(ops_temp)

                # 100 에피소드마다 저장한다
                if step % 50 == 0:
                    self.mainDQN.save(episode=step)
                    self.targetDQN.save(episode=step)
                    self.tempDQN.save(episode=step)
                step += 1
            else:
                time.sleep(1)



    def replay_train(self, mainDQN, targetDQN, train_batch):
        x_stack = np.empty(0).reshape(0, self.input_size)
        y_stack = np.empty(0).reshape(0, self.output_size)

        for state, action, reward, next_state, done in train_batch:
            Q = mainDQN.predict(state)

            if done:
                Q[0, action] = reward
            else:
                aa = targetDQN.predict(next_state)[0, np.argmax(mainDQN.predict(next_state))]
                Q[0, action] = reward + self.dis * aa
                #print(targetDQN.predict(next_state), mainDQN.predict(next_state))
                #print("Action: {}  RealReward: {}  Reward: {}".format(action, reward, aa))

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

    def control_start(self):
        import dqn
        with tf.Session() as sess:
            self.mainDQN = dqn.DQN(sess, self.input_size, self.output_size, name="main")
            self.targetDQN = dqn.DQN(sess, self.input_size, self.output_size, name="target")
            self.tempDQN = dqn.DQN(sess, self.input_size, self.output_size, name="temp")
            tf.global_variables_initializer().run()

            episode = 50
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

            start_position = 500

            episode = 101
            #REPLAY_MEMORY = self.get_memory_size(episode)
            while episode < self.max_episodes:
                e = max(0.1, min(0.5, 1. / ((episode / 200) + 1)))
                #
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

                action_state = state
                while not done and not clear:

                    if step_count % self.frame_action == 0:
                        if np.random.rand(1) < e:
                            action = self.env.get_random_actions()
                        else:
                            action = np.argmax(self.mainDQN.predict(state))
                            input_list.append(action)
                        action_state = state
                    else:
                        action = before_action

                    next_state, reward, done, clear, max_x, timeout, now_x = self.env.step(action)
                    #print state
                    step_reward += reward

                    # 앞으로 나아가지 못하는 상황이 1000프레임 이상이면 종료하고 학습한다.
                    if now_x <= before_max_x:
                        hold_frame += 1
                        if hold_frame > 1000:
                            done = True
                    else:
                        hold_frame = 0
                        before_max_x = max_x


                    if step_count % self.frame_action == self.frame_action-1 \
                            or done or timeout or clear:
                        if done or timeout and not clear:
                            step_reward = -10000
                        if clear:
                            step_reward += 100000
                            done = True

                        step_reward /= 100.0

                        self.replay_buffer.append((action_state, action, step_reward, next_state, done))
                        if len(self.replay_buffer) > self.MAX_BUFFER_SIZE:
                            self.replay_buffer.popleft()
                        step_reward = 0

                    state = next_state
                    step_count += 1

                    reward_sum += reward
                    before_action = action


                    #print next_state
                    png.from_array(next_state, 'L').save('capture/' + str(step_count) + '.png')

                #print("Buffer: {}  Episode: {}  steps: {}  max_x: {}  reward: {}".format(len(self.episode_buffer),
                #                                                                         episode, step_count, max_x,
                #                                                                         reward_sum))

                with open('input_log/input_' + str(episode), 'w') as fp:
                    fp.write(str(input_list))

                '''
                # 샘플링 하기에 작은 사이즈는 트레이닝 시키지 않는다
                if episode % 3 == 0 and len(self.replay_buffer) > 50:
                    self.episode_buffer.append((self.replay_buffer, episode, step_count, max_x, reward_sum))
                    if len(self.episode_buffer) > 0:
                        print 'buffer flush... plz wait...'
                        while len(self.episode_buffer) != 0:
                            time.sleep(1)
                    self.replay_buffer = deque()
                '''

                #if len(self.replay_buffer) > self.MAX_BUFFER_SIZE:
                if episode % 10 == 0 and len(self.replay_buffer) > 300:
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